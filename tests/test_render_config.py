"""Tests for user_data/scripts/render_config.py.

The script materializes Freqtrade configs from committed templates at
container startup. Secrets stay in env vars, never on tracked disk.
These tests pin the substitution + escape behavior so a future edit
can't silently introduce a leak by skipping placeholders or stripping
the escape pass.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


def test_render_substitutes_a_single_placeholder():
    from render_config import render
    out = render('"key": "${OKX_API_KEY}"', env={"OKX_API_KEY": "abc123"})
    assert out == '"key": "abc123"'


def test_render_substitutes_multiple_placeholders():
    from render_config import render
    out = render(
        '"k": "${OKX_API_KEY}", "s": "${OKX_API_SECRET}"',
        env={"OKX_API_KEY": "k", "OKX_API_SECRET": "s"},
    )
    assert out == '"k": "k", "s": "s"'


def test_render_leaves_non_placeholder_dollar_signs_alone():
    """Lowercase or shell-style references shouldn't match — only the
    canonical ${UPPER_SNAKE} form gets substituted."""
    from render_config import render
    text = 'price is $100 and ${lower_case} and $$escaped'
    assert render(text, env={}) == text


def test_render_raises_when_env_var_missing():
    """Missing env var must fail fast with the var name in the error — no
    silent writes of the literal ${VAR} into the rendered config."""
    from render_config import render
    with pytest.raises(KeyError) as exc:
        render('"key": "${OKX_API_KEY}"', env={})
    assert exc.value.args[0] == "OKX_API_KEY"


def test_render_escapes_backslashes():
    """A password containing a backslash must not break JSON parsing."""
    from render_config import render
    out = render('"password": "${P}"', env={"P": "weird\\path"})
    assert out == '"password": "weird\\\\path"'
    # Verify the rendered output is valid JSON-string content
    assert json.loads("{" + out + "}")["password"] == "weird\\path"


def test_render_escapes_double_quotes():
    """A password containing a double-quote must be JSON-escaped so the
    rendered config doesn't have a stray closing quote."""
    from render_config import render
    out = render('"password": "${P}"', env={"P": 'has"quote'})
    assert out == '"password": "has\\"quote"'
    assert json.loads("{" + out + "}")["password"] == 'has"quote'


def test_render_with_real_template_produces_valid_json(tmp_path):
    """End-to-end: feed the real committed momentum template through with
    a full env, and the output must parse cleanly as JSON. Catches any
    drift in the template that would break Freqtrade at startup."""
    from render_config import render
    template = (ROOT / "user_data" / "configs" / "config-momentum.json.template").read_text()
    rendered = render(template, env={
        "OKX_API_KEY": "test-key",
        "OKX_API_SECRET": "test-secret",
        "OKX_API_PASSPHRASE": "test-passphrase",
        "FT_MOMENTUM_JWT_SECRET": "test-jwt",
        "FT_MOMENTUM_API_PASSWORD": "test-rest-pw",
    })
    parsed = json.loads(rendered)
    assert parsed["exchange"]["key"] == "test-key"
    assert parsed["exchange"]["secret"] == "test-secret"
    assert parsed["exchange"]["password"] == "test-passphrase"
    assert parsed["api_server"]["password"] == "test-rest-pw"
    assert parsed["api_server"]["jwt_secret_key"] == "test-jwt"
    # Sanity: the bot_name should NOT have been substituted (no ${} in it)
    assert parsed["bot_name"] == "ft-momentum"


def test_render_sweep_template_produces_valid_json():
    """Same check for the sweep template."""
    from render_config import render
    template = (ROOT / "user_data" / "configs" / "config-sweep.json.template").read_text()
    rendered = render(template, env={
        "OKX_API_KEY": "k", "OKX_API_SECRET": "s", "OKX_API_PASSPHRASE": "p",
        "FT_SWEEP_JWT_SECRET": "j", "FT_SWEEP_API_PASSWORD": "r",
    })
    parsed = json.loads(rendered)
    assert parsed["bot_name"] == "ft-sweep"
    assert parsed["exchange"]["key"] == "k"
    assert parsed["api_server"]["password"] == "r"


def test_main_writes_output_file(tmp_path, monkeypatch):
    """The CLI entrypoint should produce the output file when env is set."""
    from render_config import main
    template = tmp_path / "in.tmpl"
    template.write_text('"k": "${MY_VAR}"')
    out = tmp_path / "out.json"
    monkeypatch.setenv("MY_VAR", "abc")

    rc = main([str(template.parent / "render"), str(template), str(out)])
    assert rc == 0
    assert out.read_text() == '"k": "abc"'


def test_main_exits_nonzero_when_env_missing(tmp_path, monkeypatch):
    """Missing env should yield exit 1 and the output file should NOT
    exist (don't leave a half-written config behind)."""
    from render_config import main
    template = tmp_path / "in.tmpl"
    template.write_text('"k": "${MISSING}"')
    out = tmp_path / "out.json"
    monkeypatch.delenv("MISSING", raising=False)

    rc = main(["render_config", str(template), str(out)])
    assert rc == 1
    assert not out.exists()


def test_main_exits_2_on_arg_count_mismatch():
    from render_config import main
    assert main(["render_config"]) == 2
    assert main(["render_config", "only_one_arg"]) == 2
