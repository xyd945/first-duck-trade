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


def test_render_raises_when_env_var_empty_string():
    """Codex finding: ${OKX_API_KEY:-} in docker-compose sets the var to ""
    when the host env doesn't have it. The previous `var not in env` check
    passed and rendered `"key": ""` into the config — freqtrade then failed
    at OKX auth time with a confusing message. Treat empty as missing."""
    from render_config import render
    with pytest.raises(KeyError) as exc:
        render('"key": "${OKX_API_KEY}"', env={"OKX_API_KEY": ""})
    assert exc.value.args[0] == "OKX_API_KEY"


def test_render_raises_when_env_var_whitespace_only():
    """A misformatted .env line like `OKX_API_KEY=   ` should also fail
    rather than rendering `"key": "   "` and producing the same auth error."""
    from render_config import render
    with pytest.raises(KeyError):
        render('"key": "${OKX_API_KEY}"', env={"OKX_API_KEY": "   \t  "})


def test_render_accepts_value_that_starts_with_whitespace():
    """A value with leading/trailing whitespace but non-whitespace content
    should pass through unmodified (we don't trim, we just check empty).
    This guards a future overzealous .strip() that would silently change
    a credential value before storing it."""
    from render_config import render
    out = render('"k": "${V}"', env={"V": " abc "})
    assert out == '"k": " abc "'


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


def test_main_writes_output_with_owner_only_mode(tmp_path, monkeypatch):
    """The rendered config holds OKX secrets — file mode must be 0600 so
    only the freqtrade user can read it inside the container. Verified on
    POSIX. Skipped on systems that don't honor POSIX permissions (e.g.
    Windows) since the deployment target is always Linux."""
    import os, stat
    if os.name != "posix":
        pytest.skip("permissions only relevant on POSIX")

    from render_config import main
    template = tmp_path / "in.tmpl"
    template.write_text('"k": "${V}"')
    out = tmp_path / "out.json"
    monkeypatch.setenv("V", "abc")

    assert main(["render_config", str(template), str(out)]) == 0
    mode = stat.S_IMODE(out.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_main_chmod_strips_world_readable_bit_on_overwrite(tmp_path, monkeypatch):
    """If an earlier render left a too-permissive file (e.g. someone ran
    chmod 0644 manually for debugging), the next render must tighten it
    back. The chmod call after write enforces this."""
    import os, stat
    if os.name != "posix":
        pytest.skip("permissions only relevant on POSIX")

    from render_config import main
    template = tmp_path / "in.tmpl"
    template.write_text('"k": "${V}"')
    out = tmp_path / "out.json"
    out.write_text("placeholder")
    os.chmod(out, 0o644)  # simulate prior loose permissions
    monkeypatch.setenv("V", "abc")

    assert main(["render_config", str(template), str(out)]) == 0
    mode = stat.S_IMODE(out.stat().st_mode)
    assert mode == 0o600, f"expected 0600 after render, got {oct(mode)}"


# ---------------------------------------------------------------------------
# Source-level guardrails: docker-compose.yml must render to /tmp so
# rendered configs never land on the host bind mount.
# ---------------------------------------------------------------------------

def test_compose_renders_freqtrade_configs_to_tmp():
    """Codex finding: the previous render target was inside user_data/configs/,
    which is host-bind-mounted — so cleartext secrets landed on host disk
    anyway. The render target must be /tmp inside the container (ephemeral,
    not mounted) for both ft-sweep and ft-momentum."""
    compose_text = (ROOT / "docker-compose.yml").read_text()

    # Each freqtrade service should have both:
    #  - render output to /tmp/config-<name>.json
    #  - freqtrade --config pointing at the same /tmp path
    for service_name in ("momentum", "sweep"):
        render_marker = f"/tmp/config-{service_name}.json"
        config_marker = f"--config /tmp/config-{service_name}.json"
        assert render_marker in compose_text, (
            f"docker-compose.yml does not render config-{service_name} to /tmp; "
            f"the bind-mounted user_data path would leak secrets to host disk"
        )
        assert config_marker in compose_text, (
            f"docker-compose.yml does not point freqtrade at the /tmp render "
            f"for {service_name}; rendered path and freqtrade --config must match"
        )

    # And the OLD path must be gone — easy regression sentinel.
    for stale in ("user_data/configs/config-momentum.json",
                  "user_data/configs/config-sweep.json"):
        # Allowed to appear as TEMPLATE source; rejected as render destination.
        for line in compose_text.splitlines():
            if stale in line and "template" not in line and "#" not in line.lstrip()[:1]:
                # Allow inline references in comments by checking comment-only lines
                pytest.fail(
                    f"docker-compose.yml still references {stale!r} as a "
                    f"non-template path; render must target /tmp now"
                )
