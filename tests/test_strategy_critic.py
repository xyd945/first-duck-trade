"""Tests for R5: strategy critic — JSON parsing + generator wiring."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# _parse_verdict_json
# ---------------------------------------------------------------------------

PASS_JSON = '{"verdict": "PASS", "summary": "looks fine", "issues": []}'

REJECT_JSON = """{
  "verdict": "REJECT",
  "summary": "8-condition AND in entry will produce 0 trades",
  "issues": [
    {"severity": "high", "category": "overfit",
     "description": "populate_entry_trend has 8 AND-joined filters"},
    {"severity": "medium", "category": "nan-guard",
     "description": "dataframe['vix'] used without fillna"}
  ]
}"""


def test_parse_pass_verdict():
    from strategy_critic import _parse_verdict_json
    out = _parse_verdict_json(PASS_JSON)
    assert out["verdict"] == "PASS"
    assert out["summary"] == "looks fine"
    assert out["issues"] == []


def test_parse_reject_with_issues():
    from strategy_critic import _parse_verdict_json
    out = _parse_verdict_json(REJECT_JSON)
    assert out["verdict"] == "REJECT"
    assert len(out["issues"]) == 2
    assert out["issues"][0]["severity"] == "high"


def test_parse_strips_markdown_fences():
    """Critic was told no fences but might add them anyway. Be forgiving."""
    from strategy_critic import _parse_verdict_json
    fenced = "```json\n" + PASS_JSON + "\n```"
    out = _parse_verdict_json(fenced)
    assert out["verdict"] == "PASS"


def test_parse_normalizes_unknown_verdict_to_pass():
    """If the LLM invents a verdict like MAYBE, treat as PASS (non-blocking)."""
    from strategy_critic import _parse_verdict_json
    weird = '{"verdict": "MAYBE", "summary": "?", "issues": []}'
    out = _parse_verdict_json(weird)
    assert out["verdict"] == "PASS"


def test_parse_handles_no_json_in_text():
    from strategy_critic import _parse_verdict_json
    out = _parse_verdict_json("This strategy looks fine to me, no JSON here.")
    assert out["verdict"] == "PASS"  # synthetic fallback
    assert "error" in out


def test_parse_handles_malformed_json():
    from strategy_critic import _parse_verdict_json
    out = _parse_verdict_json('{"verdict": "REJECT", "issues": [trailing comma,]}')
    assert out["verdict"] == "PASS"  # synthetic fallback
    assert out["error"].startswith("json_decode_error") or out["error"] == "json_decode_error"


def test_parse_handles_text_around_json():
    """Critic might prefix with explanation despite instructions."""
    from strategy_critic import _parse_verdict_json
    out = _parse_verdict_json("Here is my review:\n" + REJECT_JSON + "\n\nThat's all.")
    assert out["verdict"] == "REJECT"
    assert len(out["issues"]) == 2


def test_parse_handles_unbalanced_braces():
    from strategy_critic import _parse_verdict_json
    out = _parse_verdict_json('{"verdict": "REJECT", "issues": [{"severity": "high"]')
    assert out["verdict"] == "PASS"  # fallback
    assert "error" in out


# ---------------------------------------------------------------------------
# critic_review — mocked Anthropic client
# ---------------------------------------------------------------------------

def test_critic_review_returns_pass_when_no_api_key(monkeypatch):
    from strategy_critic import critic_review
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = critic_review("class Foo: pass")
    assert out["verdict"] == "PASS"
    assert out["error"] == "missing_api_key"


def test_critic_review_returns_pass_on_api_exception(monkeypatch):
    from strategy_critic import critic_review
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    fake_anthropic = MagicMock()
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = Exception("network down")
    fake_anthropic.Anthropic.return_value = fake_client

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = critic_review("class Foo: pass")
    assert out["verdict"] == "PASS"
    assert "network down" in out["error"]


def test_critic_review_parses_real_response(monkeypatch):
    from strategy_critic import critic_review
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=REJECT_JSON)]

    fake_anthropic = MagicMock()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    fake_anthropic.Anthropic.return_value = fake_client

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        out = critic_review("class Foo: pass")
    assert out["verdict"] == "REJECT"
    assert len(out["issues"]) == 2


# ---------------------------------------------------------------------------
# format_critic_feedback — for retry prompts
# ---------------------------------------------------------------------------

def test_format_feedback_includes_each_issue():
    from strategy_critic import format_critic_feedback
    critic = {
        "verdict": "REJECT",
        "summary": "too many filters",
        "issues": [
            {"severity": "high", "category": "overfit", "description": "8 ANDs"},
            {"severity": "medium", "category": "nan-guard", "description": "no fillna on vix"},
        ],
    }
    out = format_critic_feedback(critic)
    assert "REJECT" in out
    assert "too many filters" in out
    assert "8 ANDs" in out
    assert "no fillna on vix" in out


def test_format_feedback_empty_issues():
    from strategy_critic import format_critic_feedback
    out = format_critic_feedback({"verdict": "PASS", "summary": "ok", "issues": []})
    assert out == "ok"
