"""Tests for the provider-agnostic LLM wrapper (llm_client.chat_completion)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# default_provider — env-driven, read at call time
# ---------------------------------------------------------------------------

def test_default_provider_reads_env_at_call_time(monkeypatch):
    from llm_client import default_provider
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert default_provider() == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "DeepSeek")  # case-insensitive
    assert default_provider() == "deepseek"


def test_default_provider_falls_back_to_deepseek_when_env_unset(monkeypatch):
    from llm_client import default_provider
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert default_provider() == "deepseek"


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

def _patched_anthropic(text="hello from claude"):
    """Build a MagicMock standing in for the anthropic module."""
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=text)]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    return fake_anthropic, fake_client


def test_anthropic_call_returns_text(monkeypatch):
    from llm_client import chat_completion
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fake_anth, fake_client = _patched_anthropic("reply")

    with patch.dict("sys.modules", {"anthropic": fake_anth}):
        out = chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="anthropic",
        )
    assert out == "reply"
    # Anthropic gets system as separate kwarg, not in messages
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert "system" not in call_kwargs  # we didn't pass one


def test_anthropic_call_passes_system_as_kwarg(monkeypatch):
    from llm_client import chat_completion
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fake_anth, fake_client = _patched_anthropic()

    with patch.dict("sys.modules", {"anthropic": fake_anth}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            system="SYSTEM_PROMPT",
            provider="anthropic",
            fallback_provider=None,
        )
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "SYSTEM_PROMPT"


def test_anthropic_raises_without_api_key(monkeypatch):
    from llm_client import chat_completion
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises((RuntimeError, Exception)) as exc:
        chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="anthropic", fallback_provider=None,
        )
    assert "ANTHROPIC_API_KEY" in str(exc.value)


# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible) provider
# ---------------------------------------------------------------------------

def _patched_openai(text="hello from deepseek"):
    """Build a MagicMock standing in for the openai module."""
    fake_choice = MagicMock()
    fake_choice.message = MagicMock(content=text)
    fake_resp = MagicMock(choices=[fake_choice])
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return fake_openai, fake_client


def test_deepseek_call_returns_text(monkeypatch):
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    fake_openai, fake_client = _patched_openai("from deepseek")

    with patch.dict("sys.modules", {"openai": fake_openai}):
        out = chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="deepseek",
        )
    assert out == "from deepseek"


def test_deepseek_call_uses_custom_base_url(monkeypatch):
    """The OpenAI client must be instantiated with DeepSeek's base_url."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    fake_openai, _fake_client = _patched_openai()

    with patch.dict("sys.modules", {"openai": fake_openai}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="deepseek", fallback_provider=None,
        )
    init_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert init_kwargs["base_url"] == "https://api.deepseek.com"
    assert init_kwargs["api_key"] == "test"


def test_deepseek_default_model_is_v4_pro(monkeypatch):
    """When no model override is passed, DeepSeek calls use deepseek-v4-pro."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    fake_openai, fake_client = _patched_openai()

    with patch.dict("sys.modules", {"openai": fake_openai}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="deepseek", fallback_provider=None,
        )
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-v4-pro"


def test_deepseek_call_prepends_system_message(monkeypatch):
    """OpenAI-format puts system as the first message in the list, not a
    separate kwarg. Wrapper must translate."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    fake_openai, fake_client = _patched_openai()

    with patch.dict("sys.modules", {"openai": fake_openai}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            system="SYS",
            provider="deepseek", fallback_provider=None,
        )
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]


def test_explicit_model_override_wins_over_provider_default(monkeypatch):
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    fake_openai, fake_client = _patched_openai()

    with patch.dict("sys.modules", {"openai": fake_openai}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            model="custom-model-id",
            provider="deepseek", fallback_provider=None,
        )
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "custom-model-id"


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

def test_fallback_kicks_in_when_primary_fails(monkeypatch):
    """If DeepSeek errors and fallback is Anthropic, the response should come
    from the fallback."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    fake_openai, fake_openai_client = _patched_openai()
    fake_openai_client.chat.completions.create.side_effect = Exception("ds down")
    fake_anth, fake_anth_client = _patched_anthropic("from fallback")

    with patch.dict("sys.modules", {"openai": fake_openai, "anthropic": fake_anth}):
        out = chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="deepseek", fallback_provider="anthropic",
        )
    assert out == "from fallback"


def test_fallback_can_be_disabled(monkeypatch):
    """fallback_provider=None means primary errors propagate."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    fake_openai, fake_openai_client = _patched_openai()
    fake_openai_client.chat.completions.create.side_effect = Exception("boom")

    with patch.dict("sys.modules", {"openai": fake_openai}):
        with pytest.raises(Exception, match="boom"):
            chat_completion(
                [{"role": "user", "content": "hi"}],
                provider="deepseek", fallback_provider=None,
            )


def test_fallback_not_attempted_when_matches_primary(monkeypatch):
    """If fallback IS the primary (caller asked for anthropic with
    fallback=anthropic), don't double-call."""
    from llm_client import chat_completion
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fake_anth, fake_anth_client = _patched_anthropic()
    fake_anth_client.messages.create.side_effect = Exception("boom")

    with patch.dict("sys.modules", {"anthropic": fake_anth}):
        with pytest.raises(Exception, match="boom"):
            chat_completion(
                [{"role": "user", "content": "hi"}],
                provider="anthropic", fallback_provider="anthropic",
            )
    # Only one call total — no fallback retry
    assert fake_anth_client.messages.create.call_count == 1


def test_double_failure_chains_original_exception(monkeypatch):
    """When BOTH providers fail, the raised RuntimeError should mention
    both and chain the primary exception for debuggability."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    fake_openai, fake_openai_client = _patched_openai()
    fake_openai_client.chat.completions.create.side_effect = Exception("ds down")
    fake_anth, fake_anth_client = _patched_anthropic()
    fake_anth_client.messages.create.side_effect = Exception("anth down")

    with patch.dict("sys.modules", {"openai": fake_openai, "anthropic": fake_anth}):
        with pytest.raises(RuntimeError) as exc:
            chat_completion(
                [{"role": "user", "content": "hi"}],
                provider="deepseek", fallback_provider="anthropic",
            )
    assert "ds down" in str(exc.value)
    assert "anth down" in str(exc.value)


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------

def test_unknown_provider_raises(monkeypatch):
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="not_a_real_provider", fallback_provider=None,
        )


# ---------------------------------------------------------------------------
# Request timeout
#
# Trials #4 and #6 both hung for 2+ hours waiting on a DeepSeek HTTP
# stream body that never arrived. Without a per-request timeout, a stalled
# connection burns the entire trial. These tests pin the timeout plumbing
# so a future SDK upgrade or refactor can't silently drop it.
# ---------------------------------------------------------------------------

def test_default_request_timeout_when_env_unset(monkeypatch):
    from llm_client import request_timeout_seconds
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT_SECONDS", raising=False)
    assert request_timeout_seconds() == 300.0


def test_request_timeout_overridable_via_env(monkeypatch):
    from llm_client import request_timeout_seconds
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")
    assert request_timeout_seconds() == 45.0


def test_request_timeout_ignores_garbage_env(monkeypatch):
    from llm_client import request_timeout_seconds
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "not-a-number")
    assert request_timeout_seconds() == 300.0


def test_request_timeout_ignores_non_positive_env(monkeypatch):
    from llm_client import request_timeout_seconds
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "-5")
    assert request_timeout_seconds() == 300.0
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "0")
    assert request_timeout_seconds() == 300.0


def test_anthropic_client_receives_timeout(monkeypatch):
    """The Anthropic SDK client must be instantiated with our timeout so a
    stalled HTTP read can't hang the whole trial."""
    from llm_client import chat_completion
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "120")
    fake_anth, _ = _patched_anthropic()

    with patch.dict("sys.modules", {"anthropic": fake_anth}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="anthropic", fallback_provider=None,
        )
    init_kwargs = fake_anth.Anthropic.call_args.kwargs
    assert init_kwargs["timeout"] == 120.0


def test_openai_client_receives_timeout(monkeypatch):
    """Same plumbing for the OpenAI-compat client (used by DeepSeek)."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "90")
    fake_openai, _ = _patched_openai()

    with patch.dict("sys.modules", {"openai": fake_openai}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="deepseek", fallback_provider=None,
        )
    init_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert init_kwargs["timeout"] == 90.0


def test_timeout_uses_default_when_env_unset(monkeypatch):
    """End-to-end: with LLM_REQUEST_TIMEOUT_SECONDS unset, the SDK client
    still gets a finite timeout (the 300s default), not None."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT_SECONDS", raising=False)
    fake_openai, _ = _patched_openai()

    with patch.dict("sys.modules", {"openai": fake_openai}):
        chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="deepseek", fallback_provider=None,
        )
    init_kwargs = fake_openai.OpenAI.call_args.kwargs
    assert init_kwargs["timeout"] == 300.0
    assert init_kwargs["timeout"] is not None


def test_timeout_failure_triggers_fallback(monkeypatch):
    """When primary times out (any exception from the SDK), the wrapper's
    existing fallback path must still fire — so a hung DeepSeek doesn't
    just propagate as a bare TimeoutError to the orchestrator."""
    from llm_client import chat_completion
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    fake_openai, fake_openai_client = _patched_openai()
    fake_openai_client.chat.completions.create.side_effect = TimeoutError(
        "stream stalled at byte 0"
    )
    fake_anth, _ = _patched_anthropic("rescued by claude")

    with patch.dict("sys.modules", {"openai": fake_openai, "anthropic": fake_anth}):
        out = chat_completion(
            [{"role": "user", "content": "hi"}],
            provider="deepseek", fallback_provider="anthropic",
        )
    assert out == "rescued by claude"
