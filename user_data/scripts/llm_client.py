"""
Provider-agnostic chat completion wrapper.

Two providers supported today:

  anthropic         official Anthropic SDK, Claude family
  deepseek          OpenAI SDK pointed at https://api.deepseek.com
                    (DeepSeek's API is OpenAI-compatible)

Adding more OpenAI-compatible providers (OpenRouter, Together, Groq, Fireworks)
is a one-line entry in PROVIDER_DEFAULTS — no new client class needed.

The default provider comes from the LLM_PROVIDER env var (default: 'deepseek').
Per-call overrides win: a caller can force a specific provider for a specific
task without changing the default. This lets us mix-and-match later — e.g.
DeepSeek for the cheap generator + reflector, Claude for the more nuanced
critic — without touching the call sites.

Fallback: if the primary provider raises (network blip, rate limit, etc.),
we retry once on the fallback provider (default: 'anthropic') before raising.
Set fallback_provider=None to disable.

Shape:
  chat_completion(messages, system=..., model=..., max_tokens=..., provider=...)
  → str  (the assistant's reply text)

`messages` is OpenAI-shaped — list of {role, content} dicts. The wrapper
handles the system-prompt difference between providers (Anthropic takes
`system` as a top-level kwarg; OpenAI-compat takes it as the first message
with role='system').
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("llm_client")


# ---------------------------------------------------------------------------
# Provider config — single source of truth for model defaults, env vars,
# and base URLs. To add a provider, append one entry here.
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS: dict[str, dict] = {
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "kind": "anthropic",
    },
    "deepseek": {
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "kind": "openai_compat",
    },
    # Examples of how to add more providers later (uncomment when needed):
    # "openrouter": {
    #     "model": "openai/gpt-4o",
    #     "api_key_env": "OPENROUTER_API_KEY",
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "kind": "openai_compat",
    # },
}


def default_provider() -> str:
    """The provider selected by the LLM_PROVIDER env var (default: deepseek).
    Read at call time, not import time, so tests can monkeypatch."""
    return os.environ.get("LLM_PROVIDER", "deepseek").lower()


# Default request timeout. A DeepSeek V4 Pro reasoning-model generation
# typically takes 30-180s; 300s leaves comfortable headroom while still
# catching the silent-stall failure mode (trials #4 and #6 both hung for
# 2+ hours waiting on a stream body that never arrived). Override via
# LLM_REQUEST_TIMEOUT_SECONDS.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 300


def request_timeout_seconds() -> float:
    """Per-request HTTP timeout passed to the underlying SDK clients.
    Read at call time so tests/operators can override at runtime."""
    raw = os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS")
    if not raw:
        return float(_DEFAULT_REQUEST_TIMEOUT_SECONDS)
    try:
        val = float(raw)
    except ValueError:
        log.warning(
            f"LLM_REQUEST_TIMEOUT_SECONDS={raw!r} is not a number; "
            f"using default {_DEFAULT_REQUEST_TIMEOUT_SECONDS}s"
        )
        return float(_DEFAULT_REQUEST_TIMEOUT_SECONDS)
    if val <= 0:
        log.warning(
            f"LLM_REQUEST_TIMEOUT_SECONDS={val} must be positive; "
            f"using default {_DEFAULT_REQUEST_TIMEOUT_SECONDS}s"
        )
        return float(_DEFAULT_REQUEST_TIMEOUT_SECONDS)
    return val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat_completion(
    messages: list[dict],
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    provider: str | None = None,
    fallback_provider: str | None = "anthropic",
) -> str:
    """Call the chosen LLM and return the assistant reply as a string.

    messages   OpenAI-shaped: [{"role": "user", "content": "..."}, ...]
    system     optional system prompt (handled per-provider — Anthropic gets
               it as a top-level kwarg; OpenAI-compat gets it prepended to
               the messages list as a 'system' role message)
    model      explicit model ID; otherwise uses the provider default
    provider   'anthropic' | 'deepseek' | ... ; otherwise default_provider()
    fallback_provider  retried once if primary fails; pass None to disable

    Raises if both providers fail (or if fallback is disabled and primary
    fails). The original primary exception is chained so the caller can see
    the root cause even when the fallback also failed.
    """
    primary = (provider or default_provider()).lower()
    try:
        return _call_provider(primary, messages, system, model, max_tokens)
    except Exception as e:
        if fallback_provider and fallback_provider != primary:
            log.warning(
                f"primary provider {primary!r} failed ({type(e).__name__}: {e}); "
                f"falling back to {fallback_provider!r}"
            )
            try:
                return _call_provider(fallback_provider, messages, system, model, max_tokens)
            except Exception as fb_err:
                raise RuntimeError(
                    f"both {primary!r} and fallback {fallback_provider!r} failed: "
                    f"primary={type(e).__name__}: {e}; "
                    f"fallback={type(fb_err).__name__}: {fb_err}"
                ) from e
        raise


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def _call_provider(provider, messages, system, model, max_tokens):
    cfg = PROVIDER_DEFAULTS.get(provider)
    if not cfg:
        raise ValueError(
            f"unknown LLM provider {provider!r}; "
            f"add it to PROVIDER_DEFAULTS in llm_client.py"
        )
    kind = cfg["kind"]
    if kind == "anthropic":
        return _call_anthropic(cfg, messages, system, model, max_tokens)
    if kind == "openai_compat":
        return _call_openai_compat(cfg, messages, system, model, max_tokens)
    raise ValueError(f"unknown provider kind {kind!r} for {provider!r}")


def _call_anthropic(cfg, messages, system, model, max_tokens):
    import anthropic  # lazy import — keeps DeepSeek-only callers from needing it

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} not set")

    client = anthropic.Anthropic(api_key=api_key, timeout=request_timeout_seconds())
    kwargs = {
        "model": model or cfg["model"],
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)
    # Anthropic responses are a list of content blocks; we want the text of
    # the first one (the model only emits one text block in our usage).
    return resp.content[0].text


def _call_openai_compat(cfg, messages, system, model, max_tokens):
    import openai  # lazy import

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['api_key_env']} not set")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=cfg["base_url"],
        timeout=request_timeout_seconds(),
    )

    # OpenAI-format: system is the first message in the list, not a separate kwarg.
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    resp = client.chat.completions.create(
        model=model or cfg["model"],
        max_tokens=max_tokens,
        messages=full_messages,
    )
    return resp.choices[0].message.content
