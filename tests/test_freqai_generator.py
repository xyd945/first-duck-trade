"""Tests for the LLM-proposed FreqAI spec loop (issue #47, Phase 3a).

Covers:
  - prompt/validator anti-drift: every whitelisted feature, model param,
    and bound appears in the system prompt, and the descriptions map is in
    exact 1:1 sync with the feature library
  - propose_freqai_specs: happy path, validator-rejection retry with
    feedback, unparseable-reply retry, name collision suffixing, and that
    factory-owned fields (spec_type, generation_id) override the LLM's
  - failure-memory spec_type filtering (rule prompts don't see ML failures
    and vice versa)
  - report/overview plumbing
  - orchestrator weekly hookup is env-gated and off by default
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BASE = Path(__file__).parent.parent / "user_data"
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE))

import freqai_generator as fg  # noqa: E402


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    import strategy_registry as sr

    monkeypatch.setattr(sr, "DB_PATH", tmp_path / "test_registry.db")
    monkeypatch.setattr(sr, "REFLECTIONS_DIR", tmp_path / "reflections")
    sr.init_db()
    return sr


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    import freqai_spec as fs

    monkeypatch.setattr(fs, "CANDIDATES_DIR", tmp_path / "candidates")
    monkeypatch.setattr(fs, "FREQAI_CONFIG_DIR", tmp_path / "configs")
    return fs


@pytest.fixture
def valid_spec():
    return json.loads((BASE / "freqai_specs" / "baseline_lgbm.json").read_text())


def _llm_reply(spec: dict) -> str:
    return json.dumps(spec)


# ---------------------------------------------------------------------------
# Prompt <-> validator anti-drift
# ---------------------------------------------------------------------------

def test_feature_descriptions_in_sync_with_library():
    from indicators.freqai_features import ALL_FEATURE_KEYS

    assert set(fg.FEATURE_DESCRIPTIONS) == set(ALL_FEATURE_KEYS)


def test_system_prompt_contains_every_feature_and_bound():
    from freqai_spec import (
        HORIZON_BOUNDS, MODEL_FAMILIES, MODEL_PARAM_BOUNDS,
    )
    from indicators.freqai_features import ALL_FEATURE_KEYS

    prompt = fg.build_system_prompt()
    for key in ALL_FEATURE_KEYS:
        assert key in prompt, f"feature {key!r} missing from prompt"
    for param in MODEL_PARAM_BOUNDS:
        assert param in prompt, f"model param {param!r} missing from prompt"
    assert MODEL_FAMILIES[0] in prompt
    assert str(HORIZON_BOUNDS[0]) in prompt and str(HORIZON_BOUNDS[1]) in prompt
    # Output contract + gate awareness
    assert "ONE JSON object" in prompt
    assert "walk-forward" in prompt.lower()


def test_prompt_includes_failures_and_batch_diversity():
    prompt = fg.build_freqai_prompt(
        target_regime="ranging",
        failure_examples="PRIOR FAILED ML EXPERIMENTS:\n- FreqaiOld [FAIL_WF_UNSTABLE]",
        prior_in_batch=[{"name": "FreqaiA", "features": ["rsi"],
                         "horizon": 24, "regime": "all"}],
    )
    assert "regime: ranging" in prompt
    assert "FreqaiOld" in prompt
    assert "structurally" in prompt and "FreqaiA" in prompt


# ---------------------------------------------------------------------------
# propose_freqai_specs
# ---------------------------------------------------------------------------

def test_propose_happy_path(isolated_registry, isolated_artifacts, valid_spec):
    valid_spec["name"] = "FreqaiProposed"
    with patch("llm_client.chat_completion", return_value=_llm_reply(valid_spec)):
        results = fg.propose_freqai_specs(count=1)

    assert results[0]["success"], results[0]
    row = isolated_registry.get_strategy_by_name("FreqaiProposed")
    assert row is not None
    assert row["spec_type"] == "freqai"
    # Factory stamps its own generation_id regardless of the LLM's
    assert results[0]["spec"]["generation_id"].startswith("gen-freqai-")


def test_propose_overrides_llm_owned_fields(
    isolated_registry, isolated_artifacts, valid_spec
):
    valid_spec["name"] = "FreqaiSneaky"
    valid_spec["spec_type"] = "rule"           # LLM lies about the type
    valid_spec["generation_id"] = "x" * 200    # and emits a junk id
    with patch("llm_client.chat_completion", return_value=_llm_reply(valid_spec)):
        results = fg.propose_freqai_specs(count=1)

    assert results[0]["success"], results[0]
    spec = results[0]["spec"]
    assert spec["spec_type"] == "freqai"
    assert len(spec["generation_id"]) <= 64


def test_propose_retries_on_validator_rejection(
    isolated_registry, isolated_artifacts, valid_spec
):
    bad = dict(valid_spec, name="FreqaiBad",
               features=["rsi", "made_up_feature", "roc"])
    good = dict(valid_spec, name="FreqaiFixed")
    replies = [_llm_reply(bad), _llm_reply(good)]

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(list(messages))
        return replies[len(calls) - 1]

    with patch("llm_client.chat_completion", side_effect=fake_chat):
        results = fg.propose_freqai_specs(count=1, max_retries=1)

    assert results[0]["success"]
    assert results[0]["name"] == "FreqaiFixed"
    # Second call carried the validator's error back to the LLM
    feedback = calls[1][-1]["content"]
    assert "rejected by the validator" in feedback
    assert "unknown feature" in feedback


def test_propose_gives_up_after_retries(
    isolated_registry, isolated_artifacts, valid_spec
):
    bad = dict(valid_spec, name="FreqaiNeverGood", features=["nope", "x", "y"])
    with patch("llm_client.chat_completion", return_value=_llm_reply(bad)):
        results = fg.propose_freqai_specs(count=1, max_retries=1)

    assert not results[0]["success"]
    assert "unknown feature" in results[0]["error"]
    assert isolated_registry.get_strategy_by_name("FreqaiNeverGood") is None


def test_propose_retries_on_unparseable_reply(
    isolated_registry, isolated_artifacts, valid_spec
):
    valid_spec["name"] = "FreqaiAfterGarbage"
    replies = ["I think a good strategy would be...", _llm_reply(valid_spec)]
    with patch("llm_client.chat_completion", side_effect=replies):
        results = fg.propose_freqai_specs(count=1, max_retries=1)

    assert results[0]["success"]


def test_propose_suffixes_name_collision(
    isolated_registry, isolated_artifacts, valid_spec
):
    isolated_registry.register_strategy(
        name="FreqaiDupe", filepath="/x.py", spec_type="freqai")
    valid_spec["name"] = "FreqaiDupe"
    with patch("llm_client.chat_completion", return_value=_llm_reply(valid_spec)):
        results = fg.propose_freqai_specs(count=1)

    assert results[0]["success"]
    assert results[0]["name"] == "FreqaiDupe_2"


def test_propose_llm_error_is_contained(isolated_registry, isolated_artifacts):
    with patch("llm_client.chat_completion", side_effect=RuntimeError("api down")):
        results = fg.propose_freqai_specs(count=2)

    assert len(results) == 2
    assert all(not r["success"] for r in results)
    assert "api down" in results[0]["error"]


# ---------------------------------------------------------------------------
# Failure memory filtering + overview
# ---------------------------------------------------------------------------

def test_failure_memory_filters_by_spec_type(isolated_registry):
    sr = isolated_registry
    rid = sr.register_strategy(name="RuleFail", filepath="/r.py")
    fid = sr.register_strategy(name="FreqaiFail", filepath="/q.py",
                               spec_type="freqai")
    sr.retire_strategy(rid, reason="r", verdict="FAIL_UNPROFITABLE")
    sr.retire_strategy(fid, reason="q", verdict="FAIL_WF_UNSTABLE")

    rule_names = [f["name"] for f in sr.get_recent_failures(spec_type="rule")]
    ml_names = [f["name"] for f in sr.get_recent_failures(spec_type="freqai")]
    all_names = [f["name"] for f in sr.get_recent_failures()]

    assert rule_names == ["RuleFail"]
    assert ml_names == ["FreqaiFail"]
    assert set(all_names) == {"RuleFail", "FreqaiFail"}


def test_report_lists_freqai_experiments_only(
    isolated_registry, isolated_artifacts, valid_spec
):
    isolated_registry.register_strategy(name="RuleThing", filepath="/r.py")
    valid_spec["name"] = "FreqaiReported"
    with patch("llm_client.chat_completion", return_value=_llm_reply(valid_spec)):
        fg.propose_freqai_specs(count=1)

    report = fg.build_report()
    assert "FreqaiReported" in report
    assert "RuleThing" not in report
    # Sidecar-derived experiment shape present
    assert "24" in report  # horizon


# ---------------------------------------------------------------------------
# Weekly hookup (env-gated, off by default)
# ---------------------------------------------------------------------------

def test_weekly_hookup_is_env_gated():
    src = (BASE / "scripts" / "orchestrator.py").read_text()
    assert 'os.environ.get("FREQAI_WEEKLY_COUNT", "0")' in src
    assert "propose_freqai_specs" in src


def test_compose_defaults_weekly_count_off():
    compose = (BASE.parent / "docker-compose.yml").read_text()
    assert "FREQAI_WEEKLY_COUNT=${FREQAI_WEEKLY_COUNT:-0}" in compose
