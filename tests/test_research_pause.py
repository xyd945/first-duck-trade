"""The research pause applies even when API keys or old jobs exist."""
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


def test_default_schedule_omits_generation_and_rescue(monkeypatch):
    import orchestrator as orch
    for key in ("ENABLE_LLM_GENERATION", "ENABLE_STRATEGY_CRITIC", "ENABLE_HYPEROPT_RESCUE"):
        monkeypatch.delenv(key, raising=False)
    scheduler = Mock()
    monkeypatch.setattr(orch, "BlockingScheduler", Mock(return_value=scheduler))
    scheduler.get_jobs.return_value = []
    monkeypatch.setattr(orch.time, "sleep", Mock())
    for name in ("job_fetch_macro_data", "job_classify_regime", "job_apply_regime", "job_health_check"):
        monkeypatch.setattr(orch, name, Mock())
    orch.main()
    ids = {c.kwargs["id"] for c in scheduler.add_job.call_args_list}
    assert not ids.intersection({"generate_strategies", "hyperopt_candidates", "reflector", "llm_regime"})


@pytest.mark.parametrize("entry", ["generator", "freqai", "critic", "hyperopt", "jobs"])
def test_direct_entry_points_respect_pause(entry, monkeypatch):
    import llm_client
    import strategy_registry as registry
    import backtest_runner
    import orchestrator
    for key in ("ENABLE_LLM_GENERATION", "ENABLE_STRATEGY_CRITIC", "ENABLE_HYPEROPT_RESCUE"):
        monkeypatch.delenv(key, raising=False)
    llm = Mock(side_effect=AssertionError("LLM must stay paused"))
    process = Mock(side_effect=AssertionError("hyperopt must stay paused"))
    monkeypatch.setattr(llm_client, "chat_completion", llm)
    monkeypatch.setattr(backtest_runner.subprocess, "run", process)
    monkeypatch.setattr(registry, "get_hyperopt_candidates", Mock(side_effect=AssertionError("paused job")))
    if entry == "generator":
        from strategy_generator import generate_strategy
        assert generate_strategy()["success"] is False
    elif entry == "freqai":
        from freqai_generator import propose_freqai_specs
        assert propose_freqai_specs(count=1)[0]["success"] is False
    elif entry == "critic":
        from strategy_critic import critic_review
        assert critic_review("class Test: pass")["verdict"] == "SKIP"
    elif entry == "hyperopt":
        assert backtest_runner.run_hyperopt("Test")["success"] is False
    else:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
        for fn in (orchestrator.job_generate_strategies, orchestrator.job_reflector,
                   orchestrator.job_llm_regime_override, orchestrator.job_hyperopt_candidates):
            fn()
    llm.assert_not_called()
    process.assert_not_called()
