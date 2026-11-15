"""Service preflight must fail before a public-data connection or process launch."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "user_data/configs/h1_dryrun.json").read_text())


def test_missing_selection_requires_explicit_placeholder(tmp_path):
    from h1_service import prepare_config
    with pytest.raises(ValueError, match="selected"):
        prepare_config(config(), tmp_path, tmp_path / "ledger.sqlite", placeholder=False)
    prepared = prepare_config(config(), tmp_path, tmp_path / "ledger.sqlite", placeholder=True)
    assert prepared["h1_selection"]["placeholder"]
    assert prepared["dry_run"] is True
    assert prepared["research_params"]["ladder"] == [20, 60, 180]
    assert prepared["research_h1"]["designated_portfolio_notional"] == 250000


@pytest.mark.parametrize("change", [
    {"dry_run": False}, {"trading_mode": "futures"}, {"fee": .003}, {"strategy": "Other"},
    {"api_server": {"enabled": True}}, {"exchange": {"name": "okx", "key": "not-a-real-key"}},
    {"add_config_files": ["extra.json"]}, {"startup_candle_count": 365},
])
def test_service_refuses_execution_contract_drift(tmp_path, change):
    from h1_service import prepare_config
    cfg = config()
    cfg.update(change)
    with pytest.raises(ValueError):
        prepare_config(cfg, tmp_path, tmp_path / "ledger.sqlite", placeholder=True)


def test_dryrun_strategy_requires_in_process_verified_adapter_token(tmp_path):
    pytest.importorskip("freqtrade")
    from freqtrade.enums import RunMode
    from H1Trend import H1Trend
    from h1_service import prepare_config, PAPER_TOKEN
    cfg = prepare_config(config(), tmp_path, tmp_path / "ledger.sqlite", placeholder=True)
    cfg["runmode"] = RunMode.DRY_RUN
    with pytest.raises(RuntimeError):
        H1Trend(cfg).bot_start()
    cfg["_h1_paper_adapter"] = PAPER_TOKEN
    H1Trend(cfg).bot_start()
    cfg["runmode"] = RunMode.LIVE
    with pytest.raises(RuntimeError):
        H1Trend(cfg).bot_start()


def test_selected_row_controls_params_and_rejects_changed_source(tmp_path):
    import research
    from h1_service import prepare_config, strategy_hash
    cfg = config()
    params = {**cfg["research_params"], "ladder": [40, 120, 365], "sigma_target": .3, "cost_rate": .003}
    descriptor = {"hypothesis": "H1", "strategy": "H1Trend", "params": params, "config_hash": strategy_hash(params)}
    trial = research.start_trial(descriptor, "selection", "test", "20220101-20240701", "test")
    with research.ledger() as conn:
        conn.execute("UPDATE trial_ledger SET verdict='selected' WHERE trial_id=?", (trial,))
    import os
    prepared = prepare_config(cfg, tmp_path, os.environ["TRIAL_LEDGER_PATH"])
    assert prepared["research_params"] == {**params, "cost_rate": .0015}
    assert not prepared["h1_selection"]["placeholder"]
    with research.ledger() as conn:
        conn.execute("UPDATE trial_ledger SET config_hash='changed' WHERE trial_id=?", (trial,))
    with pytest.raises(ValueError, match="changed"):
        prepare_config(cfg, tmp_path, os.environ["TRIAL_LEDGER_PATH"])


def test_service_status_does_not_create_runtime_or_start_a_bot(tmp_path):
    from h1_service import status
    directory = tmp_path / "absent"
    result = status(directory)
    assert result["running"] is False
    assert not directory.exists()


def test_start_refuses_before_forward_date_without_spawning(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import subprocess
    from h1_service import start
    called = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: called.append(a))
    with pytest.raises(ValueError, match="2026-12-19"):
        start(ROOT / "user_data/configs/h1_dryrun.json", tmp_path / "state", tmp_path / "ledger.sqlite",
              placeholder=True, now=datetime(2026, 12, 18, tzinfo=timezone.utc))
    assert not called


def test_stop_of_absent_service_never_sends_signal(tmp_path, monkeypatch):
    import os
    from h1_service import stop
    calls = []
    monkeypatch.setattr(os, "kill", lambda *a: calls.append(a))
    assert stop(tmp_path)["running"] is False
    assert calls == []


def test_run_refuses_before_forward_date_without_constructing_bot(tmp_path, monkeypatch):
    import h1_service
    from datetime import datetime, timezone
    class Before(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 12, 18, tzinfo=timezone.utc)
    monkeypatch.setattr(h1_service, "datetime", Before)
    with pytest.raises(ValueError, match="2026-12-19"):
        h1_service.run(ROOT / "user_data/configs/h1_dryrun.json", tmp_path, tmp_path / "ledger.sqlite", placeholder=True)
    assert not (tmp_path / "trades.sqlite").exists()



