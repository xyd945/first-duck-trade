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


def test_dryrun_strategy_requires_in_process_verified_adapter_token(tmp_path, monkeypatch):
    pytest.importorskip("freqtrade")
    from freqtrade.enums import RunMode
    monkeypatch.syspath_prepend(str(ROOT / "user_data/strategies/candidates"))
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


@pytest.mark.parametrize("key,value", [("rebalance_weekday", 1), ("minimum_bars", 1),
                                           ("paper_startup", 365), ("annualisation_days", 252)])
def test_service_refuses_registered_rule_drift(tmp_path, key, value):
    from h1_service import prepare_config
    cfg = config()
    cfg["research_h1"][key] = value
    with pytest.raises(ValueError, match="registered rules"):
        prepare_config(cfg, tmp_path, tmp_path / "ledger.sqlite", placeholder=True)


def test_service_allows_one_explicit_pending_notional_change(tmp_path):
    from h1_service import prepare_config
    cfg = config()
    cfg["research_h1"]["designated_portfolio_notional"] = 100000
    assert prepare_config(cfg, tmp_path, tmp_path / "ledger.sqlite", placeholder=True)["research_h1"]["designated_portfolio_notional"] == 100000


def test_service_status_does_not_create_runtime_or_start_a_bot(tmp_path):
    from h1_service import status
    directory = tmp_path / "absent"
    result = status(directory)
    assert result["running"] is False
    assert not directory.exists()


def test_status_tolerates_a_journal_whose_schema_is_still_starting(tmp_path):
    import sqlite3
    from h1_service import status
    sqlite3.connect(tmp_path / "journal.sqlite").close()
    result = status(tmp_path)
    assert result["risk"] is None
    assert result["ready"] is False


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


@pytest.fixture
def offline_service(tmp_path, monkeypatch):
    pytest.importorskip("freqtrade")
    import h1_service
    import subprocess
    import time
    monkeypatch.setenv("H1_OFFLINE_STARTED", str(time.monotonic()))
    from datetime import datetime, timezone
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 12, 19, 0, 4, tzinfo=timezone.utc)
    monkeypatch.setattr(h1_service, "datetime", Clock)
    original = subprocess.Popen
    children = []
    mode = ["ready"]
    def spawn(command, **kwargs):
        if str(ROOT / "user_data/scripts/h1_service.py") not in command:
            return original(command, **kwargs)
        child = original([command[0], str(ROOT / "tests/fixtures/h1_service_offline.py"), mode[0], *command[1:]], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(h1_service.subprocess, "Popen", spawn)
    yield h1_service, mode, children
    for child in children:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)


@pytest.mark.parametrize("mode_name", ["ready", "lock-race"])
def test_offline_start_waits_for_ready_and_stop_releases_lock(offline_service, tmp_path, mode_name):
    service, mode, children = offline_service
    mode[0] = mode_name
    directory = tmp_path / "paper"
    result = service.start(ROOT / "user_data/configs/h1_dryrun.json", directory, tmp_path / "ledger", placeholder=True)
    assert result["ready"] is True
    assert result["heartbeat"]["state"] == "ready"
    assert (directory / "trades.sqlite").exists()
    assert (directory / "journal.sqlite").exists()
    with pytest.raises(ValueError, match="already running"):
        service.start(ROOT / "user_data/configs/h1_dryrun.json", directory, tmp_path / "ledger", placeholder=True)
    assert service.stop(directory)["running"] is False
    children[0].wait(timeout=3)
    assert service.status(directory)["ready"] is False
    assert service.status(directory)["heartbeat"]["state"] == "stopped"


def test_first_tick_transient_recovers_to_ready_within_decision_window(offline_service, tmp_path):
    import sqlite3
    from datetime import datetime
    service, mode, children = offline_service
    mode[0] = "transient-once"
    result = service.start(ROOT / "user_data/configs/h1_dryrun.json", tmp_path, tmp_path / "ledger", placeholder=True)
    assert result["ready"]
    assert result["heartbeat"]["batch_status"] == "finished"
    assert datetime.fromisoformat(result["heartbeat"]["asof"]).minute < 5
    assert not result["heartbeat"]["risk"]["kill_switch_active"]
    service.stop(tmp_path)
    children[0].wait(timeout=3)
    with sqlite3.connect(tmp_path / "journal.sqlite") as conn:
        errors = conn.execute("SELECT payload FROM paper_records WHERE kind='error'").fetchall()
    assert len(errors) == 1
    assert json.loads(errors[0][0])["error"] == "offline transient ticker failure"


@pytest.mark.parametrize("mode_name", ["exit", "tick-error", "entry-error", "hang"])
def test_failed_start_reaps_only_its_child(offline_service, tmp_path, monkeypatch, mode_name):
    service, mode, children = offline_service
    mode[0] = mode_name
    monkeypatch.setattr(service, "STARTUP_TIMEOUT", 1.5, raising=False)
    monkeypatch.setattr(service, "STOP_TIMEOUT", .3, raising=False)
    with pytest.raises(RuntimeError):
        service.start(ROOT / "user_data/configs/h1_dryrun.json", tmp_path, tmp_path / "ledger", placeholder=True)
    assert children[0].poll() is not None
    assert service.status(tmp_path)["running"] is False
    if mode_name in ("tick-error", "entry-error"):
        import sqlite3
        with sqlite3.connect(tmp_path / "journal.sqlite") as conn:
            record = conn.execute("SELECT payload FROM paper_records WHERE kind='error'").fetchone()
        assert record is not None
        assert json.loads(record[0])["time"] == "2026-12-19T00:00:00+00:00"
        if mode_name == "entry-error":
            with sqlite3.connect(tmp_path / "trades.sqlite") as conn:
                assert conn.execute("SELECT count(*) FROM orders").fetchone()[0] == 1
            mode[0] = "resume-entry"
            result = service.start(ROOT / "user_data/configs/h1_dryrun.json", tmp_path, tmp_path / "ledger", placeholder=True)
            assert result["ready"]
            service.stop(tmp_path)
            children[-1].wait(timeout=3)
            with sqlite3.connect(tmp_path / "trades.sqlite") as conn:
                assert dict(conn.execute("SELECT ft_pair,count(*) FROM orders GROUP BY ft_pair")) == {
                    "BTC/USDT": 1, "SOL/USDT": 1, "XRP/USDT": 1}
            with sqlite3.connect(tmp_path / "journal.sqlite") as conn:
                assert conn.execute("SELECT payload FROM paper_records WHERE kind='error'").fetchone() == record


def test_stop_rejects_foreign_pid_without_signal(tmp_path, monkeypatch):
    import fcntl
    import os
    import h1_service
    calls = []
    with (tmp_path / "service.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        h1_service.heartbeat(tmp_path, {"state": "ready"})
        monkeypatch.setattr(os, "kill", lambda *args: calls.append(args))
        with pytest.raises(ValueError, match="PID identity changed"):
            h1_service.stop(tmp_path)
    assert calls == []