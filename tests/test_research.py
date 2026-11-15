"""Research integrity checks use synthetic OHLCV and local SQLite only."""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "user_data/scripts"))
import research
from tests.test_research_snapshots import PAIRS, source_data


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    source = source_data(tmp_path / "source")
    s1 = research.freeze_snapshot(source, tmp_path / "S1", "20200101", "20200107")
    h1 = json.loads(s1.read_text())["data_snapshot_hash"]
    s2 = research.freeze_snapshot(source, tmp_path / "S2", "20200101", "20200112", previous=(s1, h1))
    config = {
        "hypothesis": "H1", "warmup": ["20200101", "20200101"],
        "splits": {"dev": ["20200102", "20200104"], "selection": ["20200105", "20200107"],
                   "eval": ["20200108", "20200112"]},
        "snapshots": {"S1": {"manifest": str(s1), "hash": h1},
                      "S2": {"manifest": str(s2), "hash": json.loads(s2.read_text())["data_snapshot_hash"]}},
        "params": {"ladder": [20, 60, 180], "sigma_target": 0.45, "vol_window": 60,
                   "rebalance_band": 0.025, "cost_rate": 0.003},
    }
    path = tmp_path / "research.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("RESEARCH_CONFIG", str(path))
    monkeypatch.setenv("TRIAL_LEDGER_PATH", str(tmp_path / "trials.sqlite"))
    return path, config


@pytest.fixture
def runtime(experiment, tmp_path, monkeypatch):
    import backtest_runner as runner
    base = tmp_path / "user_data"
    strategies = base / "strategies/candidates"
    strategies.mkdir(parents=True)
    (strategies / "Toy.py").write_text("class Toy: pass\n")
    (base / "config.json").write_text(json.dumps({"dry_run": True, "fee": 0.003,
        "exchange": {"name": "okx", "pair_whitelist": [p.replace('-', '/') for p in PAIRS]},
        "pairlists": [{"method": "StaticPairList"}], "trading_mode": "spot"}))
    monkeypatch.setattr(runner, "BASE_DIR", base)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "HOST_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_GIT_SHA", "0" * 40)  # explicit synthetic revision
    return runner, base


def test_inclusive_windows_keep_one_continuous_path(experiment):
    _, config = experiment
    assert hasattr(research, "path_window"), "Split runs must retain the development prefix"
    policy = research.load_config()
    assert research.window(policy, "dev") == "20200102-20200105"
    assert research.window(policy, "selection") == "20200105-20200108"
    assert research.path_window(policy, "selection") == "20200102-20200108"
    assert research.path_window(policy, "eval") == "20200102-20200113"
    with pytest.raises(ValueError, match="entire"):
        research.window(policy, "eval", "20200108-20200110")
    with pytest.raises(ValueError, match="escapes"):
        research.window(policy, "dev", "20200104-20200106")
    config["splits"]["selection"][0] = "20200104"
    with pytest.raises(ValueError, match="overlap"):
        research.load_config(config)


def test_loading_policy_does_not_open_s2(experiment, monkeypatch):
    path, config = experiment
    Path(config["snapshots"]["S2"]["manifest"]).unlink()
    assert hasattr(research, "path_window"), "Policy must separate S1 from sealed S2"
    assert research.load_config()["splits"] == config["splits"]


def test_g0_config_hash_ignores_cost_and_snapshot_but_not_source(runtime, experiment):
    _, base = runtime
    _, policy = experiment
    descriptor, _ = research.describe_run(base, policy, "Toy", "1d")
    params = {k: policy["params"][k] for k in ("ladder", "sigma_target", "vol_window", "rebalance_band")}
    expected = hashlib.sha256((base / "strategies/candidates/Toy.py").read_bytes() +
                              json.dumps(params, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert research.config_hash(descriptor) == expected
    engine = base / "config.json"
    policy["params"]["cost_rate"] = 0.0015
    cfg = json.loads(engine.read_text())
    cfg["fee"] = 0.0015
    engine.write_text(json.dumps(cfg))
    revised, _ = research.describe_run(base, policy, "Toy", "1d")
    assert research.config_hash(revised) == expected
    (base / "strategies/candidates/Toy.py").write_text("class Toy: changed = True\n")
    changed, _ = research.describe_run(base, policy, "Toy", "1d")
    assert research.config_hash(changed) != expected


def test_ledger_counts_distinct_selection_hashes_including_errors(experiment):
    cases = [("a", "dev", "ok"), ("a", "selection", "ok"), ("a", "selection", "ok"),
             ("b", "selection", "baseline"), ("c", "eval", "sensitivity"), ("d", "eval", "error")]
    for key, split, verdict in cases:
        descriptor = {"hypothesis": "H1", "config_hash": key * 64, "data_snapshot_hash": "fixture",
                      "params": {"cost_rate": 0.003}}
        trial = research.start_trial(descriptor, split, "backtest", "20200105-20200108", "fixture-sha")
        # A crash remains an error, never an unrecognised RUNNING verdict.
        with research.ledger() as conn:
            row = conn.execute("SELECT verdict FROM trial_ledger WHERE trial_id=?", (trial,)).fetchone()
        assert row["verdict"] == "error"
        research.finish_trial(trial, {"success": verdict != "error"}, verdict=verdict)
    crash = research.start_trial(descriptor, "dev", "hyperopt", "20200102-20200105", "fixture-sha", epochs=15)
    with research.ledger() as conn:
        rows = conn.execute("SELECT * FROM trial_ledger ORDER BY rowid").fetchall()
    assert list(rows[0].keys()) == (
        "trial_id created_utc hypothesis repo git_sha config_hash data_snapshot_hash split "
        "window_start window_end params_json n_trades_or_rebalances net_return net_sharpe "
        "max_drawdown verdict notes".split())
    assert [r["verdict"] for r in rows] == ["ok", "ok", "rerun", "baseline", "sensitivity", "error", "error"]
    assert rows[0]["window_end"] == "20200107"  # ledger dates are inclusive
    assert rows[0]["net_return"] is None  # fix 4 supplies accounting, not fake engine net metrics
    report = research.search_budget()["H1"]
    assert report["N"] == 2
    assert report["trials"] == 7
    assert report["unfinished"] == 1
    assert report["hyperopt_epochs_requested"] == 15
    with pytest.raises(ValueError, match="verdict"):
        research.finish_trial(crash, {"success": True}, verdict="COMPLETED")


def test_selection_row_unlocks_atomic_once_only_preregistered_evaluation(runtime, experiment):
    from concurrent.futures import ThreadPoolExecutor
    _, base = runtime
    _, policy = experiment
    descriptor, _ = research.describe_run(base, policy, "Toy", "1d")
    registration = research.preregister(descriptor)
    assert hasattr(research, "record_selection"), "Evaluation must check an actual selected ledger row"
    with pytest.raises(ValueError, match="selection"):
        research.claim_evaluation(registration, descriptor)
    trial = research.start_trial(descriptor, "selection", "backtest", "20200105-20200108", "fixture-sha")
    research.finish_trial(trial, {"success": True})
    selected = select_before_cutoff(trial)
    with research.ledger() as conn:
        row = conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (selected,)).fetchone()
    assert row["verdict"] == "selected"
    assert row["config_hash"] == research.config_hash(descriptor)
    with pytest.raises(ValueError, match="selected"):
        research.record_selection(trial)

    def claim(_):
        try:
            research.claim_evaluation(registration, descriptor)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, range(2))) == [False, True]
    assert research.preregister(descriptor) == registration
    assert claim(0) is False
    with pytest.raises(ValueError, match="opened"):
        research.check_not_evaluated("H1")
    revised = {**descriptor, "engine_config_hash": "new-cost"}
    with pytest.raises(ValueError, match="selection"):
        research.preregister(revised)


def fake_engine(cmd, **kwargs):
    """Only the Docker boundary is mocked; emit a labelled synthetic Freqtrade zip."""
    import io
    import zipfile
    from types import SimpleNamespace
    run_dir = Path(cmd[cmd.index("--volume") + 1].split(":")[0])
    start, end = cmd[cmd.index("--timerange") + 1].split("-")
    exported = cmd[cmd.index("--backtest-directory") + 1]
    export_dir = run_dir / Path(exported).relative_to("/freqtrade/user_data")
    export_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(research.date(start), research.date(end) - pd.Timedelta(days=1))
    wallets = pd.DataFrame({"date": dates, "currency": "BTC", "rate": 10., "balance": 2.})
    buf = io.BytesIO()
    wallets.to_feather(buf)
    trade = {"pair": "BTC/USDT", "open_date": "2020-01-03 00:00:00+00:00",
             "close_date": (research.date(end) - pd.Timedelta(days=1)).isoformat(),
             "profit_abs": 20.}
    stats = {"strategy": {"Toy": {"trades": [trade], "total_trades": 1,
              "sharpe": 7., "profit_total": 0.5, "backtest_start": research.date(start).isoformat(),
              "backtest_end": (research.date(end) - pd.Timedelta(days=1)).isoformat()}}}
    with zipfile.ZipFile(export_dir / "fixture.zip", "w") as z:
        z.writestr("backtest-result-fixture.json", json.dumps(stats))
        z.writestr("backtest-result-fixture_Toy_wallet.feather", buf.getvalue())
    return SimpleNamespace(returncode=0, stdout="SYNTHETIC TEST FIXTURE", stderr="")


def test_selection_runs_continuously_then_slices_without_resetting_positions(runtime, monkeypatch):
    runner, base = runtime
    calls = []
    def engine(cmd, **kwargs):
        calls.append(cmd)
        run_dir = Path(cmd[cmd.index("--volume") + 1].split(":")[0])
        assert not (run_dir / "trial_ledger.sqlite").exists()
        frame = pd.read_feather(run_dir / "data/BTC_USDT-1d.feather")
        assert frame.date.max() == pd.Timestamp("2020-01-07", tz="UTC")
        assert frame.date.min() == pd.Timestamp("2020-01-01", tz="UTC")
        return fake_engine(cmd, **kwargs)
    monkeypatch.setattr(runner.subprocess, "run", engine)
    result = runner.run_backtest("Toy", timeframe="1d", split="selection")
    assert result["success"], result
    assert len(calls) == 1
    assert calls[0][calls[0].index("--timerange") + 1] == "20200102-20200108"
    assert "--no-deps" in calls[0] and "--cache" in calls[0]
    assert result["timerange"] == "20200105-20200108"
    assert result["path_timerange"] == "20200102-20200108"
    slice_data = json.loads(Path(result["slice_path"]).read_text())
    assert slice_data["trades"][0]["open_date"] == "2020-01-03 00:00:00+00:00"
    assert slice_data["opening_wallet"][0]["balance"] == 2.
    wallet = pd.read_feather(Path(result["slice_path"]).with_suffix(".feather"))
    assert wallet.date.min() == pd.Timestamp("2020-01-05", tz="UTC")
    assert wallet.date.max() == pd.Timestamp("2020-01-07", tz="UTC")
    assert "sharpe" not in result  # whole-path metrics must not masquerade as selection metrics
    assert result["path_metrics"]["sharpe"] == 7.
    with research.ledger() as conn:
        row = conn.execute("SELECT * FROM trial_ledger").fetchone()
    assert row["verdict"] == "ok" and row["split"] == "selection"
    assert row["window_start"] == "20200105" and row["window_end"] == "20200107"


def select_before_cutoff(trial):
    from unittest.mock import patch
    with patch("research.datetime", wraps=research.datetime) as clock:
        clock.now.return_value = research.date("20200112")
        return research.record_selection(trial)


def select_for_test(runner):
    registration = runner.preregister_evaluation("Toy", timeframe="1d")
    selection = runner.run_backtest("Toy", timeframe="1d", split="selection")
    assert selection["success"], selection
    select_before_cutoff(selection["trial_id"])
    return registration


def test_evaluation_never_opens_s2_without_selection_and_consumes_failures(runtime, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    runner, _ = runtime
    registration = runner.preregister_evaluation("Toy", timeframe="1d")
    with monkeypatch.context() as guard:
        reader = Mock(side_effect=AssertionError("S2 must not be read"))
        guard.setattr(research, "verify_snapshot", reader)
        denied = runner.run_evaluation("Toy", registration=registration, timeframe="1d")
        assert not denied["success"] and "selection" in denied["error"]
        reader.assert_not_called()
    monkeypatch.setattr(runner.subprocess, "run", fake_engine)
    assert select_for_test(runner) == registration
    process = Mock(return_value=SimpleNamespace(returncode=2, stdout="", stderr="fixture failure"))
    monkeypatch.setattr(runner.subprocess, "run", process)
    first = runner.run_evaluation("Toy", registration=registration, timeframe="1d")
    assert not first["success"] and first["split"] == "eval"
    second = runner.run_evaluation("Toy", registration=registration, timeframe="1d")
    assert "already opened" in second["error"]
    assert process.call_count == 1
    rejected = runner.run_backtest("Toy", timeframe="1d", split="selection")
    assert "already opened" in rejected["error"]
    import strategy_registry
    with pytest.raises(ValueError, match="evaluation"):
        strategy_registry.record_backtest(1, first)


def test_future_evaluation_fails_before_touching_s2(runtime, experiment, monkeypatch):
    runner, _ = runtime
    path, policy = experiment
    policy["splits"]["eval"][1] = "29990101"
    path.write_text(json.dumps(policy))
    monkeypatch.setattr(runner.subprocess, "run", fake_engine)
    registration = select_for_test(runner)
    result = runner.run_evaluation("Toy", registration=registration, timeframe="1d")
    assert not result["success"] and "cutoff" in result["error"]
    with research.ledger() as conn:
        row = conn.execute("SELECT opened_utc FROM evaluation_registrations").fetchone()
    assert row["opened_utc"] is None


def test_tamper_failure_retains_trial_identity(runtime, experiment, monkeypatch):
    from unittest.mock import Mock
    runner, base = runtime
    _, cfg = experiment
    expected, _ = research.describe_run(base, cfg, "Toy", "1d")
    manifest = Path(cfg["snapshots"]["S1"]["manifest"])
    (manifest.parent / "BTC-USDT.csv").write_bytes(b"tampered fixture")
    process = Mock(side_effect=AssertionError("must not run engine"))
    monkeypatch.setattr(runner.subprocess, "run", process)
    result = runner.run_backtest("Toy", timeframe="1d")
    assert "hash mismatch" in result["error"]
    assert result["config_hash"] == research.config_hash(expected)
    assert research.search_budget()["H1"]["failed"] == 1
    process.assert_not_called()


def test_hyperopt_batches_record_failures_and_requested_epochs(runtime, monkeypatch):
    import subprocess
    from unittest.mock import Mock
    runner, _ = runtime
    monkeypatch.setenv("ENABLE_HYPEROPT_RESCUE", "true")
    engine = Mock(side_effect=subprocess.TimeoutExpired("fixture", 1))
    monkeypatch.setattr(runner.subprocess, "run", engine)
    result = runner.run_hyperopt("Toy", timeframe="1d", epochs=15)
    assert not result["success"] and result["split"] == "dev"
    cmd = engine.call_args.args[0]
    assert cmd[cmd.index("--timerange") + 1] == "20200102-20200105"
    assert research.search_budget()["H1"]["hyperopt_epochs_requested"] == 15


@pytest.mark.parametrize("change", ["fee", "timeframe", "venue", "pairs", "sidecar", "extra_param", "nested_key"])
def test_h1_execution_cannot_drift_from_registered_inputs(runtime, experiment, change):
    _, base = runtime
    _, policy = experiment
    cfg_path = base / "config.json"
    cfg = json.loads(cfg_path.read_text())
    timeframe = "1d"
    if change == "fee":
        cfg["fee"] = 0.0
    elif change == "venue":
        cfg["exchange"]["name"] = "binance"
    elif change == "pairs":
        cfg["exchange"]["pair_whitelist"] = ["BTC/USDT"]
    elif change == "timeframe":
        timeframe = "1h"
    elif change == "extra_param":
        policy["params"]["unhashed_lookback"] = 9
    elif change == "nested_key":
        cfg["exchange"]["ccxt_config"] = {"apiKey": "synthetic-not-a-key"}
    else:
        (base / "strategies/candidates/Toy.json").write_text('{"params":{"buy":{"hidden":1}}}')
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        research.describe_run(base, policy, "Toy", timeframe)


def test_staged_strategy_receives_the_recorded_parameters(runtime, experiment):
    _, base = runtime
    _, policy = experiment
    descriptor, engine = research.describe_run(base, policy, "Toy", "1d")
    run = research.prepare_run(base, policy, descriptor, engine, "fixture", "20200102-20200105", "dev")
    actual = json.loads((run / "config.json").read_text())
    assert actual.get("research_params") == policy["params"]


def test_baseline_runs_and_cost_sensitivities_do_not_increase_N(runtime, monkeypatch):
    runner, _ = runtime
    monkeypatch.setattr(runner.subprocess, "run", fake_engine)
    assert "verdict" in __import__("inspect").signature(runner.run_backtest).parameters
    for verdict in ("baseline", "sensitivity"):
        result = runner.run_backtest("Toy", timeframe="1d", verdict=verdict)
        assert result["success"], result
    assert research.search_budget()["H1"]["N"] == 0
    with research.ledger() as conn:
        assert [r[0] for r in conn.execute("SELECT verdict FROM trial_ledger ORDER BY rowid")] == ["baseline", "sensitivity"]


def test_cli_carries_explicit_split_and_registration(runtime, monkeypatch, capsys):
    runner, _ = runtime
    monkeypatch.setattr(runner.subprocess, "run", fake_engine)
    assert hasattr(runner, "main"), "CLI must expose explicit research splits and registration"
    assert runner.main(["Toy", "--split", "dev", "--timeframe", "1d"]) == 0
    assert json.loads(capsys.readouterr().out)["timerange"] == "20200102-20200105"
    assert runner.main(["Toy", "--register-eval", "--timeframe", "1d"]) == 0
    registration = json.loads(capsys.readouterr().out)["registration"]
    assert runner.main(["Toy", "--evaluation", registration, "--timeframe", "1d"]) == 1
    assert "selection" in json.loads(capsys.readouterr().out)["error"]
    assert runner.main(["Toy", "--evaluate", "--timeframe", "1d"]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL_SANITY"


def test_research_cli_freezes_pinned_window_and_reports_budget(experiment, tmp_path, capsys):
    assert hasattr(research, "main"), "Snapshot and budget CLI are required"
    assert research.main(["freeze", "S1", "--source", str(tmp_path / "source"),
                          "--destination", str(tmp_path / "cli-snapshot")]) == 0
    result = json.loads(capsys.readouterr().out)
    manifest = research.verify_snapshot(result["manifest"], result["data_snapshot_hash"])
    assert manifest["window_end"] == "20200107"
    assert research.main(["budget"]) == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_h1_example_pins_g0_dates_and_requires_real_hashes():
    path = Path(__file__).resolve().parents[1] / "user_data/configs/research-h1.example.json"
    assert path.exists(), "Ship the G0 split defaults, not moving timeranges"
    config = research.load_config(json.loads(path.read_text()))
    assert config["warmup"] == ["20180111", "20190113"]
    assert config["splits"] == {"dev": ["20190114", "20211231"],
                                "selection": ["20220101", "20240630"],
                                "eval": ["20240701", "20261218"]}
    assert config["snapshots"]["S1"]["hash"] == ""
    assert config["snapshots"]["S2"]["hash"] == ""


def test_selection_cannot_be_backfilled_after_forward_start(runtime, monkeypatch):
    runner, _ = runtime
    monkeypatch.setattr(runner.subprocess, "run", fake_engine)
    runner.preregister_evaluation("Toy", timeframe="1d")
    trial = runner.run_backtest("Toy", timeframe="1d")
    with pytest.raises(ValueError, match="deadline"):
        research.record_selection(trial["trial_id"])


def test_search_budget_accepts_other_repo_rows_with_plain_text_notes(experiment):
    with research.ledger() as conn:
        conn.execute("""INSERT INTO trial_ledger
            (trial_id, created_utc, hypothesis, repo, config_hash, split, params_json, verdict, notes)
            VALUES ('foreign', '2026-11-15T00:00:00Z', 'H2', 'alphasplice', 'foreign-hash',
                    'selection', '{}', 'ok', 'A human-readable note')""")
    assert research.search_budget()["H2"]["N"] == 1


def test_evaluation_exports_full_path_without_polluting_feedback(runtime, experiment, monkeypatch):
    runner, _ = runtime
    _, policy = experiment
    monkeypatch.setattr(runner.subprocess, "run", fake_engine)
    registration = select_for_test(runner)
    result = runner.run_evaluation("Toy", registration=registration, timeframe="1d")
    assert result["success"], result
    assert result["path_timerange"] == "20200102-20200113"
    assert result["timerange"] == "20200108-20200113"
    snapshot = json.loads(Path(result["slice_path"]).read_text())
    assert snapshot["opening_wallet"][0]["date"].startswith("2020-01-07")
    with research.ledger() as conn:
        row = conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (result["trial_id"],)).fetchone()
    assert row["verdict"] == "ok"
    assert row["data_snapshot_hash"] == policy["snapshots"]["S2"]["hash"]
    assert research.search_budget()["H1"]["N"] == 1
