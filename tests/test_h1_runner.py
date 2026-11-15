"""Full ledger-to-engine-to-independent-accounting integration, synthetic prices only."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("freqtrade")
from tests.test_h1_engine import ROOT, engine_config, offline_exchange, prices


@pytest.fixture
def frozen_run(tmp_path):
    import research
    base = tmp_path / "user_data"
    (base / "strategies/candidates").mkdir(parents=True)
    shutil.copy2(ROOT / "user_data/strategies/candidates/H1Trend.py", base / "strategies/candidates/H1Trend.py")
    source = tmp_path / "source"
    source.mkdir()
    cfg = engine_config("H1BuyHold")
    for pair, frame in prices(cfg["exchange"]["pair_whitelist"]).items():
        frame.to_feather(source / f"{pair.replace('/', '_')}-1d.feather")
    manifest = research.freeze_snapshot(source, tmp_path / "frozen", "20180111", "20190204")
    policy = json.loads((ROOT / "user_data/configs/research-h1.example.json").read_text())
    policy["splits"] = {"dev": ["20190114", "20190204"], "selection": ["20190205", "20240630"], "eval": ["20240701", "20261218"]}
    policy["snapshots"]["S1"] = {"manifest": str(manifest), "hash": json.loads(manifest.read_text())["data_snapshot_hash"]}
    for k in ("user_data_dir", "datadir", "strategy_path"):
        cfg[k] = str(cfg[k])
    cfg["research_costs"] = policy["costs"]
    cfg_path = base / "engine.json"
    cfg_path.write_text(json.dumps(cfg))
    return base, policy, cfg_path, cfg


def test_real_export_populates_net_ledger_and_rerun(frozen_run):
    import research
    from h1_backtest import run_development
    base, policy, path, cfg = frozen_run
    exchange = offline_exchange(cfg)
    try:
        result = run_development(base, policy, "H1BuyHold", path, exchange=exchange)
    finally:
        exchange.close()
    assert result["success"], result
    assert result["total_trades"] == 1
    # Invest 10,000 / 1.003 at 100. Final BTC price is 80. No terminal sell fee.
    assert result["net_return"] == pytest.approx(.8 / 1.003 - 1)
    assert result["cost_total"] == pytest.approx(10000 * .003 / 1.003)
    with research.ledger() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM trial_ledger")]
    assert len(rows) == 1
    assert rows[0]["verdict"] == "baseline"
    assert rows[0]["net_return"] == result["net_return"]
    assert Path(json.loads(rows[0]["notes"])["params"]["batch_path"]).exists()
    assert research.search_budget()["H1"]["N"] == 0


@pytest.mark.parametrize("strategy", ["H1Trend", "H1SimpleTrend", "H1EqualWeight"])
def test_resizing_exports_reconcile_after_same_day_sells(frozen_run, strategy):
    import research
    from h1_backtest import run_development
    base, policy, path, cfg = frozen_run
    exchange = offline_exchange(cfg)
    try:
        result = run_development(base, policy, strategy, path, exchange=exchange)
    finally:
        exchange.close()
    assert result["success"], result
    with research.ledger() as conn:
        row = dict(conn.execute("SELECT * FROM trial_ledger").fetchone())
    assert row["net_return"] == result["net_return"]
    assert row["verdict"] == ("ok" if strategy == "H1Trend" else "baseline")


def test_full_registered_grid_is_logged_and_identical_repeat_is_rerun(frozen_run):
    import research
    from H1Trend import grid_params
    from h1_backtest import run_development
    base, policy, path, cfg = frozen_run
    grid = grid_params(policy)
    for params in [*grid.values(), grid["S60"]]:
        policy["params"] = params
        exchange = offline_exchange(cfg)
        try:
            result = run_development(base, policy, "H1Trend", path, exchange=exchange)
        finally:
            exchange.close()
        assert result["success"], result
    with research.ledger() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM trial_ledger ORDER BY rowid")]
    assert len(rows) == 10
    assert len({r["config_hash"] for r in rows}) == 9
    assert [r["verdict"] for r in rows] == ["ok"] * 9 + ["rerun"]
    assert research.search_budget()["H1"]["N"] == 0


def test_failure_before_engine_still_gets_one_ledger_row(frozen_run):
    import research
    from h1_backtest import run_development
    base, policy, path, _ = frozen_run
    policy["snapshots"]["S1"]["hash"] = "tampered"
    result = run_development(base, policy, "H1Trend", path)
    assert not result["success"]
    assert "hash" in result["error"]
    with research.ledger() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM trial_ledger")]
    assert len(rows) == 1 and rows[0]["verdict"] == "error"


def test_engine_rejects_unlogged_calls():
    from h1_backtest import H1Backtesting
    with pytest.raises(ValueError, match="ledger"):
        H1Backtesting(engine_config())
