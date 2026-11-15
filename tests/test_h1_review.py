"""Forward reviews keep one cumulative row per UTC review week."""
import json
from datetime import datetime, timezone
import pandas as pd
import pytest

pytest.importorskip("freqtrade")
from tests.test_h1_paper import paper_bot, clock, quotes, rows
from tests.test_h1_engine import ROOT


def test_forward_missing_daily_evidence_is_missed_and_idempotent(paper_bot, tmp_path, clock):
    from h1_paper import PaperAdapter
    from h1_review import write_forward
    from h1_service import strategy_hash
    import research
    cfg = paper_bot.config
    identity = {"selection": {"placeholder": False, "config_hash": strategy_hash(cfg["research_params"]), "trial_id": "selected-test"},
                "params": cfg["research_params"], "rules": cfg["research_h1"], "git_sha": "test"}
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    PaperAdapter(paper_bot, tmp_path / "journal.sqlite")
    now = datetime(2026, 12, 28, 1, tzinfo=timezone.utc)
    first = write_forward(tmp_path, "2026-12-27", now=now)
    second = write_forward(tmp_path, "2026-12-27", now=now)
    assert first == second
    with research.ledger() as conn:
        ledger_rows = list(conn.execute("SELECT * FROM trial_ledger WHERE split='forward'"))
    assert len(ledger_rows) == 1
    row = ledger_rows[0]
    assert row["window_start"] == "20261219"
    assert row["window_end"] == "20261227"
    assert row["verdict"] == "missed"
    assert row["net_return"] is None


def test_complete_forward_review_reconciles_real_fills_and_runs_backtest(paper_bot, tmp_path, clock):
    import numpy as np
    from h1_paper import PaperAdapter, reconcile
    from h1_review import write_forward
    from h1_service import strategy_hash
    from tests.test_h1_paper import enter
    bot = paper_bot
    clock.current = datetime(2026, 12, 19, tzinfo=timezone.utc)
    cfg = bot.config
    identity = {"selection": {"placeholder": False, "config_hash": strategy_hash(cfg["research_params"]), "trial_id": "selected-test"},
                "params": cfg["research_params"], "rules": cfg["research_h1"], "git_sha": "test"}
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    adapter.observe(quotes(bot, clock.current), clock.current)
    enter(bot, "BTC/USDT", 20., clock.current)
    dates = pd.date_range(end="2026-12-27", periods=500, tz="UTC")
    prices = 100 + np.sin(np.arange(500))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for pair in bot.active_pair_whitelist:
        pd.DataFrame(dict(date=dates, open=prices, high=prices, low=prices, close=prices, volume=1000.)).to_feather(
            data_dir / f"{pair.replace('/', '_')}-1d.feather")
    for day in pd.date_range("2026-12-19", "2026-12-27", tz="UTC"):
        close = float(prices[dates.get_loc(day)])
        adapter.save("batch", day.isoformat(), {"time": day.isoformat(), "status": "finished", "target_weights": dict.fromkeys(bot.active_pair_whitelist, 0.)})
        adapter.save("nav", day.isoformat(), {"date": day.isoformat(), "equity": 7997. + 20 * close})
        adapter.save("reference", day.isoformat(), dict.fromkeys(bot.active_pair_whitelist, 100.))
    row = write_forward(tmp_path, "2026-12-27", now=datetime(2026, 12, 28, 1, tzinfo=timezone.utc))
    assert row["verdict"] == "ok", row["notes"]
    assert row["n_trades_or_rebalances"] == 1
    assert row["net_return"] == pytest.approx((7997. + 20 * prices[-1]) / 10000 - 1)
    notes = json.loads(row["notes"])
    assert notes["comparison"]["total"] == 36
    assert notes["comparison"]["match_percent"] < 100
    assert notes["average_cost_per_side"] == pytest.approx(.0015)
    assert row["data_snapshot_hash"]

