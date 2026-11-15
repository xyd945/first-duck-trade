"""Forward reviews keep one cumulative row per UTC review week."""
import json
from datetime import datetime, timezone
import pandas as pd
import pytest

pytest.importorskip("freqtrade")
from tests.test_h1_paper import paper_bot, clock, quotes, rows
from tests.test_h1_engine import ROOT


def review_cli(directory, through="2026-12-27"):
    import subprocess
    import sys
    script = '''import sys
from datetime import datetime, timezone
sys.path.insert(0, "user_data/scripts")
import h1_review
class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 12, 28, 1, tzinfo=timezone.utc)
h1_review.datetime = Clock
raise SystemExit(h1_review.main())
'''
    return subprocess.run([sys.executable, "-c", script, "--directory", str(directory), "--through", through],
                          cwd=ROOT, text=True, capture_output=True, timeout=30)


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
    notes = json.loads(row["notes"])
    assert notes["comparison"]["total"] == 36
    assert notes["comparison"]["matched"] == 0
    assert len(notes["missing_days"]) == 9


@pytest.mark.parametrize("case", ["ok", "missing", "late", "error", "nav", "cost", "invalid", "broken"])
def test_complete_forward_review_reconciles_real_fills_and_runs_backtest(paper_bot, tmp_path, clock, case):
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
    key = "2026-12-19T00:00:00+00:00"
    if case in ("missing", "nav"):
        import sqlite3
        with sqlite3.connect(tmp_path / "journal.sqlite") as conn:
            conn.execute("DELETE FROM paper_records WHERE kind=? AND id=?", ("batch" if case == "missing" else "nav", key))
    elif case in ("late", "error"):
        batch = adapter.get("batch", key)
        batch.update(status="missed" if case == "late" else "finished", risk_reason=case if case == "late" else "ok")
        adapter.save("batch", key, batch)
        if case == "error":
            adapter.save("error", key, {"time": key, "error": "transient failure before recovery"})
    elif case == "cost":
        adapter.save("reference", key, dict.fromkeys(bot.active_pair_whitelist, 99.))
    elif case == "invalid":
        identity["selection"]["config_hash"] = "changed"
        (tmp_path / "identity.json").write_text(json.dumps(identity))
    elif case == "broken":
        (data_dir / "BTC_USDT-1d.feather").write_bytes(b"corrupt fixture")
    output = review_cli(tmp_path)
    assert output.stdout, output.stderr
    row = json.loads(output.stdout)
    assert row["verdict"] == {"missing": "missed", "late": "missed", "error": "error", "nav": "missed",
                              "invalid": "invalid", "broken": "error"}.get(case, "ok"), row["verdict"]
    assert output.returncode == (0 if row["verdict"] == "ok" else 1)
    notes = json.loads(row["notes"])
    assert notes["comparison"]["total"] == 36
    assert notes["comparison"]["match_percent"] < 100
    if case in ("invalid", "broken"):
        assert row["net_return"] is None
        assert not notes["comparison_available"]
        return
    assert notes["comparison_available"]
    if case in ("late", "error", "missing"):
        failed = [r for r in notes["comparison"]["details"] if r["time"] == key]
        assert len(failed) == 4
        assert not any(r["match"] for r in failed)
    if case == "nav":
        assert row["net_return"] is None
        return
    assert row["n_trades_or_rebalances"] == 1
    assert row["net_return"] == pytest.approx((7997. + 20 * prices[-1]) / 10000 - 1)
    assert notes["cost_check_pass"] is (case != "cost")
    assert notes["average_cost_per_side"] == pytest.approx((1 + .15) / 99 if case == "cost" else .0015)
    assert row["data_snapshot_hash"]


def test_weekly_cli_writes_and_returns_the_same_row(tmp_path):
    import os
    from h1_service import strategy_hash
    from tests.test_h1_service import config
    cfg = config()
    (tmp_path / "identity.json").write_text(json.dumps({
        "selection": {"placeholder": True, "config_hash": strategy_hash(cfg["research_params"])},
        "params": cfg["research_params"], "rules": cfg["research_h1"], "git_sha": "test"}))
    first = review_cli(tmp_path)
    assert first.stdout.strip(), first.stderr
    row = json.loads(first.stdout)
    assert first.returncode == 1
    assert row["split"] == "forward"
    assert row["verdict"] == "invalid"
    second = review_cli(tmp_path)
    assert json.loads(second.stdout) == row
    import sqlite3
    with sqlite3.connect(os.environ["TRIAL_LEDGER_PATH"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()[0] == 1


@pytest.mark.parametrize("through", ["2026-12-18", "2026-12-28", "2026-12-27T12:00:00", "not-a-date"])
def test_cli_refuses_invalid_or_incomplete_days(tmp_path, through):
    result = review_cli(tmp_path, through)
    assert result.returncode == 2
    assert "identity.json" not in result.stderr


def test_interrupted_review_retains_denominator_and_cannot_be_backfilled(paper_bot, tmp_path, monkeypatch):
    import h1_review
    import research
    from h1_service import strategy_hash
    cfg = paper_bot.config
    (tmp_path / "identity.json").write_text(json.dumps({
        "selection": {"placeholder": False, "config_hash": strategy_hash(cfg["research_params"])},
        "params": cfg["research_params"], "rules": cfg["research_h1"], "git_sha": "test"}))
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(h1_review, "records", interrupted)
    now = datetime(2026, 12, 28, 1, tzinfo=timezone.utc)
    with pytest.raises(KeyboardInterrupt):
        h1_review.write_forward(tmp_path, "2026-12-27", now=now)
    row = h1_review.write_forward(tmp_path, "2026-12-27", now=now)
    assert row["verdict"] == "error"
    assert json.loads(row["notes"])["comparison"]["total"] == 36
    with pytest.raises(ValueError, match="already"):
        h1_review.write_forward(tmp_path, "2026-12-28", now=datetime(2026, 12, 29, tzinfo=timezone.utc))
