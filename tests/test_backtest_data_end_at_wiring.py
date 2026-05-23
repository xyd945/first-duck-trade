"""Tests for wiring backtest_data_end_at end-to-end.

Phase 1 added the column. This PR populates it:
  * parse_backtest_artifact reads backtest_end from the result JSON
  * parse_backtest_output extracts it from the "Backtested ... -> END" line
  * record_backtest persists it
  * a backfill migration fills it for existing pre-PR rows from `timerange`

The driving need: the existing 10 "active" strategies had empty
backtest_data_end_at after Phase 1 migration, which failed the
deployment-eligibility data-freshness filter even though several of
them had been backtested days ago on current data. The backfill plus
the new write path makes them deployment-eligible.
"""

import json
import sqlite3
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# parse_backtest_artifact — JSON path
# ---------------------------------------------------------------------------

def test_artifact_parses_backtest_end_field(tmp_path):
    """The JSON Freqtrade writes contains 'backtest_end' as a typed
    timestamp string. parse_backtest_artifact must surface it as
    backtest_data_end_at on the result dict."""
    from backtest_runner import parse_backtest_artifact

    payload = {
        "strategy": {"Foo": {
            "total_trades": 30, "profit_total": 0.02, "profit_total_abs": 20,
            "profit_mean": 0.001, "max_drawdown_account": 0.05,
            "max_drawdown_abs": 5.0, "sharpe": 1.5, "sortino": 1.2,
            "profit_factor": 1.5, "winrate": 0.6, "holding_avg": "1d",
            "backtest_days": 90, "starting_balance": 1000,
            "backtest_end": "2026-05-19 00:00:00",
        }},
        "strategy_comparison": [],
    }
    zp = tmp_path / "r.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("r.json", json.dumps(payload))
    r = parse_backtest_artifact(zp, "Foo")
    assert r["backtest_data_end_at"] == "2026-05-19 00:00:00"


def test_artifact_handles_missing_backtest_end_gracefully(tmp_path):
    """Older Freqtrade JSONs without backtest_end → empty string,
    not crash. The backfill or fallback path takes care of it."""
    from backtest_runner import parse_backtest_artifact
    payload = {"strategy": {"Foo": {
        "total_trades": 0, "profit_total": 0, "profit_total_abs": 0,
        "profit_mean": 0, "max_drawdown_account": 0, "max_drawdown_abs": 0,
        "sharpe": 0, "sortino": 0, "profit_factor": 0, "winrate": 0,
        "holding_avg": "", "backtest_days": 0, "starting_balance": 0,
    }}, "strategy_comparison": []}
    zp = tmp_path / "r.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("r.json", json.dumps(payload))
    r = parse_backtest_artifact(zp, "Foo")
    assert r["backtest_data_end_at"] == ""


# ---------------------------------------------------------------------------
# parse_backtest_output — console regex path
# ---------------------------------------------------------------------------

def test_console_parser_extracts_end_date_from_backtested_line():
    """The console output has 'Backtested 2026-02-18 00:00:00 -> 2026-05-19 00:00:00 | ...'
    — we already match this to compute backtest_days; now also stash the
    end timestamp."""
    from backtest_runner import parse_backtest_output
    output = (
        "Result for strategy X\n"
        "│    TOTAL │     5 │  0.5 │  10.000 │  1.0 │\n"
        "Backtested 2026-02-18 00:00:00 -> 2026-05-19 00:00:00 | Max open trades: 3\n"
    )
    r = parse_backtest_output(output, "X")
    assert r["backtest_data_end_at"] == "2026-05-19 00:00:00"


def test_console_parser_omits_end_when_period_line_absent():
    """If the period line is missing entirely, the field is absent
    rather than defaulting to a stale value."""
    from backtest_runner import parse_backtest_output
    output = "Result for strategy X\n│    TOTAL │     0 │  0.0 │  0.000 │  0.0 │\n"
    r = parse_backtest_output(output, "X")
    # Either absent or empty — both acceptable; never silently a stale date
    assert r.get("backtest_data_end_at", "") == ""


# ---------------------------------------------------------------------------
# record_backtest — persistence
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    import strategy_registry as reg
    monkeypatch.setattr(reg, "DB_PATH", tmp_path / "test.db")
    reg.init_db()
    return reg


def test_record_backtest_persists_backtest_data_end_at(isolated_registry):
    """The new write path: record_backtest sees backtest_data_end_at on
    the result dict and stores it in the DB column."""
    reg = isolated_registry
    reg.register_strategy(name="X", filepath="/tmp/x.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("X")["id"]

    reg.record_backtest(sid, {
        "total_trades": 30, "sharpe": 1.5, "profit_total_pct": 2.0,
        "backtest_data_end_at": "2026-05-22 18:00:00",
    })

    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT backtest_data_end_at FROM backtest_results WHERE strategy_id=?",
        (sid,),
    ).fetchone()
    conn.close()
    assert row[0] == "2026-05-22 18:00:00"


def test_record_backtest_persists_empty_when_field_absent(isolated_registry):
    """Backwards compat: a caller that doesn't pass the field gets empty
    string (which the backfill or eligibility filter handles)."""
    reg = isolated_registry
    reg.register_strategy(name="X", filepath="/tmp/x.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("X")["id"]
    reg.record_backtest(sid, {"total_trades": 30, "sharpe": 1.5})

    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT backtest_data_end_at FROM backtest_results WHERE strategy_id=?",
        (sid,),
    ).fetchone()
    conn.close()
    assert row[0] == ""


# ---------------------------------------------------------------------------
# Backfill migration
# ---------------------------------------------------------------------------

def test_backfill_uses_timerange_end_when_available(tmp_path, monkeypatch):
    """The driving case: a pre-PR row has timerange='20260218-20260519'
    and backtest_data_end_at=''. Backfill should populate from the
    timerange end (2026-05-19)."""
    import strategy_registry as reg
    monkeypatch.setattr(reg, "DB_PATH", tmp_path / "test.db")
    reg.init_db()

    # Insert a row directly with the old (post-Phase-1, pre-this-PR) shape:
    # column exists but is empty, timerange is populated.
    reg.register_strategy(name="Pre", filepath="/tmp/p.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("Pre")["id"]
    conn = sqlite3.connect(reg.DB_PATH)
    conn.execute(
        "INSERT INTO backtest_results (strategy_id, timerange, sharpe, "
        "backtest_data_end_at, created_at) VALUES (?, ?, ?, ?, ?)",
        (sid, "20260218-20260519", 1.5, "",
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    # Re-run init_db to trigger the backfill
    reg.init_db()

    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT backtest_data_end_at FROM backtest_results WHERE strategy_id=?",
        (sid,),
    ).fetchone()
    conn.close()
    # Backfill formats YYYYMMDD as 'YYYY-MM-DD 00:00:00'
    assert row[0] == "2026-05-19 00:00:00"


def test_backfill_falls_back_to_created_at_when_timerange_missing(tmp_path, monkeypatch):
    import strategy_registry as reg
    monkeypatch.setattr(reg, "DB_PATH", tmp_path / "test.db")
    reg.init_db()
    reg.register_strategy(name="NoTR", filepath="/tmp/n.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("NoTR")["id"]
    fallback_time = "2026-05-15T12:00:00+00:00"
    conn = sqlite3.connect(reg.DB_PATH)
    conn.execute(
        "INSERT INTO backtest_results (strategy_id, timerange, sharpe, "
        "backtest_data_end_at, created_at) VALUES (?, ?, ?, ?, ?)",
        (sid, "", 1.5, "", fallback_time),
    )
    conn.commit()
    conn.close()

    reg.init_db()

    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT backtest_data_end_at FROM backtest_results WHERE strategy_id=?",
        (sid,),
    ).fetchone()
    conn.close()
    assert row[0] == fallback_time


def test_backfill_idempotent_does_not_overwrite_existing_values(tmp_path, monkeypatch):
    """Running migration a second time must not blow away values from
    the regular write path."""
    import strategy_registry as reg
    monkeypatch.setattr(reg, "DB_PATH", tmp_path / "test.db")
    reg.init_db()
    reg.register_strategy(name="Set", filepath="/tmp/s.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("Set")["id"]

    explicit = "2026-05-22 12:00:00"
    conn = sqlite3.connect(reg.DB_PATH)
    conn.execute(
        "INSERT INTO backtest_results (strategy_id, timerange, sharpe, "
        "backtest_data_end_at, created_at) VALUES (?, ?, ?, ?, ?)",
        (sid, "20260218-20260519", 1.5, explicit,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    reg.init_db()  # backfill should NOT touch this row

    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT backtest_data_end_at FROM backtest_results WHERE strategy_id=?",
        (sid,),
    ).fetchone()
    conn.close()
    assert row[0] == explicit  # untouched


# ---------------------------------------------------------------------------
# End-to-end: backfill unblocks eligibility for existing approved row
# ---------------------------------------------------------------------------

def test_backfilled_row_passes_deployment_eligibility_filter(tmp_path, monkeypatch):
    """The integration test that actually motivates this PR: a row that
    pre-dates the column and is `research_status='approved'` becomes
    deployment-eligible after the backfill runs."""
    import strategy_registry as reg
    monkeypatch.setattr(reg, "DB_PATH", tmp_path / "test.db")
    reg.init_db()
    reg.register_strategy(name="Winner", filepath="/tmp/w.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("Winner")["id"]

    # Simulate a pre-PR backtest_results row with empty backtest_data_end_at
    # and a recent timerange end
    now = datetime.now(timezone.utc)
    recent_end_yyyymmdd = (now - timedelta(days=2)).strftime("%Y%m%d")
    range_start = (now - timedelta(days=92)).strftime("%Y%m%d")
    timerange = f"{range_start}-{recent_end_yyyymmdd}"
    conn = sqlite3.connect(reg.DB_PATH)
    conn.execute(
        "INSERT INTO backtest_results (strategy_id, timerange, sharpe, "
        "profit_total_pct, max_drawdown_pct, total_trades, "
        "backtest_data_end_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, timerange, 1.5, 3.0, 5.0, 30, "", now.isoformat()),
    )
    conn.execute(
        "UPDATE strategies SET research_status='approved' WHERE id=?",
        (sid,),
    )
    conn.commit()
    conn.close()

    # Before backfill: eligibility filter rejects because data_end_at is empty
    eligible_before = reg.get_deployment_eligible()
    assert all(r["name"] != "Winner" for r in eligible_before), (
        "test setup wrong — Winner should NOT be eligible with empty data_end_at"
    )

    # Run backfill
    reg.init_db()

    # After backfill: Winner is eligible because data_end_at is now
    # populated from the recent timerange end
    eligible_after = reg.get_deployment_eligible()
    assert any(r["name"] == "Winner" for r in eligible_after), (
        f"Winner should now be eligible after backfill. Got: "
        f"{[r['name'] for r in eligible_after]}"
    )
