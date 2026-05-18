"""Tests for R4 v2: hyperopt orchestrator wiring.

Covers registry.get_hyperopt_candidates (filter / sort / age) and
mark_hyperopt_outcome (state transitions + promotion).
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    import strategy_registry as sr
    monkeypatch.setattr(sr, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(sr, "REFLECTIONS_DIR", tmp_path / "reflections")
    sr.init_db()
    return sr


def _seed(sr, tmp_path, name, regime, verdict, status="retired",
          total_trades=0, sharpe=0.0, profit_pct=0.0, retired_at=None):
    """Insert a strategy + its latest backtest. retired_at defaults to now."""
    import sqlite3
    sid = sr.register_strategy(
        name=name, filepath=str(tmp_path / f"{name}.py"),
        thesis="t", target_regime=regime,
    )
    if total_trades or sharpe or profit_pct:
        sr.record_backtest(sid, {
            "total_trades": total_trades,
            "sharpe": sharpe,
            "profit_total_pct": profit_pct,
        })
    if status == "retired":
        sr.retire_strategy(sid, reason="seed", verdict=verdict)
    if retired_at:  # override the auto-now
        conn = sr.get_db()
        conn.execute("UPDATE strategies SET retired_at = ? WHERE id = ?", (retired_at, sid))
        conn.commit()
        conn.close()
    return sid


# ---------------------------------------------------------------------------
# get_hyperopt_candidates
# ---------------------------------------------------------------------------

def test_excludes_fail_backtest_crashes(isolated_registry, tmp_path):
    """Strategies that crashed in backtest must NOT be hyperopted — their code is broken."""
    sr = isolated_registry
    _seed(sr, tmp_path, "Crashed", "all", "FAIL_BACKTEST", total_trades=0)
    _seed(sr, tmp_path, "TooFew", "all", "FAIL_TOO_FEW", total_trades=3)
    _seed(sr, tmp_path, "Unprof", "all", "FAIL_UNPROFITABLE", total_trades=42)

    names = [c["name"] for c in sr.get_hyperopt_candidates(limit=10)]
    assert "Crashed" not in names
    assert "TooFew" in names
    assert "Unprof" in names


def test_sorts_by_trade_count_descending(isolated_registry, tmp_path):
    """More-trades-first: easier to rescue than 0-trade strategies."""
    sr = isolated_registry
    _seed(sr, tmp_path, "ZeroTrades", "all", "FAIL_TOO_FEW", total_trades=0)
    _seed(sr, tmp_path, "FiftyTrades", "all", "FAIL_UNPROFITABLE", total_trades=50)
    _seed(sr, tmp_path, "TenTrades", "all", "FAIL_UNPROFITABLE", total_trades=10)

    names = [c["name"] for c in sr.get_hyperopt_candidates(limit=10)]
    assert names == ["FiftyTrades", "TenTrades", "ZeroTrades"]


def test_respects_limit(isolated_registry, tmp_path):
    sr = isolated_registry
    for i in range(5):
        _seed(sr, tmp_path, f"S{i}", "all", "FAIL_TOO_FEW", total_trades=i)
    assert len(sr.get_hyperopt_candidates(limit=2)) == 2


def test_excludes_too_old(isolated_registry, tmp_path):
    """Don't re-hyperopt strategies retired weeks ago — that's already been judged."""
    sr = isolated_registry
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    _seed(sr, tmp_path, "Old", "all", "FAIL_UNPROFITABLE", total_trades=20, retired_at=old)
    _seed(sr, tmp_path, "Recent", "all", "FAIL_UNPROFITABLE", total_trades=10, retired_at=recent)

    names = [c["name"] for c in sr.get_hyperopt_candidates(limit=10, max_age_days=14)]
    assert names == ["Recent"]


def test_excludes_active_and_candidate(isolated_registry, tmp_path):
    """Only retired strategies are eligible."""
    sr = isolated_registry
    sid_active = sr.register_strategy(
        name="ActiveOne", filepath=str(tmp_path / "ActiveOne.py"), target_regime="all",
    )
    sr.promote_strategy(sid_active)
    sr.register_strategy(  # candidate
        name="StillCandidate", filepath=str(tmp_path / "StillCandidate.py"),
        target_regime="all",
    )
    _seed(sr, tmp_path, "RetiredOne", "all", "FAIL_UNPROFITABLE", total_trades=10)

    names = [c["name"] for c in sr.get_hyperopt_candidates(limit=10)]
    assert names == ["RetiredOne"]


# ---------------------------------------------------------------------------
# mark_hyperopt_outcome
# ---------------------------------------------------------------------------

def test_mark_outcome_no_promote_updates_verdict_only(isolated_registry, tmp_path):
    sr = isolated_registry
    sid = _seed(sr, tmp_path, "Tried", "all", "FAIL_UNPROFITABLE", total_trades=15)

    sr.mark_hyperopt_outcome(sid, verdict="HYPEROPT_NO_EDGE", reason="still bad")

    conn = sr.get_db()
    row = conn.execute(
        "SELECT status, failure_verdict, failure_reason, promoted_at FROM strategies WHERE id=?",
        (sid,),
    ).fetchone()
    conn.close()
    assert row["status"] == "retired"  # stayed retired
    assert row["failure_verdict"] == "HYPEROPT_NO_EDGE"
    assert row["failure_reason"] == "still bad"
    assert row["promoted_at"] is None


def test_mark_outcome_promote_flips_to_active(isolated_registry, tmp_path):
    sr = isolated_registry
    sid = _seed(sr, tmp_path, "Rescued", "all", "FAIL_UNPROFITABLE", total_trades=15)

    sr.mark_hyperopt_outcome(
        sid, verdict="HYPEROPT_PROMOTE", reason="rescued: 25 trades, 1.2% profit",
        promote=True,
    )

    conn = sr.get_db()
    row = conn.execute(
        "SELECT status, failure_verdict, failure_reason, retired_at, promoted_at "
        "FROM strategies WHERE id=?", (sid,),
    ).fetchone()
    conn.close()
    assert row["status"] == "active"
    assert row["failure_verdict"] == "HYPEROPT_PROMOTE"
    assert "rescued" in row["failure_reason"]
    assert row["retired_at"] is None  # cleared
    assert row["promoted_at"] is not None
