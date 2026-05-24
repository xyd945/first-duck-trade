"""Tests for the deployment-lifecycle schema migration (Phase 1).

We're splitting overloaded `status` into research_status + deployment_status
so the registry can be honest about "passed gates" vs "actually trading".
Phase 1 is additive only — `status` is preserved through Phase 4 as a
compatibility shim. These tests pin the migration's backfill behavior,
the new query helpers, and the eligibility filters that the Phase 2
reconciler will read.
"""

import importlib
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    import strategy_registry as reg
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(reg, "DB_PATH", db_path)
    reg.init_db()
    return reg


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migration_adds_research_status_and_deployment_columns(isolated_registry):
    reg = isolated_registry
    conn = sqlite3.connect(reg.DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(strategies)")]
    for new_col in (
        "research_status", "deployment_status", "deployed_at",
        "last_deployment_error", "deployment_blocked_until",
    ):
        assert new_col in cols, f"missing migrated column {new_col}"


def test_migration_adds_backtest_data_end_at(isolated_registry):
    reg = isolated_registry
    conn = sqlite3.connect(reg.DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(backtest_results)")]
    assert "backtest_data_end_at" in cols


def test_migration_backfills_research_status_from_legacy_status(tmp_path, monkeypatch):
    """Existing `status='active'` row → `research_status='approved'` after
    migration. The legacy column stays populated for the compatibility
    shim period."""
    import strategy_registry as reg
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(reg, "DB_PATH", db_path)

    # Build a registry, manually insert a row with the OLD pre-migration
    # schema state (research_status absent), then re-run init_db() to
    # exercise the backfill.
    reg.init_db()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO strategies (name, filepath, status, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("LegacyActive", "/tmp/legacy.py", "active", datetime.now(timezone.utc).isoformat()),
    )
    conn.execute(
        "INSERT INTO strategies (name, filepath, status, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("LegacyRetired", "/tmp/retired.py", "retired", datetime.now(timezone.utc).isoformat()),
    )
    # Clear research_status to simulate a pre-migration row
    conn.execute("UPDATE strategies SET research_status='' WHERE name IN ('LegacyActive','LegacyRetired')")
    conn.commit()
    conn.close()

    # Re-run init_db — backfill should map active → approved, retired → retired
    reg.init_db()

    conn = sqlite3.connect(db_path)
    rows = dict(conn.execute(
        "SELECT name, research_status FROM strategies "
        "WHERE name IN ('LegacyActive', 'LegacyRetired')"
    ).fetchall())
    conn.close()
    assert rows["LegacyActive"] == "approved"
    assert rows["LegacyRetired"] == "retired"


def test_migration_default_deployment_status_is_not_deployed(isolated_registry):
    """New rows inserted via register_strategy after migration must
    default to deployment_status='not_deployed', not None."""
    reg = isolated_registry
    reg.register_strategy(
        name="FreshOne", filepath="/tmp/x.py",
        thesis="t", target_regime="ranging",
        generation_id="g1", archetype="mean_reversion",
    )
    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT deployment_status FROM strategies WHERE name='FreshOne'"
    ).fetchone()
    conn.close()
    assert row[0] == "not_deployed"


def test_migration_is_idempotent(isolated_registry):
    """init_db can be called repeatedly without raising 'duplicate column'
    or corrupting data. Important because the module runs init_db on
    import, and tests + the orchestrator + the monitor all import it."""
    reg = isolated_registry
    for _ in range(3):
        reg.init_db()  # must not raise


# ---------------------------------------------------------------------------
# get_deployment_eligible — the hard-filter contract
# ---------------------------------------------------------------------------

def _seed_strategy_with_backtest(
    reg, *, name, sharpe=1.0, profit=2.0, drawdown=5.0, trades=30,
    backtest_age_days=1, data_age_days=1,
    research_status="approved", deployment_status="not_deployed",
    blocked_until=None,
):
    """Helper: insert a strategy + a backtest_results row with explicit
    timing knobs. The migration-friendly path."""
    conn = sqlite3.connect(reg.DB_PATH)
    now = datetime.now(timezone.utc)
    bt_at = (now - timedelta(days=backtest_age_days)).isoformat()
    data_end = (now - timedelta(days=data_age_days)).isoformat()

    conn.execute(
        "INSERT INTO strategies (name, filepath, research_status, "
        "deployment_status, deployment_blocked_until, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, f"/tmp/{name}.py", research_status, deployment_status,
         blocked_until, now.isoformat()),
    )
    sid = conn.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()[0]
    conn.execute(
        "INSERT INTO backtest_results (strategy_id, sharpe, profit_total_pct, "
        "max_drawdown_pct, total_trades, backtest_data_end_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, sharpe, profit, drawdown, trades, data_end, bt_at),
    )
    conn.commit()
    conn.close()


def test_eligibility_passes_clean_strategy(isolated_registry):
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="Clean")
    eligible = reg.get_deployment_eligible()
    assert any(r["name"] == "Clean" for r in eligible)


def test_eligibility_rejects_too_few_trades(isolated_registry):
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="Sparse", trades=10)  # < 20
    eligible = reg.get_deployment_eligible()
    assert all(r["name"] != "Sparse" for r in eligible)


def test_eligibility_rejects_negative_profit(isolated_registry):
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="Loser", profit=-1.0)
    assert all(r["name"] != "Loser" for r in reg.get_deployment_eligible())


def test_eligibility_rejects_negative_sharpe(isolated_registry):
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="BadRisk", sharpe=-0.5)
    assert all(r["name"] != "BadRisk" for r in reg.get_deployment_eligible())


def test_eligibility_rejects_deep_drawdown(isolated_registry):
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="DeepDD", drawdown=20.0)  # > 15
    assert all(r["name"] != "DeepDD" for r in reg.get_deployment_eligible())


def test_eligibility_rejects_stale_backtest_run(isolated_registry):
    """backtest run 40 days ago — older than the 30-day default."""
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="StaleRun", backtest_age_days=40)
    assert all(r["name"] != "StaleRun" for r in reg.get_deployment_eligible())


def test_eligibility_rejects_stale_data_even_with_recent_run(isolated_registry):
    """Codex's specific concern: a backtest run yesterday on candles that
    end 90 days ago should fail eligibility. last_backtest_at fresh ≠
    market data fresh."""
    reg = isolated_registry
    _seed_strategy_with_backtest(
        reg, name="StaleData", backtest_age_days=1, data_age_days=90,
    )
    eligible = reg.get_deployment_eligible()
    assert all(r["name"] != "StaleData" for r in eligible)


def test_eligibility_rejects_unapproved_research_status(isolated_registry):
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="Candidate", research_status="candidate")
    assert all(r["name"] != "Candidate" for r in reg.get_deployment_eligible())


def test_eligibility_includes_currently_deployed_so_winners_stay_picked(isolated_registry):
    """A currently-deployed strategy MUST appear in the eligible pool —
    otherwise the per-tick re-selection picks a different top-3, the
    reconciler sees the previous winner as 'no longer desired', stops
    its container, and on the very next tick (when the strategy is now
    'stopped' and so eligible again) picks it back up. Observed live
    during the Phase 3 shakedown: id 127 churned every 5 minutes for
    hours, never running long enough to fire any entries.

    The idempotency the reconciler relies on: if a deployed strategy is
    in desired AND its container is already running, intended_starts is
    empty for it (no clobber) and intended_stops is empty (still desired).
    Quiet steady state."""
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="Already", deployment_status="deployed")
    names = [r["name"] for r in reg.get_deployment_eligible()]
    assert "Already" in names, (
        f"deployed-status row must remain in eligible pool to prevent churn; "
        f"got {names}"
    )


def test_deployed_winner_survives_consecutive_eligibility_reads(isolated_registry):
    """End-to-end shape of the no-churn invariant: a deployed strategy
    that's the top sharpe in the pool remains the top sharpe after
    being deployed. The reconciler can call get_deployment_eligible
    twice in a row + sort by sharpe + take top-N and get the SAME
    answer both times. That's what prevents the start/stop ping-pong."""
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="Best",   sharpe=2.5,
                                 deployment_status="deployed")  # already up
    _seed_strategy_with_backtest(reg, name="Second", sharpe=1.5)
    _seed_strategy_with_backtest(reg, name="Third",  sharpe=1.0)

    # Two reads in a row — should be identical (deterministic), with
    # 'Best' present in both.
    first  = sorted(r["name"] for r in reg.get_deployment_eligible())
    second = sorted(r["name"] for r in reg.get_deployment_eligible())
    assert first == second
    assert "Best" in first


def test_eligibility_respects_deployment_blocked_until_cooldown(isolated_registry):
    """A risk-stopped strategy with blocked_until in the future is NOT
    eligible — prevents the reconciler from instantly redeploying
    something it just stopped for a safety reason."""
    reg = isolated_registry
    future = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    _seed_strategy_with_backtest(
        reg, name="Cooling", deployment_status="stopped", blocked_until=future,
    )
    assert all(r["name"] != "Cooling" for r in reg.get_deployment_eligible())


def test_eligibility_accepts_after_cooldown_expires(isolated_registry):
    """Once blocked_until is in the past, the strategy is eligible again."""
    reg = isolated_registry
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _seed_strategy_with_backtest(
        reg, name="CooledOff", deployment_status="stopped", blocked_until=past,
    )
    assert any(r["name"] == "CooledOff" for r in reg.get_deployment_eligible())


# ---------------------------------------------------------------------------
# get_currently_deployed
# ---------------------------------------------------------------------------

def test_currently_deployed_returns_only_deployed_rows(isolated_registry):
    reg = isolated_registry
    _seed_strategy_with_backtest(reg, name="Running1", deployment_status="deployed")
    _seed_strategy_with_backtest(reg, name="Running2", deployment_status="deployed")
    _seed_strategy_with_backtest(reg, name="Standby",  deployment_status="not_deployed")
    _seed_strategy_with_backtest(reg, name="Halted",   deployment_status="stopped")

    deployed = reg.get_currently_deployed()
    names = {r["name"] for r in deployed}
    assert names == {"Running1", "Running2"}
