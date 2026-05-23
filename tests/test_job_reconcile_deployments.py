"""Tests for the Phase 2 observe-only reconciler job + drift log.

The job MUST be observation-only for Phase 2 — no Docker SDK mutation
even when RECONCILER_ACTING=true is set (Phase 3 wires that flag to
real action). These tests pin both:

  * the drift_log table records each tick correctly
  * the job's logic computes the right desired-vs-running diff
  * the job never reaches the real Docker daemon
  * RECONCILER_ACTING is read + plumbed but does NOT change behavior
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    import strategy_registry as reg
    monkeypatch.setattr(reg, "DB_PATH", tmp_path / "test.db")
    reg.init_db()
    return reg


def _seed_strategy(reg, *, name, sharpe=1.0, profit=2.0, drawdown=5.0,
                   trades=30, research_status="approved",
                   deployment_status="not_deployed", path=""):
    conn = sqlite3.connect(reg.DB_PATH)
    now = datetime.now(timezone.utc)
    bt_ts = now.isoformat()
    conn.execute(
        "INSERT INTO strategies (name, filepath, research_status, "
        "deployment_status, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, f"/tmp/{name}.py", research_status, deployment_status, bt_ts),
    )
    sid = conn.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()[0]
    conn.execute(
        "INSERT INTO backtest_results (strategy_id, sharpe, profit_total_pct, "
        "max_drawdown_pct, total_trades, backtest_data_end_at, "
        "trades_export_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, sharpe, profit, drawdown, trades, bt_ts, path, bt_ts),
    )
    conn.commit()
    conn.close()
    return sid


# ---------------------------------------------------------------------------
# Drift log table
# ---------------------------------------------------------------------------

def test_drift_log_table_exists_after_migration(isolated_registry):
    reg = isolated_registry
    conn = sqlite3.connect(reg.DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    conn.close()
    assert "deployment_drift_log" in tables


def test_record_drift_log_returns_id_and_persists(isolated_registry):
    reg = isolated_registry
    rid = reg.record_drift_log(
        desired=[{"id": 1, "name": "A"}],
        running=[{"name": "ft-deployed-old", "status": "running"}],
        intended_starts=[{"strategy_id": 1, "name": "A"}],
        intended_stops=[{"container_name": "ft-deployed-old"}],
        skipped_eligible=[],
        reconciler_acting=False,
        notes="test",
    )
    assert rid > 0

    rows = reg.get_recent_drift_logs(limit=10)
    assert len(rows) == 1
    assert rows[0]["reconciler_acting"] == 0
    assert rows[0]["desired"][0]["name"] == "A"
    assert rows[0]["running"][0]["name"] == "ft-deployed-old"
    assert rows[0]["intended_starts"][0]["name"] == "A"
    assert rows[0]["intended_stops"][0]["container_name"] == "ft-deployed-old"
    assert rows[0]["notes"] == "test"


def test_get_recent_drift_logs_returns_newest_first(isolated_registry):
    reg = isolated_registry
    for i in range(3):
        reg.record_drift_log(
            desired=[], running=[], intended_starts=[],
            intended_stops=[], skipped_eligible=[],
            reconciler_acting=False, notes=f"tick-{i}",
        )
    rows = reg.get_recent_drift_logs(limit=10)
    notes = [r["notes"] for r in rows]
    assert notes == ["tick-2", "tick-1", "tick-0"]


# ---------------------------------------------------------------------------
# job_reconcile_deployments — observe-only behavior
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_deployment_manager():
    """Patch DeploymentManager so the job doesn't reach the real docker
    socket. Returns the MagicMock that stands in for the manager."""
    with patch("deployment_manager.DeploymentManager") as cls:
        instance = MagicMock()
        instance.list_deployed.return_value = []
        cls.return_value = instance
        yield instance


def test_job_runs_without_eligible_strategies(isolated_registry, mock_deployment_manager,
                                              monkeypatch):
    """Empty eligible pool: job runs cleanly, records one drift entry,
    no intended starts/stops."""
    import orchestrator
    monkeypatch.setattr(orchestrator, "INSTANCES", {})  # silence health-side noise
    orchestrator.job_reconcile_deployments()

    rows = isolated_registry.get_recent_drift_logs(limit=1)
    assert len(rows) == 1
    assert rows[0]["intended_starts"] == []
    assert rows[0]["intended_stops"] == []


def test_job_intended_starts_when_desired_strategy_not_running(
    isolated_registry, mock_deployment_manager, monkeypatch
):
    """Eligible strategy with no matching running container → would-start."""
    _seed_strategy(isolated_registry, name="Winner", sharpe=2.0)
    mock_deployment_manager.list_deployed.return_value = []

    import orchestrator
    orchestrator.job_reconcile_deployments()

    rows = isolated_registry.get_recent_drift_logs(limit=1)
    assert rows[0]["desired"][0]["name"] == "Winner"
    assert len(rows[0]["intended_starts"]) == 1
    assert rows[0]["intended_starts"][0]["name"] == "Winner"
    assert rows[0]["intended_starts"][0]["container_name"] == "ft-deployed-winner"


def test_job_intended_stops_when_running_not_in_desired(
    isolated_registry, mock_deployment_manager, monkeypatch
):
    """Container running for a strategy that's NOT in the desired set
    (no longer approved, or beaten by a better one) → would-stop."""
    mock_deployment_manager.list_deployed.return_value = [{
        "name": "ft-deployed-orphan",
        "status": "running",
        "id": "abc",
        "labels": {},
        "strategy_id": 99,
        "strategy_name": "OrphanStrategy",
        "deployment_generation": 1,
    }]

    import orchestrator
    orchestrator.job_reconcile_deployments()

    rows = isolated_registry.get_recent_drift_logs(limit=1)
    assert len(rows[0]["intended_stops"]) == 1
    assert rows[0]["intended_stops"][0]["container_name"] == "ft-deployed-orphan"


def test_job_converges_when_desired_equals_running(
    isolated_registry, mock_deployment_manager, monkeypatch
):
    """Container is exactly the desired strategy → no intent on either side."""
    _seed_strategy(isolated_registry, name="Steady", sharpe=2.0)
    mock_deployment_manager.list_deployed.return_value = [{
        "name": "ft-deployed-steady",
        "status": "running",
        "id": "id1",
        "labels": {},
        "strategy_id": 1,
        "strategy_name": "Steady",
        "deployment_generation": 1,
    }]

    import orchestrator
    orchestrator.job_reconcile_deployments()

    rows = isolated_registry.get_recent_drift_logs(limit=1)
    assert rows[0]["intended_starts"] == []
    assert rows[0]["intended_stops"] == []


def test_job_records_reconciler_acting_flag_from_env(
    isolated_registry, mock_deployment_manager, monkeypatch
):
    """RECONCILER_ACTING is plumbed end-to-end. In Phase 2 it's NOT acted
    upon (no docker mutations either way), but the recorded value lets
    operators verify the flag is wired before flipping it in Phase 3."""
    monkeypatch.setenv("RECONCILER_ACTING", "true")
    import orchestrator
    orchestrator.job_reconcile_deployments()

    rows = isolated_registry.get_recent_drift_logs(limit=1)
    assert rows[0]["reconciler_acting"] == 1


def test_job_never_invokes_docker_mutation_methods(
    isolated_registry, mock_deployment_manager, monkeypatch
):
    """The hard guarantee for Phase 2: even with RECONCILER_ACTING=true
    and intended starts/stops, no real container action is taken."""
    monkeypatch.setenv("RECONCILER_ACTING", "true")
    _seed_strategy(isolated_registry, name="Winner", sharpe=2.0)
    mock_deployment_manager.list_deployed.return_value = [{
        "name": "ft-deployed-orphan",
        "status": "running",
        "id": "abc",
        "labels": {},
        "strategy_id": 99,
        "strategy_name": "OrphanStrategy",
        "deployment_generation": 1,
    }]

    import orchestrator
    orchestrator.job_reconcile_deployments()

    # CRITICAL: no mutation methods called, regardless of the intent
    mock_deployment_manager.start.assert_not_called()
    mock_deployment_manager.stop_graceful.assert_not_called()
    mock_deployment_manager.stop_hard.assert_not_called()
    mock_deployment_manager.remove.assert_not_called()


def test_job_swallows_docker_list_failure_and_still_records(
    isolated_registry, mock_deployment_manager, monkeypatch
):
    """If the SDK can't reach the docker daemon, the job logs the error
    and proceeds with running=[] so the drift log still captures the
    eligibility decision. Phase 4 drift alarm can later notice the gap."""
    _seed_strategy(isolated_registry, name="Winner", sharpe=2.0)
    mock_deployment_manager.list_deployed.side_effect = Exception("docker.sock denied")

    import orchestrator
    orchestrator.job_reconcile_deployments()

    rows = isolated_registry.get_recent_drift_logs(limit=1)
    assert rows[0]["running"] == []
    # Winner still wants to start (running view was empty)
    assert any(s["name"] == "Winner" for s in rows[0]["intended_starts"])
