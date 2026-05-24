"""Tests for the Phase 3 acting path of the reconciler.

Phase 3 turns the reconciler from observe-only into a real container
manager — but ONLY for strategy ids explicitly opted in via the
RECONCILER_ALLOWLIST env var. Everything else stays observe-only even
when RECONCILER_ACTING=true.

This shakedown safety lets the operator deploy one strategy at a time,
watch it for ~24h, then add more — without a single env flag flip
spinning up everything in the eligible pool.

The tests below pin every layer of that safety:

  * empty allowlist + RECONCILER_ACTING=true → still observe-only
  * non-empty allowlist → only those ids actually get acted on
  * intents OUTSIDE the allowlist still get logged but not invoked
  * status transitions are written on every action (start/stop/fail)
  * the cooldown is set when a deployment fails
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# Allowlist parser
# ---------------------------------------------------------------------------

def test_allowlist_parser_empty_returns_empty_set():
    from orchestrator import _parse_allowlist
    assert _parse_allowlist("") == set()
    assert _parse_allowlist("   ") == set()


def test_allowlist_parser_handles_simple_csv():
    from orchestrator import _parse_allowlist
    assert _parse_allowlist("127,131,42") == {127, 131, 42}


def test_allowlist_parser_strips_whitespace():
    from orchestrator import _parse_allowlist
    assert _parse_allowlist(" 127 ,  131,42 ") == {127, 131, 42}


def test_allowlist_parser_drops_non_integer_tokens_silently():
    """A typo'd token must not crash the reconciler or partially-enable
    action on the WRONG ids."""
    from orchestrator import _parse_allowlist
    assert _parse_allowlist("127,foo,131") == {127, 131}


def test_allowlist_parser_drops_empty_tokens_from_trailing_commas():
    from orchestrator import _parse_allowlist
    assert _parse_allowlist("127,,131,") == {127, 131}


# ---------------------------------------------------------------------------
# Status transition helper
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    import strategy_registry as reg
    monkeypatch.setattr(reg, "DB_PATH", tmp_path / "test.db")
    reg.init_db()
    return reg


def test_mark_deployment_status_writes_status_and_clears_error(isolated_registry):
    reg = isolated_registry
    reg.register_strategy(name="X", filepath="/tmp/x.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("X")["id"]

    reg.mark_deployment_status(sid, "deploying")
    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT deployment_status, last_deployment_error, deployed_at "
        "FROM strategies WHERE id=?", (sid,)
    ).fetchone()
    conn.close()
    assert row[0] == "deploying"
    assert row[1] == ""
    assert row[2] is None  # only `deployed` sets deployed_at


def test_mark_deployment_status_deployed_writes_deployed_at(isolated_registry):
    reg = isolated_registry
    reg.register_strategy(name="X", filepath="/tmp/x.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("X")["id"]

    reg.mark_deployment_status(sid, "deployed")
    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT deployment_status, deployed_at FROM strategies WHERE id=?",
        (sid,),
    ).fetchone()
    conn.close()
    assert row[0] == "deployed"
    assert row[1] is not None and row[1] != ""


def test_mark_deployment_status_failed_writes_error_and_cooldown(isolated_registry):
    reg = isolated_registry
    reg.register_strategy(name="X", filepath="/tmp/x.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("X")["id"]

    reg.mark_deployment_status(sid, "failed",
                               error="docker.sock: permission denied",
                               block_for_hours=1.0)

    conn = sqlite3.connect(reg.DB_PATH)
    row = conn.execute(
        "SELECT deployment_status, last_deployment_error, deployment_blocked_until "
        "FROM strategies WHERE id=?", (sid,)
    ).fetchone()
    conn.close()
    assert row[0] == "failed"
    assert "permission denied" in row[1]
    # Cooldown is roughly now + 1h
    blocked = datetime.fromisoformat(row[2])
    expected = datetime.now(timezone.utc) + timedelta(hours=1)
    assert abs((blocked - expected).total_seconds()) < 60


def test_mark_deployment_status_rejects_unknown_state(isolated_registry):
    reg = isolated_registry
    reg.register_strategy(name="X", filepath="/tmp/x.py", thesis="t",
                          target_regime="ranging", generation_id="g")
    sid = reg.get_strategy_by_name("X")["id"]
    with pytest.raises(ValueError, match="must be one of"):
        reg.mark_deployment_status(sid, "in_a_weird_state")


# ---------------------------------------------------------------------------
# Reconciler — acting paths
# ---------------------------------------------------------------------------

def _seed_eligible(reg, *, name, sid_target=None, sharpe=2.0, path=""):
    """Seed an approved strategy that will pass the eligibility filter."""
    conn = sqlite3.connect(reg.DB_PATH)
    now = datetime.now(timezone.utc)
    conn.execute(
        "INSERT INTO strategies (name, filepath, research_status, "
        "deployment_status, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, f"/tmp/{name}.py", "approved", "not_deployed", now.isoformat()),
    )
    sid = conn.execute("SELECT id FROM strategies WHERE name=?", (name,)).fetchone()[0]
    conn.execute(
        "INSERT INTO backtest_results (strategy_id, sharpe, profit_total_pct, "
        "max_drawdown_pct, total_trades, backtest_data_end_at, trades_export_path, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, sharpe, 2.0, 5.0, 30, now.isoformat(), path, now.isoformat()),
    )
    conn.commit()
    conn.close()
    return sid


@pytest.fixture
def mock_manager():
    with patch("deployment_manager.DeploymentManager") as cls:
        instance = MagicMock()
        instance.list_deployed.return_value = []
        cls.return_value = instance
        yield instance


def test_reconciler_does_nothing_when_acting_false(isolated_registry, mock_manager,
                                                    monkeypatch):
    """Default state: even with intended starts, no Docker mutation
    happens. This is the Phase 2 contract preserved."""
    sid = _seed_eligible(isolated_registry, name="Winner")
    monkeypatch.delenv("RECONCILER_ACTING", raising=False)
    monkeypatch.setenv("RECONCILER_ALLOWLIST", str(sid))

    import orchestrator
    orchestrator.job_reconcile_deployments()

    mock_manager.start.assert_not_called()
    mock_manager.stop_graceful.assert_not_called()
    # State unchanged
    conn = sqlite3.connect(isolated_registry.DB_PATH)
    row = conn.execute(
        "SELECT deployment_status FROM strategies WHERE id=?", (sid,)
    ).fetchone()
    conn.close()
    assert row[0] == "not_deployed"


def test_reconciler_with_acting_true_empty_allowlist_is_safety_net(
    isolated_registry, mock_manager, monkeypatch
):
    """Codex's safety net: setting RECONCILER_ACTING=true ALONE — without
    explicit allowlist — must NOT touch any container. Operator has to
    opt each id in deliberately."""
    sid = _seed_eligible(isolated_registry, name="Winner")
    monkeypatch.setenv("RECONCILER_ACTING", "true")
    monkeypatch.delenv("RECONCILER_ALLOWLIST", raising=False)

    import orchestrator
    orchestrator.job_reconcile_deployments()

    mock_manager.start.assert_not_called()
    mock_manager.stop_graceful.assert_not_called()


def test_reconciler_starts_only_allowlisted_intent(isolated_registry, mock_manager,
                                                    monkeypatch):
    """The shakedown success case: two eligible strategies, allowlist has
    only one id. The reconciler starts ONE container, leaves the other
    as a logged-but-unactioned intent."""
    sid_a = _seed_eligible(isolated_registry, name="Allowed", sharpe=3.0)
    sid_b = _seed_eligible(isolated_registry, name="NotAllowed", sharpe=2.5)
    monkeypatch.setenv("RECONCILER_ACTING", "true")
    monkeypatch.setenv("RECONCILER_ALLOWLIST", str(sid_a))

    import orchestrator
    orchestrator.job_reconcile_deployments()

    # Exactly one start call, and it was for the allowed id
    assert mock_manager.start.call_count == 1
    spec_arg = mock_manager.start.call_args.args[0]
    assert spec_arg.strategy_id == sid_a
    assert spec_arg.strategy_name == "Allowed"
    # dry_run kwarg was False (real action)
    assert mock_manager.start.call_args.kwargs.get("dry_run") is False

    # NotAllowed was logged in intended_starts but no start() call for it
    conn = sqlite3.connect(isolated_registry.DB_PATH)
    rows = dict(conn.execute(
        "SELECT name, deployment_status FROM strategies WHERE id IN (?, ?)",
        (sid_a, sid_b),
    ).fetchall())
    conn.close()
    assert rows["Allowed"] == "deployed"
    assert rows["NotAllowed"] == "not_deployed"  # observed only


def test_reconciler_records_failed_status_when_start_raises(
    isolated_registry, mock_manager, monkeypatch
):
    """If the Docker SDK throws during start, the strategy must end up
    deployment_status='failed' with the error AND a cooldown so the
    next tick doesn't immediately retry."""
    sid = _seed_eligible(isolated_registry, name="WillFail")
    monkeypatch.setenv("RECONCILER_ACTING", "true")
    monkeypatch.setenv("RECONCILER_ALLOWLIST", str(sid))
    mock_manager.start.side_effect = RuntimeError("docker daemon unreachable")

    import orchestrator
    orchestrator.job_reconcile_deployments()

    conn = sqlite3.connect(isolated_registry.DB_PATH)
    row = conn.execute(
        "SELECT deployment_status, last_deployment_error, deployment_blocked_until "
        "FROM strategies WHERE id=?", (sid,)
    ).fetchone()
    conn.close()
    assert row[0] == "failed"
    assert "RuntimeError" in row[1]
    assert "docker daemon unreachable" in row[1]
    assert row[2] is not None  # cooldown set


def test_reconciler_stops_only_allowlisted_intent(isolated_registry, mock_manager,
                                                   monkeypatch):
    """An orphaned running container whose strategy_id IS in the allowlist
    gets stopped; one NOT in the allowlist stays running (operator must
    opt in explicitly even for stops)."""
    mock_manager.list_deployed.return_value = [
        {"name": "ft-deployed-old-allowed", "status": "running", "id": "a",
         "labels": {}, "strategy_id": 555, "strategy_name": "OldAllowed",
         "deployment_generation": 1},
        {"name": "ft-deployed-orphan", "status": "running", "id": "b",
         "labels": {}, "strategy_id": 999, "strategy_name": "OrphanOutsideAllowlist",
         "deployment_generation": 1},
    ]
    monkeypatch.setenv("RECONCILER_ACTING", "true")
    monkeypatch.setenv("RECONCILER_ALLOWLIST", "555")

    import orchestrator
    orchestrator.job_reconcile_deployments()

    # Both were intended to stop (neither in desired); only the allowlisted
    # one actually got stopped.
    assert mock_manager.stop_graceful.call_count == 1
    assert mock_manager.stop_graceful.call_args.args[0] == "OldAllowed"
    assert mock_manager.stop_graceful.call_args.kwargs.get("dry_run") is False


# ---------------------------------------------------------------------------
# _build_deployed_env
# ---------------------------------------------------------------------------

def test_build_deployed_env_propagates_okx_and_ft_vars(monkeypatch):
    """The render_config.py inside the deployed container needs these
    exact var names. If a future edit renames or drops one, the deployed
    container will fail to start. Pin the contract."""
    from orchestrator import _build_deployed_env
    monkeypatch.setenv("OKX_API_KEY", "k")
    monkeypatch.setenv("OKX_API_SECRET", "s")
    monkeypatch.setenv("OKX_API_PASSPHRASE", "p")
    monkeypatch.setenv("FT_DEPLOYED_JWT_SECRET", "j")
    monkeypatch.setenv("FT_DEPLOYED_API_PASSWORD", "r")

    env = _build_deployed_env("FundingFoo", "funding-foo")
    assert env["OKX_API_KEY"] == "k"
    assert env["OKX_API_SECRET"] == "s"
    assert env["OKX_API_PASSPHRASE"] == "p"
    assert env["FT_DEPLOYED_JWT_SECRET"] == "j"
    assert env["FT_DEPLOYED_API_PASSWORD"] == "r"
    assert env["STRATEGY_NAME"] == "FundingFoo"
    assert env["STRATEGY_SLUG"] == "funding-foo"


def test_build_deployed_env_returns_empty_string_for_unset_vars(monkeypatch):
    """Missing env propagates as empty — the render script will then
    fail loudly inside the deployed container (PR #38's contract). We
    don't want the orchestrator to silently substitute defaults."""
    from orchestrator import _build_deployed_env
    for v in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE",
              "FT_DEPLOYED_JWT_SECRET", "FT_DEPLOYED_API_PASSWORD"):
        monkeypatch.delenv(v, raising=False)
    env = _build_deployed_env("X", "x")
    assert env["OKX_API_KEY"] == ""
    assert env["FT_DEPLOYED_API_PASSWORD"] == ""


def test_build_deployed_env_sets_pythonpath_to_user_data():
    """Regression from the Phase 3 shakedown: without PYTHONPATH the
    deployed container's freqtrade can't import the generated strategy
    (which does `from indicators.external_data import ...`) and dies
    with "No module named 'indicators'". ft-momentum / ft-sweep also
    set this in their docker-compose blocks for the same reason."""
    from orchestrator import _build_deployed_env
    env = _build_deployed_env("X", "x")
    assert env.get("PYTHONPATH") == "/freqtrade/user_data"
