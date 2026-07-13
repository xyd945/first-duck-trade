"""FreqAI candidate lifecycle tests (issue #47).

Covers:
  - registry migration adds spec_type (idempotent, defaults legacy rows
    to 'rule')
  - registration/promotion/retirement of freqai candidates alongside
    rule-based ones
  - freqai exclusion from hyperopt rescue and deployment eligibility
  - backtest_runner command construction for freqai runs
  - materialize/register writes the strategy + config + sidecar artifacts
  - the orchestrator's mandatory-walk-forward rule for freqai candidates
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

BASE = Path(__file__).parent.parent / "user_data"
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE))


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the registry module at a fresh tmp DB + reflections dir."""
    import strategy_registry as sr

    monkeypatch.setattr(sr, "DB_PATH", tmp_path / "test_registry.db")
    monkeypatch.setattr(sr, "REFLECTIONS_DIR", tmp_path / "reflections")
    sr.init_db()
    return sr


@pytest.fixture
def baseline_spec():
    return json.loads((BASE / "freqai_specs" / "baseline_lgbm.json").read_text())


# ---------------------------------------------------------------------------
# Registry: migration + registration
# ---------------------------------------------------------------------------

def test_migration_adds_spec_type(isolated_registry):
    sr = isolated_registry
    conn = sr.get_db()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(strategies)").fetchall()}
    conn.close()
    assert "spec_type" in cols
    sr.init_db()  # idempotent
    sr.init_db()


def test_legacy_rows_default_to_rule(isolated_registry):
    sr = isolated_registry
    sid = sr.register_strategy(name="OldRule", filepath="/tmp/OldRule.py")
    row = sr.get_strategy_by_name("OldRule")
    assert row["id"] == sid
    assert row["spec_type"] == "rule"


def test_register_freqai_candidate_sets_spec_type(isolated_registry):
    sr = isolated_registry
    sr.register_strategy(
        name="FreqaiThing", filepath="/tmp/FreqaiThing.py",
        archetype="ml_regressor", spec_type="freqai",
    )
    row = sr.get_strategy_by_name("FreqaiThing")
    assert row["spec_type"] == "freqai"
    assert row["status"] == "candidate"


def test_freqai_candidate_promotes_and_retires_like_rule(isolated_registry):
    sr = isolated_registry
    sid = sr.register_strategy(
        name="FreqaiPromote", filepath="/f.py", spec_type="freqai",
    )
    sr.promote_strategy(sid)
    assert sr.get_strategy_by_name("FreqaiPromote")["status"] == "active"

    sid2 = sr.register_strategy(
        name="FreqaiRetire", filepath="/f2.py", spec_type="freqai",
    )
    sr.retire_strategy(sid2, reason="wf unstable", verdict="FAIL_WF_UNSTABLE")
    row = sr.get_strategy_by_name("FreqaiRetire")
    assert row["status"] == "retired"
    assert row["failure_verdict"] == "FAIL_WF_UNSTABLE"


def test_rule_candidates_unaffected_by_freqai_presence(isolated_registry):
    """Acceptance criterion: freqai candidates must not break the
    rule-based path. Mixed pool: both register, both retrievable, order
    preserved."""
    sr = isolated_registry
    sr.register_strategy(name="RuleOne", filepath="/r1.py")
    sr.register_strategy(name="FreqaiOne", filepath="/q1.py", spec_type="freqai")
    sr.register_strategy(name="RuleTwo", filepath="/r2.py")

    cands = sr.get_candidates()
    assert [c["name"] for c in cands] == ["RuleOne", "FreqaiOne", "RuleTwo"]
    assert [c["spec_type"] for c in cands] == ["rule", "freqai", "rule"]


# ---------------------------------------------------------------------------
# Registry: exclusions
# ---------------------------------------------------------------------------

def test_hyperopt_rescue_excludes_freqai(isolated_registry):
    sr = isolated_registry
    rule_id = sr.register_strategy(name="RuleFail", filepath="/r.py")
    freqai_id = sr.register_strategy(
        name="FreqaiFail", filepath="/q.py", spec_type="freqai")
    for sid in (rule_id, freqai_id):
        sr.record_backtest(sid, {"total_trades": 12, "sharpe": -0.5})
        sr.retire_strategy(sid, reason="unprofitable", verdict="FAIL_UNPROFITABLE")

    names = [c["name"] for c in sr.get_hyperopt_candidates(limit=10)]
    assert "RuleFail" in names
    assert "FreqaiFail" not in names


def test_deployment_eligible_excludes_freqai(isolated_registry):
    sr = isolated_registry
    now = datetime.now(timezone.utc).isoformat()
    good_bt = {
        "total_trades": 40, "profit_total_pct": 12.0, "sharpe": 1.5,
        "max_drawdown_pct": 5.0, "backtest_data_end_at": now,
    }

    rule_id = sr.register_strategy(name="RuleWinner", filepath="/r.py")
    freqai_id = sr.register_strategy(
        name="FreqaiWinner", filepath="/q.py", spec_type="freqai")
    for sid in (rule_id, freqai_id):
        sr.record_backtest(sid, good_bt)
        sr.promote_strategy(sid)
        conn = sr.get_db()
        conn.execute(
            "UPDATE strategies SET research_status='approved' WHERE id=?", (sid,))
        conn.commit()
        conn.close()

    names = [s["name"] for s in sr.get_deployment_eligible()]
    assert "RuleWinner" in names
    assert "FreqaiWinner" not in names


# ---------------------------------------------------------------------------
# backtest_runner: freqai command construction
# ---------------------------------------------------------------------------

class _FakeProc:
    returncode = 0
    stdout = "no results"
    stderr = ""


def _capture_cmd(monkeypatch):
    import backtest_runner as br

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(br.subprocess, "run", fake_run)
    return br, captured


def test_run_backtest_freqai_uses_freqai_service_and_model(monkeypatch):
    br, captured = _capture_cmd(monkeypatch)
    br.run_backtest(
        "FreqaiLgbmBaseline",
        timerange="20260101-20260401",
        config_path="/freqtrade/user_data/configs/freqai/FreqaiLgbmBaseline.json",
        freqai_model="LightGBMRegressor",
    )
    cmd = captured["cmd"]
    assert "freqtrade-freqai" in cmd
    assert "freqtrade-backtest" not in cmd
    assert "--freqaimodel" in cmd
    assert cmd[cmd.index("--freqaimodel") + 1] == "LightGBMRegressor"
    assert "--timerange" in cmd
    # sandbox profile still applies (resource limits, on-demand service)
    assert "--profile" in cmd and "backtest" in cmd


def test_run_backtest_freqai_requires_timerange(monkeypatch):
    br, captured = _capture_cmd(monkeypatch)
    result = br.run_backtest("FreqaiLgbmBaseline", freqai_model="LightGBMRegressor")
    assert result["success"] is False
    assert "timerange" in result["error"]
    assert "cmd" not in captured  # never reached docker


def test_run_backtest_rule_path_unchanged(monkeypatch):
    br, captured = _capture_cmd(monkeypatch)
    br.run_backtest("SomeRuleStrategy")
    cmd = captured["cmd"]
    assert "freqtrade-backtest" in cmd
    assert "freqtrade-freqai" not in cmd
    assert "--freqaimodel" not in cmd


# ---------------------------------------------------------------------------
# Materialization + registration artifacts
# ---------------------------------------------------------------------------

def test_materialize_writes_all_artifacts(baseline_spec, tmp_path, monkeypatch):
    import freqai_spec as fs

    monkeypatch.setattr(fs, "CANDIDATES_DIR", tmp_path / "candidates")
    monkeypatch.setattr(fs, "FREQAI_CONFIG_DIR", tmp_path / "configs")

    artifacts = fs.materialize_freqai_candidate(baseline_spec)

    strategy = Path(artifacts["filepath"])
    config = Path(artifacts["config_path"])
    sidecar = Path(artifacts["sidecar_path"])
    assert strategy.exists() and config.exists() and sidecar.exists()

    rendered_cfg = json.loads(config.read_text())
    assert rendered_cfg["freqai"]["identifier"] == baseline_spec["name"]
    assert json.loads(sidecar.read_text())["model"]["family"] == "LightGBMRegressor"
    # Sidecar is discoverable from the strategy path alone (orchestrator's view)
    assert fs.load_spec_sidecar(strategy) == baseline_spec


def test_register_freqai_candidate_end_to_end(
    baseline_spec, tmp_path, monkeypatch, isolated_registry
):
    import freqai_spec as fs

    monkeypatch.setattr(fs, "CANDIDATES_DIR", tmp_path / "candidates")
    monkeypatch.setattr(fs, "FREQAI_CONFIG_DIR", tmp_path / "configs")

    sid = fs.register_freqai_candidate(baseline_spec)
    row = isolated_registry.get_strategy_by_name(baseline_spec["name"])
    assert row["id"] == sid
    assert row["spec_type"] == "freqai"
    assert row["archetype"] == "ml_regressor"
    assert row["status"] == "candidate"


def test_purge_model_artifacts(tmp_path, monkeypatch):
    import freqai_spec as fs

    monkeypatch.setattr(fs, "BASE_DIR", tmp_path)
    model_dir = tmp_path / "models" / "FreqaiX"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_text("x")

    assert fs.purge_model_artifacts("FreqaiX") is True
    assert not model_dir.exists()
    assert fs.purge_model_artifacts("FreqaiX") is False  # already gone


# ---------------------------------------------------------------------------
# Orchestrator: mandatory walk-forward for freqai
# ---------------------------------------------------------------------------

def _orchestrator_source() -> str:
    return (BASE / "scripts" / "orchestrator.py").read_text()


def test_orchestrator_forces_walk_forward_for_freqai():
    """Walk-forward must run for freqai candidates even when the
    R7_WALK_FORWARD env toggle is off."""
    src = _orchestrator_source()
    assert "if enable_wf or is_freqai:" in src


def test_orchestrator_blocks_freqai_promotion_without_strict_wf_pass():
    """A skipped walk-forward (passed=True, skipped=True) must never
    promote a freqai candidate, even in non-strict mode."""
    src = _orchestrator_source()
    assert "if is_freqai and not is_strict_pass(wf_verdict):" in src
    assert "FAIL_ML_NO_WALKFORWARD" in src


def test_orchestrator_purges_freqai_models_after_evaluation():
    src = _orchestrator_source()
    assert "purge_model_artifacts" in src


def test_run_walk_forward_retries_transient_crash():
    """A window that crashes once and succeeds on retry must not count as
    crashed (transient docker/network failures were turning into permanent
    FAIL_WF_CRASH verdicts)."""
    from pipeline_gates import gate_walk_forward, run_walk_forward

    calls = {}

    def flaky_fn(name, tr):
        calls[tr] = calls.get(tr, 0) + 1
        if calls[tr] == 1 and len(calls) == 1:  # first window, first attempt
            return {"success": False, "error": "transient"}
        return {"success": True, "sharpe": 0.5}

    results = run_walk_forward(
        "X", flaky_fn, n_splits=3, days_per_split=60, retry_delay_seconds=0,
    )
    assert all(r.get("success") for r in results)
    assert gate_walk_forward(results)["verdict"] != "FAIL_WF_CRASH"


def test_run_walk_forward_permanent_crash_still_fails():
    from pipeline_gates import gate_walk_forward, run_walk_forward

    def dead_fn(name, tr):
        return {"success": False, "error": "boom"}

    results = run_walk_forward(
        "X", dead_fn, n_splits=2, days_per_split=60, retry_delay_seconds=0,
    )
    assert gate_walk_forward(results)["verdict"] == "FAIL_WF_CRASH"


def test_gate_walk_forward_skip_is_not_strict_pass():
    """The property the mandatory-WF rule relies on: a WF skip fails
    is_strict_pass while a real pass clears it."""
    from pipeline_gates import _skip, gate_walk_forward, is_strict_pass

    skip = _skip("SKIP_WF", "walk-forward disabled")
    assert skip["passed"] is True
    assert not is_strict_pass(skip)

    real = gate_walk_forward([
        {"success": True, "sharpe": 1.2},
        {"success": True, "sharpe": 0.8},
        {"success": True, "sharpe": 0.4},
    ])
    assert is_strict_pass(real)

    unstable = gate_walk_forward([
        {"success": True, "sharpe": 4.0},
        {"success": True, "sharpe": -1.0},
        {"success": True, "sharpe": 0.1},
    ])
    assert not is_strict_pass(unstable)
    assert unstable["verdict"] == "FAIL_WF_UNSTABLE"
