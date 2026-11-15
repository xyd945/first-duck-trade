"""Risk wiring tests mock only filesystem locations, clock and REST/Docker edges."""
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "user_data/scripts"))
import orchestrator as orch
from tests.test_paper_risk import NOW, snapshot, state


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import deployment_manager
    monkeypatch.setattr(orch, "RISK_STATE_FILE", tmp_path / "risk.json")
    monkeypatch.setattr(orch, "RISK_SNAPSHOT_FILE", tmp_path / "account.json", raising=False)
    manager = Mock()
    manager.list_deployed.return_value = [{"name": "ft-deployed-dynamic", "status": "running"}]
    monkeypatch.setattr(deployment_manager, "DeploymentManager", Mock(return_value=manager))
    client = Mock()
    client.stop_entries.return_value = {"status": "paused"}
    client.get_profit.return_value = {"profit_all_coin": 0.}
    monkeypatch.setattr(orch, "FreqtradeClient", Mock(return_value=client))
    clock = Mock(wraps=orch.datetime)
    clock.now.return_value = NOW
    monkeypatch.setattr(orch, "datetime", clock)
    monkeypatch.setattr(orch, "notify_kill_switch", Mock())
    data = snapshot()
    data["accounts"][0]["instances"] = ["sweep", "momentum", "ft-deployed-dynamic"]
    data["accounts"][0]["positions"][1]["instance"] = "ft-deployed-dynamic"
    orch.RISK_SNAPSHOT_FILE.write_text(json.dumps(data))
    orch.RISK_STATE_FILE.write_text(json.dumps(state()))
    return client, manager


def test_job_halts_entries_across_legacy_and_dynamic_instances_without_liquidating(runtime):
    client, _ = runtime
    data = json.loads(orch.RISK_SNAPSHOT_FILE.read_text())
    data["accounts"][0]["cash"] -= 50.
    orch.RISK_SNAPSHOT_FILE.write_text(json.dumps(data))
    orch.job_check_risk()
    saved = json.loads(orch.RISK_STATE_FILE.read_text())
    assert saved.get("daily_halt"), "Daily loss is still not enforced"
    assert saved["equity"] == 9950.
    assert client.stop_entries.call_count == 3
    client.force_exit_all.assert_not_called()
    client.stop.assert_not_called()
    client.start.assert_not_called()
    client.get_profit.assert_not_called()


def test_missing_snapshot_retries_stops_even_after_kill_is_latched(runtime):
    client, _ = runtime
    orch.RISK_SNAPSHOT_FILE.unlink()
    orch.job_check_risk()
    assert json.loads(orch.RISK_STATE_FILE.read_text())["kill_switch_active"]
    orch.job_check_risk()
    assert client.stop_entries.call_count == 6
    client.force_exit_all.assert_not_called()


def test_discovery_failure_latches_kill_and_still_halts_known_instances(runtime):
    client, manager = runtime
    manager.list_deployed.side_effect = RuntimeError("fixture discovery failure")
    orch.job_check_risk()
    saved = json.loads(orch.RISK_STATE_FILE.read_text())
    assert saved["kill_switch_active"]
    assert not saved["allow_new_exposure"]
    assert client.stop_entries.call_count == 2


def test_disabled_execution_halts_entries_without_a_false_risk_alarm(runtime):
    client, _ = runtime
    result = orch.job_check_risk()
    assert result["allow_new_exposure"]
    assert client.stop_entries.call_count == 3
    orch.notify_kill_switch.assert_not_called()


def test_state_write_failure_still_halts_every_instance(runtime, monkeypatch):
    client, _ = runtime
    monkeypatch.setattr(orch, "save_risk_state", Mock(side_effect=OSError("fixture full disk")))
    result = orch.job_check_risk()
    assert client.stop_entries.call_count == 3
    assert result["kill_switch_active"] and not result["allow_new_exposure"]


def test_final_state_write_failure_is_not_an_allow_decision(runtime, monkeypatch):
    monkeypatch.setattr(orch, "save_risk_state", Mock(side_effect=[None, OSError("fixture full disk")]))
    result = orch.job_check_risk()
    assert result["kill_switch_active"] and not result["allow_new_exposure"]


def test_stop_failure_is_latched_and_does_not_skip_other_instances(runtime):
    client, _ = runtime
    client.stop_entries.side_effect = [None, {"status": "paused"}, {"status": "paused"}]
    result = orch.job_check_risk()
    assert client.stop_entries.call_count == 3
    assert result["kill_switch_active"] and not result["allow_new_exposure"]
    assert result["stop_failures"] == ["sweep"]
    assert json.loads(orch.RISK_STATE_FILE.read_text())["kill_switch_active"]


def test_missing_or_corrupt_risk_state_is_not_permission_to_start(tmp_path, monkeypatch):
    path = tmp_path / "risk.json"
    monkeypatch.setattr(orch, "RISK_STATE_FILE", path)
    assert orch.load_risk_state()["kill_switch_active"]
    path.write_text("{broken")
    assert orch.load_risk_state()["kill_switch_active"]


def test_regime_routing_cannot_restart_execution(runtime, monkeypatch):
    client, _ = runtime
    monkeypatch.setattr(orch, "load_regime_state", Mock(return_value={"regime": "ranging"}))
    orch.job_apply_regime()
    client.start.assert_not_called()
    client.force_exit_all.assert_not_called()


def test_rest_entry_start_and_forced_liquidation_are_disabled(monkeypatch):
    client = orch.FreqtradeClient("http://fixture.invalid", "fixture", "")
    request = Mock(return_value=[])
    monkeypatch.setattr(client, "_request", request)
    client.start()
    client.force_exit_all()
    request.assert_not_called()


def test_stop_entries_uses_pause_endpoint_not_stop_or_liquidation(monkeypatch):
    client = orch.FreqtradeClient("http://fixture.invalid", "fixture", "")
    request = Mock(return_value={"status": "paused"})
    monkeypatch.setattr(client, "_request", request)
    assert client.stop_entries() == {"status": "paused"}
    request.assert_called_once_with("POST", "stopentry")


def test_direct_deployment_start_stays_disabled_even_with_legacy_flags(monkeypatch):
    from deployment_manager import DeploymentManager, DeployedContainerSpec
    monkeypatch.setenv("RECONCILER_ACTING", "true")
    monkeypatch.setenv("RECONCILER_ALLOWLIST", "1")
    docker = Mock()
    docker.containers.list.return_value = []
    manager = DeploymentManager(docker_client=docker)
    with pytest.raises(RuntimeError, match="Execution is disabled"):
        manager.start(DeployedContainerSpec(1, "Fixture", 1), dry_run=False)
    docker.containers.run.assert_not_called()
