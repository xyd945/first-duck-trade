"""Operator recovery uses the real stopped service and append-only receipts."""
import json
import sqlite3
import subprocess
import sys

import pytest

from tests.test_h1_service import ROOT, offline_service


def operator_cli(action, directory, reason="Reconciled offline fixture"):
    return subprocess.run([
        sys.executable, str(ROOT / "user_data/scripts/h1_service.py"), action,
        "--directory", str(directory), "--ledger", str(directory / "ledger"),
        "--placeholder", "--reason", reason,
    ], text=True, capture_output=True, timeout=15)


def records(directory):
    with sqlite3.connect(directory / "journal.sqlite") as conn:
        return {(kind, key): json.loads(value) for kind, key, value in conn.execute("SELECT * FROM paper_records")}


def orders(directory):
    with sqlite3.connect(directory / "trades.sqlite") as conn:
        return conn.execute("SELECT * FROM orders ORDER BY id").fetchall()


@pytest.mark.parametrize("malformed", [False, True])
def test_stopped_reconcile_restores_ready_and_preserves_receipts(offline_service, tmp_path, monkeypatch, malformed):
    service, mode, children = offline_service
    mode[0] = "ready" if malformed else "resume-entry"
    cfg, ledger = ROOT / "user_data/configs/h1_dryrun.json", tmp_path / "ledger"
    if malformed:
        mode[0] = "tick-error"
        with monkeypatch.context() as patch:
            patch.setattr(service, "STARTUP_TIMEOUT", 1.5)
            with pytest.raises(RuntimeError, match="timed out"):
                service.start(cfg, tmp_path, ledger, placeholder=True)
        mode[0] = "ready"
        previous = {"asof": "2026-12-19T00:00:00+00:00", "kill_switch_active": True,
                    "allow_new_exposure": False, "reason": "Risk input rejected: 'day'"}
        assert not any(k == "batch" for k, _ in records(tmp_path))
    else:
        service.start(cfg, tmp_path, ledger, placeholder=True)
        service.stop(tmp_path)
        children[-1].wait(timeout=3)
        previous = records(tmp_path)[("risk", "current")]
        previous.update(kill_switch_active=True, allow_new_exposure=False, peak_equity=250500.)
    with sqlite3.connect(tmp_path / "journal.sqlite") as conn:
        conn.execute("INSERT OR REPLACE INTO paper_records VALUES ('risk','current',?)", (json.dumps(previous),))
    before = records(tmp_path)
    filled = orders(tmp_path)
    assert service.status(tmp_path)["risk"]["kill_switch_active"]
    assert bool(filled) is not malformed
    with pytest.raises(RuntimeError, match="halted"):
        service.start(cfg, tmp_path, ledger, placeholder=True)
    output = operator_cli("reconcile", tmp_path)
    assert output.returncode == 0, output.stderr
    receipt = json.loads(output.stdout)
    after = records(tmp_path)
    assert receipt["reason"] == "Reconciled offline fixture"
    assert receipt["previous"]["kill_switch_active"]
    assert receipt["new"]["capital"] == 250000.
    assert receipt["new"]["peak_equity"] == (250000. if malformed else 250500.)
    assert receipt["new"]["allow_new_exposure"]
    assert not receipt["new"]["kill_switch_active"]
    assert receipt in [v for (kind, _), v in after.items() if kind == "reconcile"]
    assert any(kind == "latch" for kind, _ in after)
    for key, value in before.items():
        if key[0] not in ("risk", "snapshot"):
            assert after[key] == value
    assert orders(tmp_path) == filled
    result = service.start(cfg, tmp_path, ledger, placeholder=True)
    assert result["ready"]
    service.stop(tmp_path)
    children[-1].wait(timeout=3)
    assert orders(tmp_path) == filled
    assert all(records(tmp_path)[k] == v for k, v in after.items() if k[0] not in ("risk", "snapshot"))


def test_code_identity_changes_require_stopped_audited_reidentify(offline_service, tmp_path):
    import research
    service, mode, children = offline_service
    cfg, ledger = ROOT / "user_data/configs/h1_dryrun.json", tmp_path / "ledger"
    service.start(cfg, tmp_path, ledger, placeholder=True)
    output = operator_cli("reidentify", tmp_path)
    assert output.returncode != 0
    assert not any(k == "reidentify" for k, _ in records(tmp_path))
    service.stop(tmp_path)
    children[-1].wait(timeout=3)
    path = tmp_path / "identity.json"
    original = json.loads(path.read_text())
    previous = {**original, "git_sha": "old-pinned-commit"}
    path.write_text(json.dumps(previous))
    before = records(tmp_path)
    with pytest.raises(RuntimeError, match="exited"):
        service.start(cfg, tmp_path, ledger, placeholder=True)
    output = operator_cli("reidentify", tmp_path, "Approved code upgrade")
    assert output.returncode == 0, output.stderr
    receipt = json.loads(output.stdout)
    assert receipt["previous"] == previous
    assert receipt["new"] == original
    assert receipt["new"]["git_sha"] == research.git_sha(ROOT)
    assert receipt["reason"] == "Approved code upgrade"
    assert json.loads(path.read_text()) == original
    after = records(tmp_path)
    assert any(k == "reidentify" and v == receipt for (k, _), v in after.items())
    assert all(after[k] == v for k, v in before.items())
    assert service.start(cfg, tmp_path, ledger, placeholder=True)["ready"]
    service.stop(tmp_path)
    children[-1].wait(timeout=3)
    # A second deliberate code change appends a new receipt rather than replacing the first.
    path.write_text(json.dumps({**original, "git_sha": "another-pinned-commit"}))
    output = operator_cli("reidentify", tmp_path, "Second approved upgrade")
    assert output.returncode == 0, output.stderr
    assert sum(k == "reidentify" for k, _ in records(tmp_path)) == 2
    assert all(records(tmp_path)[k] == v for k, v in after.items() if k[0] == "reidentify")


@pytest.mark.parametrize("action", ["reconcile", "reidentify"])
def test_operator_recovery_refuses_running_service_and_empty_reason(offline_service, tmp_path, action):
    service, mode, children = offline_service
    service.start(ROOT / "user_data/configs/h1_dryrun.json", tmp_path, tmp_path / "ledger", placeholder=True)
    before = records(tmp_path)
    output = operator_cli(action, tmp_path)
    assert output.returncode != 0
    assert "already running" in output.stderr
    assert records(tmp_path) == before
    assert service.status(tmp_path)["running"]
    service.stop(tmp_path)
    children[-1].wait(timeout=3)
    output = operator_cli(action, tmp_path, " ")
    assert output.returncode == 2
    assert "requires --reason" in output.stderr
    assert records(tmp_path) == before


@pytest.mark.parametrize("bad", ["rules", "params", "selection", "audit"])
def test_reidentify_does_not_bypass_frozen_contract_or_audit(offline_service, tmp_path, bad):
    service, mode, children = offline_service
    service.start(ROOT / "user_data/configs/h1_dryrun.json", tmp_path, tmp_path / "ledger", placeholder=True)
    service.stop(tmp_path)
    children[-1].wait(timeout=3)
    path = tmp_path / "identity.json"
    previous = {**json.loads(path.read_text()), "git_sha": "prior-code"}
    if bad != "audit":
        previous[bad]["unexpected"] = True
    else:
        with sqlite3.connect(tmp_path / "journal.sqlite") as conn:
            conn.execute("CREATE TRIGGER refuse_audit BEFORE INSERT ON paper_records WHEN NEW.kind='reidentify' BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    path.write_text(json.dumps(previous))
    before = records(tmp_path)
    output = operator_cli("reidentify", tmp_path)
    assert output.returncode != 0
    assert ("audit unavailable" if bad == "audit" else "cannot change") in output.stderr
    assert json.loads(path.read_text()) == previous
    assert records(tmp_path) == before


def test_restart_after_monitoring_gap_latches_until_explicit_reconcile(offline_service, tmp_path):
    service, mode, children = offline_service
    cfg, ledger = ROOT / "user_data/configs/h1_dryrun.json", tmp_path / "ledger"
    service.start(cfg, tmp_path, ledger, placeholder=True)
    service.stop(tmp_path)
    children[-1].wait(timeout=3)
    state = records(tmp_path)[("risk", "current")]
    state.update(asof="2026-12-18T23:00:00+00:00", day="2026-12-18")
    with sqlite3.connect(tmp_path / "journal.sqlite") as conn:
        conn.execute("UPDATE paper_records SET payload=? WHERE kind='risk' AND id='current'", (json.dumps(state),))
    with pytest.raises(RuntimeError, match="halted"):
        service.start(cfg, tmp_path, ledger, placeholder=True)
    assert service.status(tmp_path)["risk"]["kill_switch_active"]
    output = operator_cli("reconcile", tmp_path, "Recovered from outage")
    assert output.returncode == 0, output.stderr
    latch = records(tmp_path)[("latch", "2026-12-19T00:00:00+00:00")]
    assert service.start(cfg, tmp_path, ledger, placeholder=True)["ready"]
    service.stop(tmp_path)
    children[-1].wait(timeout=3)
    assert records(tmp_path)[("latch", "2026-12-19T00:00:00+00:00")] == latch
