"""Pin the orchestrator's INSTANCES dict to env-driven REST API credentials.

Codex review flagged the hardcoded passwords. PR moved them to
FT_*_API_PASSWORD env vars with the prior literal as fallback so dev
still works without wiring secrets. These tests guard against a
future edit that quietly removes either the env-read or the fallback.
"""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


def _reload_orchestrator(monkeypatch):
    """Re-import orchestrator so module-level INSTANCES re-reads os.environ
    against whatever monkeypatch has set. Without this, the import-time
    snapshot would lock in the env values from session start."""
    if "orchestrator" in sys.modules:
        del sys.modules["orchestrator"]
    return importlib.import_module("orchestrator")


def test_instances_use_env_password_when_set(monkeypatch):
    monkeypatch.setenv("FT_MOMENTUM_API_PASSWORD", "live-momentum-pw")
    monkeypatch.setenv("FT_SWEEP_API_PASSWORD",    "live-sweep-pw")
    orch = _reload_orchestrator(monkeypatch)
    assert orch.INSTANCES["momentum"]["password"] == "live-momentum-pw"
    assert orch.INSTANCES["sweep"]["password"] == "live-sweep-pw"


def test_instances_fall_back_to_legacy_literal_when_env_unset(monkeypatch):
    """Dev / test machines without secrets should still produce a working
    orchestrator import. The fallback matches the literal the freqtrade
    template defaults imply, so behavior is preserved."""
    monkeypatch.delenv("FT_MOMENTUM_API_PASSWORD", raising=False)
    monkeypatch.delenv("FT_SWEEP_API_PASSWORD", raising=False)
    orch = _reload_orchestrator(monkeypatch)
    assert orch.INSTANCES["momentum"]["password"] == "CHANGE_ME_momentum_password"
    assert orch.INSTANCES["sweep"]["password"] == "CHANGE_ME_sweep_password"


def test_instances_username_overridable_via_env(monkeypatch):
    """Less-common knob but in the same module — confirm symmetric handling."""
    monkeypatch.setenv("FT_MOMENTUM_API_USERNAME", "ops-bot")
    orch = _reload_orchestrator(monkeypatch)
    assert orch.INSTANCES["momentum"]["username"] == "ops-bot"


def test_instances_no_hardcoded_password_in_source():
    """Source-level safety net: the orchestrator module's text must not
    contain the legacy passwords as the SOLE source — they should only
    appear inside an os.environ.get(..., DEFAULT) call. If someone later
    drops the env-read wrapper and reintroduces a bare assignment, this
    test fires."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    # The literal must still appear (it's the fallback), but only inside
    # an env-lookup default expression. The "_instance_password" call shape
    # OR a direct os.environ.get with the literal as default counts.
    for literal in ("CHANGE_ME_momentum_password", "CHANGE_ME_sweep_password"):
        # Find every line containing the literal
        hits = [ln for ln in src.splitlines() if literal in ln]
        assert hits, f"expected {literal!r} as a fallback default in orchestrator.py"
        for ln in hits:
            # Allow it only when it's an argument to the helper or to os.environ.get
            stripped = ln.strip()
            allowed = (
                "_instance_password(" in stripped
                or "os.environ.get(" in stripped
            )
            assert allowed, (
                f"{literal!r} appears in orchestrator.py outside an env-lookup "
                f"default ({stripped!r}); env-driven loading must be preserved"
            )
