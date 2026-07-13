"""Tests for the fail-closed pipeline-gate aggregation.

Codex flagged that several gates returned PASS verdicts when their input
data was missing — so a candidate could promote on "no evidence" rather
than "evidence of pass". This PR introduces:

  * is_strict_pass(verdict)  — the single source-of-truth predicate for
                               "this verdict really cleared the gate, not
                               just shrugged because data was unavailable".
  * gate_beat_buyhold        — rejects the degenerate 0%/0% case explicitly
                               instead of returning PASS_BH_PROFIT.
  * orchestrator             — wraps gate aggregation with strict_mode
                               (default true via STRICT_PROMOTION_GATES env),
                               always appends a verdict per expected gate.

These tests pin every step so a future edit can't quietly flip back to
lenient aggregation.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# is_strict_pass — the predicate
# ---------------------------------------------------------------------------

def test_strict_pass_accepts_real_pass():
    from pipeline_gates import is_strict_pass
    assert is_strict_pass({"passed": True, "verdict": "PASS_REGIME",
                           "reason": "ok", "details": {}}) is True


def test_strict_pass_rejects_skipped_verdict():
    """A _skip()-shaped verdict carries passed=True for legacy aggregation
    but skipped=True. Strict callers must read it as fail."""
    from pipeline_gates import _skip, is_strict_pass
    v = _skip("SKIP_X", "no data")
    assert v["passed"] is True
    assert v["skipped"] is True
    assert is_strict_pass(v) is False


def test_strict_pass_rejects_real_fail():
    from pipeline_gates import _fail, is_strict_pass
    assert is_strict_pass(_fail("FAIL_X", "nope")) is False


def test_strict_pass_handles_missing_keys_safely():
    """Defensive: a malformed verdict shouldn't crash the predicate.
    Both keys default to "absent" → False → safe (fail-closed)."""
    from pipeline_gates import is_strict_pass
    assert is_strict_pass({}) is False
    assert is_strict_pass({"verdict": "MYSTERY"}) is False


# ---------------------------------------------------------------------------
# gate_beat_buyhold — degenerate 0% / 0% must now skip, not pass
# ---------------------------------------------------------------------------

def test_buyhold_degenerate_zero_strategy_zero_hodl_is_skip_not_pass():
    """The case Codex spotted: log line
       'PASS_BH_PROFIT — strategy 0.00% clears 70%-of-HODL floor (0.00%)'
       made a no-trade strategy look like it cleared the gate. New behavior
       is an explicit skip so strict mode rejects it."""
    from pipeline_gates import gate_beat_buyhold, is_strict_pass
    bt = {"profit_total_pct": 0.0, "max_drawdown_pct": 0.0}
    bh = {"profit_pct": 0.0, "max_drawdown_pct": 0.0}
    v = gate_beat_buyhold(bt, bh)
    assert v["verdict"] == "PASS_BH_DEGENERATE"
    assert v.get("skipped") is True
    assert is_strict_pass(v) is False


def test_buyhold_legitimate_pass_still_passes_strict():
    """Regression: a real beat-HODL must still strict-pass."""
    from pipeline_gates import gate_beat_buyhold, is_strict_pass
    bt = {"profit_total_pct": 5.0, "max_drawdown_pct": 3.0}
    bh = {"profit_pct": 2.0, "max_drawdown_pct": 8.0}
    v = gate_beat_buyhold(bt, bh)
    assert v["verdict"] == "PASS_BH_PROFIT"
    assert is_strict_pass(v) is True


def test_buyhold_safer_drawdown_still_passes_strict():
    """The other PASS path — meaningfully lower drawdown — also strict-passes."""
    from pipeline_gates import gate_beat_buyhold, is_strict_pass
    bt = {"profit_total_pct": 1.0, "max_drawdown_pct": 1.0}
    bh = {"profit_pct": 5.0, "max_drawdown_pct": 10.0}  # 9pp DD advantage
    v = gate_beat_buyhold(bt, bh)
    assert v["verdict"] == "PASS_BH_SAFER"
    assert is_strict_pass(v) is True


def test_buyhold_real_fail_still_fails():
    """A genuine fail (loses on both profit AND DD) must still fail strict."""
    from pipeline_gates import gate_beat_buyhold, is_strict_pass
    bt = {"profit_total_pct": -2.0, "max_drawdown_pct": 8.0}
    bh = {"profit_pct": 5.0, "max_drawdown_pct": 10.0}
    v = gate_beat_buyhold(bt, bh)
    assert v["verdict"] == "FAIL_BH"
    assert is_strict_pass(v) is False


def test_buyhold_skip_on_explicit_error_unchanged():
    """The bh-data-unavailable path still _skip()s with PASS_BH_NA and is
    rejected by strict mode."""
    from pipeline_gates import gate_beat_buyhold, is_strict_pass
    v = gate_beat_buyhold({"profit_total_pct": 5.0, "max_drawdown_pct": 1.0},
                          {"error": "no btc data file"})
    assert v["verdict"] == "PASS_BH_NA"
    assert is_strict_pass(v) is False


# ---------------------------------------------------------------------------
# Orchestrator: source-level guardrails for strict-mode behavior
# ---------------------------------------------------------------------------

def test_orchestrator_uses_strict_mode_aggregation():
    """Confirm the orchestrator reads STRICT_PROMOTION_GATES and uses
    is_strict_pass in the aggregation. If a future edit drops either
    half, gates can silently fail open again."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    assert "STRICT_PROMOTION_GATES" in src
    assert "is_strict_pass(v) for v in gate_verdicts" in src


def test_orchestrator_appends_skip_verdict_for_missing_regime_data():
    """When regime_fractions is None the gate must still append a verdict
    (not silently omit). Source-level check that the else: branch exists."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    # The pattern we're guarding: "if regime_fractions is not None: <run>
    # else: <append _skip>". Both halves must be present.
    assert "if regime_fractions is not None:" in src
    assert 'SKIP_REGIME' in src


def test_orchestrator_appends_skip_verdict_for_missing_btc_path():
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    assert "if btc_path:" in src
    assert 'SKIP_BH' in src


def test_orchestrator_appends_skip_verdict_when_walk_forward_disabled():
    """Walk-forward is opt-in via env. Old code: when disabled, no verdict
    at all → silent pass. New: explicit SKIP_WF appended so strict mode
    catches it (or operator must explicitly run WF for promotion)."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    # Issue #47 widened the guard: walk-forward also force-runs for freqai
    # candidates. The SKIP_WF fallback for disabled-WF rule candidates stays.
    assert "if enable_wf or is_freqai:" in src
    assert 'SKIP_WF' in src


def test_orchestrator_correlation_exception_becomes_explicit_fail():
    """Codex spotted that the correlation gate's exception handler used to
    silently log and continue (= implicit pass). Now it appends a
    FAIL_CORR_ERROR verdict so promotion can't ride on a swallowed error."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    assert "FAIL_CORR_ERROR" in src


def test_orchestrator_strict_mode_defaults_to_true():
    """The env-var default must be true — operators have to opt INTO the
    legacy lenient behavior, not opt out of strict."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    # Find the env read and confirm it parses to True by default
    import re
    m = re.search(
        r'os\.environ\.get\(\s*[\'"]STRICT_PROMOTION_GATES[\'"]\s*,\s*[\'"]([^\'"]+)[\'"]',
        src,
    )
    assert m, "STRICT_PROMOTION_GATES env read not found"
    default_str = m.group(1).lower()
    # The orchestrator should NOT treat the default as false/0/no
    assert default_str not in ("0", "false", "no", ""), (
        f"STRICT_PROMOTION_GATES default is {default_str!r} — must default "
        f"to a truthy value so production fails closed by default"
    )
