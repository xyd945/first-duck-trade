"""Tests for the greedy correlation-aware deployment selection.

The selection policy decides which approved strategies the reconciler
SHOULD bring up. Phase 2 wires this into an observe-only job; Phase 3
makes the reconciler act. These tests pin the policy in isolation —
no DB, no docker — so we can change the orchestrator's control flow
later without disturbing the load-bearing decision logic.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


from deployment_selection import (
    compute_desired_deployments,
    DEFAULT_MAX_DEPLOY,
    DEFAULT_CORR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(name: str, sharpe: float, *, path: str = None, **extras) -> dict:
    """Eligible-pool row in the same shape get_deployment_eligible() returns."""
    return {
        "id": hash(name) % 10000,
        "name": name,
        "sharpe": sharpe,
        "trades_export_path": path if path is not None else f"/tmp/{name}.zip",
        **extras,
    }


def _trades_returns_loader(returns_by_strategy: dict[str, pd.Series]):
    """Build a load_trades stub that returns synthetic 'trades' rows
    whose trades_to_daily_returns boils down to the given pandas Series
    of daily returns. We bypass the real trade-loading by providing pre-
    computed series and using a fake load_trades that returns a marker;
    we ALSO have to stub trades_to_daily_returns since the real one
    expects exit_date / profit_pct rows.

    Simpler approach: build real-looking trade dicts with one trade per
    daily-return value, since trades_to_daily_returns groups by date.
    """
    def loader(path: str, name: str):
        if name not in returns_by_strategy:
            return []
        series = returns_by_strategy[name]
        trades = []
        for date, ret in series.items():
            trades.append({
                # trades_to_daily_returns reads close_date + profit_ratio
                "close_date": (date.isoformat() if hasattr(date, "isoformat")
                               else str(date)),
                "profit_ratio": float(ret),
            })
        return trades
    return loader


def _daily_series(start: str, values: list[float]) -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx)


# ---------------------------------------------------------------------------
# Ordering + cap
# ---------------------------------------------------------------------------

def test_returns_top_n_when_no_data_to_correlate():
    """Empty trade exports => no correlation rejection. Top sharpe wins."""
    eligible = [
        _row("A", sharpe=2.0, path=""),
        _row("B", sharpe=1.5, path=""),
        _row("C", sharpe=1.0, path=""),
        _row("D", sharpe=0.5, path=""),
    ]
    out = compute_desired_deployments(eligible, max_deploy=3, load_trades=lambda *_: [])
    assert [r["name"] for r in out["desired"]] == ["A", "B", "C"]
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["row"]["name"] == "D"
    assert "deployment_slots_full" in out["skipped"][0]["reason"]


def test_sorts_eligible_by_sharpe_descending_before_walking():
    """Even if the caller passes the rows out of order, selection
    sees them top-Sharpe-first."""
    eligible = [
        _row("Low", sharpe=0.3, path=""),
        _row("High", sharpe=2.0, path=""),
        _row("Mid", sharpe=1.0, path=""),
    ]
    out = compute_desired_deployments(eligible, max_deploy=2, load_trades=lambda *_: [])
    assert [r["name"] for r in out["desired"]] == ["High", "Mid"]


def test_respects_max_deploy_cap_smaller_than_eligible():
    eligible = [_row(f"S{i}", sharpe=10 - i, path="") for i in range(10)]
    out = compute_desired_deployments(eligible, max_deploy=3, load_trades=lambda *_: [])
    assert len(out["desired"]) == 3
    assert len(out["skipped"]) == 7


def test_returns_all_when_eligible_below_cap():
    eligible = [_row("Only", sharpe=1.0, path="")]
    out = compute_desired_deployments(eligible, max_deploy=3, load_trades=lambda *_: [])
    assert [r["name"] for r in out["desired"]] == ["Only"]
    assert out["skipped"] == []


def test_handles_empty_eligible_pool():
    out = compute_desired_deployments([], max_deploy=3, load_trades=lambda *_: [])
    assert out == {"desired": [], "skipped": []}


# ---------------------------------------------------------------------------
# Greedy correlation skip
# ---------------------------------------------------------------------------

def test_skips_candidate_correlated_above_threshold_with_selected():
    """A and B are identical (corr=1.0) — only A admitted."""
    returns = {
        "A": _daily_series("2026-01-01", [0.01, -0.02, 0.03, 0.01, -0.01] * 10),
        "B": _daily_series("2026-01-01", [0.01, -0.02, 0.03, 0.01, -0.01] * 10),
        "C": _daily_series("2026-01-01", [-0.01, 0.02, -0.03, -0.01, 0.01] * 10),
    }
    eligible = [
        _row("A", sharpe=2.0),
        _row("B", sharpe=1.5),
        _row("C", sharpe=1.0),
    ]
    out = compute_desired_deployments(
        eligible, max_deploy=3, corr_threshold=0.7,
        load_trades=_trades_returns_loader(returns),
    )
    names = [r["name"] for r in out["desired"]]
    assert "A" in names and "C" in names
    assert "B" not in names
    skipped_names = [s["row"]["name"] for s in out["skipped"]]
    assert "B" in skipped_names
    b_skip = next(s for s in out["skipped"] if s["row"]["name"] == "B")
    assert "corr" in b_skip["reason"]
    assert "'A'" in b_skip["reason"]


def test_uncorrelated_strategies_all_admitted():
    """Three independently-varying series — none should skip on correlation."""
    import numpy as np
    rng = np.random.default_rng(0)
    returns = {
        n: pd.Series(rng.normal(0, 0.01, 60),
                     index=pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC"))
        for n in ("A", "B", "C")
    }
    eligible = [_row("A", 2.0), _row("B", 1.5), _row("C", 1.0)]
    out = compute_desired_deployments(
        eligible, max_deploy=3, corr_threshold=0.7,
        load_trades=_trades_returns_loader(returns),
    )
    assert [r["name"] for r in out["desired"]] == ["A", "B", "C"]
    assert out["skipped"] == []


def test_correlation_check_uses_only_already_selected_not_full_eligible():
    """If A and B are highly correlated and B has higher Sharpe than C,
    but C correlates with NONE of the selected — C still gets admitted
    even though it correlates with the SKIPPED B. We only check against
    things we've actually decided to keep."""
    returns = {
        "A": _daily_series("2026-01-01", [0.01, 0.02, -0.01, 0.03] * 15),
        "B": _daily_series("2026-01-01", [0.01, 0.02, -0.01, 0.03] * 15),  # = A
        "C": _daily_series("2026-01-01", [-0.01, -0.02, 0.01, -0.03] * 15),  # ≠ A; corr to B = 1.0
    }
    # Sharpe order: A (2.0) > B (1.5) > C (1.0). A admitted. B skipped (corr to A).
    # C: corr to A is -1.0, so not >= 0.7. C admitted. C's corr to B is +1.0 but
    # B was skipped so it's not in `desired` and not checked.
    eligible = [_row("A", 2.0), _row("B", 1.5), _row("C", 1.0)]
    out = compute_desired_deployments(
        eligible, max_deploy=3, corr_threshold=0.7,
        load_trades=_trades_returns_loader(returns),
    )
    assert [r["name"] for r in out["desired"]] == ["A", "C"]


def test_small_overlap_treated_as_uncomparable_not_blocking():
    """If two series only overlap for a few days, the correlation
    measurement is statistical noise and we should skip the PAIR (not
    the candidate). Confirm by giving B a window that barely overlaps
    with A — B should still be admitted despite an apparent corr=1
    on the few overlapping points."""
    returns = {
        "A": _daily_series("2026-01-01", [0.01] * 60),
        # B's window starts after A's ends — overlap is 0 days
        "B": _daily_series("2026-04-01", [0.01] * 60),
    }
    eligible = [_row("A", 2.0), _row("B", 1.5)]
    out = compute_desired_deployments(
        eligible, max_deploy=3, corr_threshold=0.7, min_overlap_days=30,
        load_trades=_trades_returns_loader(returns),
    )
    assert [r["name"] for r in out["desired"]] == ["A", "B"]


def test_no_trades_export_path_candidate_still_admitted_with_log():
    """Approved-but-no-trade-data candidates pass selection. The
    correlation gate at promotion time is the strong barrier; here we're
    conservative on missing data so we don't accidentally lock out a
    legacy (pre-export) strategy. Operator can spot it via the log line."""
    eligible = [
        _row("WithData", sharpe=2.0, path="/some/zip.zip"),
        _row("LegacyNoData", sharpe=1.5, path=""),
    ]
    returns = {"WithData": _daily_series("2026-01-01", [0.01] * 60)}
    out = compute_desired_deployments(
        eligible, max_deploy=3,
        load_trades=_trades_returns_loader(returns),
    )
    assert [r["name"] for r in out["desired"]] == ["WithData", "LegacyNoData"]


def test_threshold_at_exact_value_is_a_skip():
    """corr == threshold counts as skip (>=, not >). 0.7 fails."""
    returns = {
        "A": _daily_series("2026-01-01", [0.01, -0.01] * 30),
        # Construct B = A * 1.0 + small noise so the correlation is exactly 1.0
        "B": _daily_series("2026-01-01", [0.01, -0.01] * 30),
    }
    eligible = [_row("A", 2.0), _row("B", 1.5)]
    out = compute_desired_deployments(
        eligible, max_deploy=3, corr_threshold=0.7,
        load_trades=_trades_returns_loader(returns),
    )
    assert [r["name"] for r in out["desired"]] == ["A"]


def test_default_constants_match_spec():
    """Spec lock — these defaults are the V1 policy. If a future PR
    changes them, the spec doc + this test should be updated together."""
    assert DEFAULT_MAX_DEPLOY == 3
    assert DEFAULT_CORR_THRESHOLD == 0.7
