"""Tests for the mini-backtest overfit fix.

Trials #5 and #6 showed strategies passing mini (90 days ending today)
with sharpe 1+ and then blowing up in full (200 days). The mini was
covering the same recent slice as full, so a strategy overfit to that
slice scored well in both — until it touched the older 110 days of full.

Fix: mini now uses an OUT-OF-SAMPLE window ending 30 days before today,
so the recent month is reserved for full to evaluate. Acceptance is also
AND (profit AND sharpe), not OR, with the trade floor bumped to 10 —
trials showed OR-acceptance let near-zero-edge strategies through.

These tests pin both interventions against regression.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# run_mini_backtest — out-of-sample window
# ---------------------------------------------------------------------------

def _parse_timerange(tr: str) -> tuple[datetime, datetime]:
    """Parse a Freqtrade timerange string '20260101-20260401' into UTC dates."""
    start_s, end_s = tr.split("-")
    return (
        datetime.strptime(start_s, "%Y%m%d").replace(tzinfo=timezone.utc),
        datetime.strptime(end_s, "%Y%m%d").replace(tzinfo=timezone.utc),
    )


def test_mini_backtest_skips_recent_30_days_by_default():
    """The mini window must END 30 days before now, not today. This is
    what makes it out-of-sample relative to the full backtest."""
    from backtest_runner import run_mini_backtest

    captured = {}
    def fake_run_backtest(strategy_name, timerange=None, **kw):
        captured["timerange"] = timerange
        return {"success": True, "total_trades": 0}

    with patch("backtest_runner.run_backtest", side_effect=fake_run_backtest):
        run_mini_backtest("S")

    now = datetime.now(timezone.utc)
    _, end = _parse_timerange(captured["timerange"])
    gap_days = (now.date() - end.date()).days
    # End should be ~30 days before today (allow ±1 day for clock-edge)
    assert 29 <= gap_days <= 31, f"mini ends {gap_days}d before today; expected ~30"


def test_mini_backtest_window_length_is_90_days_by_default():
    """The mini window is 90 days long even after the skip, so freqtrade
    has enough room before/after the 200-candle startup prelude."""
    from backtest_runner import run_mini_backtest

    captured = {}
    def fake(strategy_name, timerange=None, **kw):
        captured["timerange"] = timerange
        return {"success": True, "total_trades": 0}

    with patch("backtest_runner.run_backtest", side_effect=fake):
        run_mini_backtest("S")

    start, end = _parse_timerange(captured["timerange"])
    length_days = (end - start).days
    assert length_days == 90


def test_mini_backtest_skip_recent_days_overridable():
    """An operator should be able to switch back to today-anchored
    evaluation by passing skip_recent_days=0."""
    from backtest_runner import run_mini_backtest

    captured = {}
    def fake(strategy_name, timerange=None, **kw):
        captured["timerange"] = timerange
        return {"success": True, "total_trades": 0}

    with patch("backtest_runner.run_backtest", side_effect=fake):
        run_mini_backtest("S", skip_recent_days=0)

    now = datetime.now(timezone.utc)
    _, end = _parse_timerange(captured["timerange"])
    assert (now.date() - end.date()).days <= 1


def test_mini_backtest_window_length_overridable():
    """Length is configurable for ops experimentation."""
    from backtest_runner import run_mini_backtest

    captured = {}
    def fake(strategy_name, timerange=None, **kw):
        captured["timerange"] = timerange
        return {"success": True, "total_trades": 0}

    with patch("backtest_runner.run_backtest", side_effect=fake):
        run_mini_backtest("S", days=60, skip_recent_days=15)

    start, end = _parse_timerange(captured["timerange"])
    assert (end - start).days == 60


# ---------------------------------------------------------------------------
# Acceptance criteria — AND not OR, min trades = 10
# ---------------------------------------------------------------------------

def _success_gen():
    return {
        "success": True,
        "filepath": Path("/tmp/MyStrategy.py"),
        "validation": MagicMock(passed=True, errors=[], warnings=[],
                                normalized_warnings=[], passed_after_normalization=True),
        "critic": {"verdict": "PASS", "summary": "ok", "issues": []},
        "generation_id": "gen-test-v0",
        "class_name": "MyStrategy",
    }


def test_acceptance_rejects_profit_zero_with_only_sharpe_positive():
    """Trial #5 cell 15: 43 trades, profit 0.04% (rounds to ~0), sharpe 0.02
    type strategies passed via the old `profit > 0 OR sharpe > 0` clause.
    New AND clause must reject when profit is non-positive."""
    from strategy_generator import generate_and_iterate

    bt = {"success": True, "total_trades": 50, "profit_total_pct": 0.0, "sharpe": 1.5}
    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=1,
            backtest_fn=lambda name: bt,
        )
    assert result["accepted"] is False, "profit=0 with positive sharpe must NOT accept"


def test_acceptance_rejects_sharpe_zero_with_only_profit_positive():
    """Symmetric: positive profit alone is not enough when risk-adjusted
    return is flat."""
    from strategy_generator import generate_and_iterate

    bt = {"success": True, "total_trades": 50, "profit_total_pct": 2.0, "sharpe": 0.0}
    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=1,
            backtest_fn=lambda name: bt,
        )
    assert result["accepted"] is False, "sharpe=0 with positive profit must NOT accept"


def test_acceptance_passes_when_profit_and_sharpe_both_positive():
    """Both positive is the genuine edge case we want — green-light."""
    from strategy_generator import generate_and_iterate

    bt = {"success": True, "total_trades": 30, "profit_total_pct": 1.5, "sharpe": 0.4}
    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=1,
            backtest_fn=lambda name: bt,
        )
    assert result["accepted"] is True


def test_acceptance_default_min_trades_is_ten():
    """Trial #5 cell 8 (17 trades) and similar would have been counted
    under accept_min_trades=5. We bumped to 10 to give a thin statistical
    cushion; verify the new default."""
    from strategy_generator import generate_and_iterate

    bt = {"success": True, "total_trades": 9, "profit_total_pct": 5.0, "sharpe": 2.0}
    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=1,
            backtest_fn=lambda name: bt,
        )
    assert result["accepted"] is False, "9 trades must fall under new default floor (10)"

    bt_ten = {**bt, "total_trades": 10}
    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=1,
            backtest_fn=lambda name: bt_ten,
        )
    assert result["accepted"] is True
