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

@pytest.mark.parametrize("name,split", [("run_mini_backtest", "dev"), ("run_full_backtest", "selection")])
@pytest.mark.parametrize("kwargs", [{}, {"timeframe": "4h", "timerange": "20200102-20200103"}])
def test_wrappers_use_configured_splits_not_rolling_dates(name, split, kwargs):
    import backtest_runner
    with patch("backtest_runner.run_backtest", return_value={"success": True}) as run:
        getattr(backtest_runner, name)("S", **kwargs)
    run.assert_called_once_with("S", split=split, **kwargs)


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
