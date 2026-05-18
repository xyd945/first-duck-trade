"""Tests for R6: multi-turn generate → backtest → refine loop."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# _summarize_backtest_for_llm — diagnostic branches
# ---------------------------------------------------------------------------

def test_diagnostic_zero_trades():
    from strategy_generator import _summarize_backtest_for_llm
    out = _summarize_backtest_for_llm({"success": True, "total_trades": 0})
    assert "ZERO TRADES" in out
    assert "macro_confidence" in out  # nudges the LLM toward the right fix


def test_diagnostic_too_few_trades():
    from strategy_generator import _summarize_backtest_for_llm
    out = _summarize_backtest_for_llm(
        {"success": True, "total_trades": 3, "profit_total_pct": 0.5, "sharpe": 0.1}
    )
    assert "only 3 trades" in out
    assert "Loosen" in out or "loosen" in out


def test_diagnostic_unprofitable():
    from strategy_generator import _summarize_backtest_for_llm
    out = _summarize_backtest_for_llm(
        {"success": True, "total_trades": 50, "profit_total_pct": -1.5,
         "sharpe": -0.3, "max_drawdown_pct": 8.0}
    )
    assert "lost money" in out
    assert "-1.50%" in out or "-1.5" in out  # accept either rendering
    assert "thesis" in out.lower()


def test_diagnostic_passable():
    from strategy_generator import _summarize_backtest_for_llm
    out = _summarize_backtest_for_llm(
        {"success": True, "total_trades": 20, "profit_total_pct": 0.8, "sharpe": 0.5}
    )
    assert "passable" in out
    assert "20 trades" in out


def test_diagnostic_backtest_failed():
    from strategy_generator import _summarize_backtest_for_llm
    out = _summarize_backtest_for_llm({"success": False, "error": "import failed"})
    assert "FAILED to backtest" in out
    assert "import failed" in out


# ---------------------------------------------------------------------------
# generate_and_iterate — mocked generate_strategy + backtest
# ---------------------------------------------------------------------------

def _make_validation(class_name="MyStrategy"):
    v = MagicMock()
    v.class_name = class_name
    v.passed = True
    return v


def _success_gen(class_name="MyStrategy", critic_verdict="PASS"):
    return {
        "success": True,
        "filepath": Path(f"/tmp/{class_name}.py"),
        "validation": _make_validation(class_name),
        "critic": {"verdict": critic_verdict, "summary": "ok", "issues": []},
        "generation_id": "gen-test-v0",
        "class_name": class_name,
    }


def test_iterative_accepts_on_first_turn_when_good():
    """No iteration needed if first attempt already meets the bar."""
    from strategy_generator import generate_and_iterate

    good_bt = {"success": True, "total_trades": 30, "profit_total_pct": 1.2, "sharpe": 0.8}

    with patch("strategy_generator.generate_strategy", return_value=_success_gen()) as mock_gen:
        result = generate_and_iterate(
            target_regime="all", max_turns=3,
            backtest_fn=lambda name: good_bt,
        )

    assert result["accepted"] is True
    assert result["turns_used"] == 1
    assert result["mini_backtest"]["total_trades"] == 30
    assert mock_gen.call_count == 1  # no retries


def test_iterative_rescues_on_second_turn():
    """First attempt 0 trades, second attempt has trades → ACCEPTED."""
    from strategy_generator import generate_and_iterate

    backtest_results = [
        {"success": True, "total_trades": 0, "profit_total_pct": 0, "sharpe": 0},
        {"success": True, "total_trades": 25, "profit_total_pct": 0.5, "sharpe": 0.3},
    ]
    bt_calls = iter(backtest_results)

    with patch("strategy_generator.generate_strategy", return_value=_success_gen()) as mock_gen:
        result = generate_and_iterate(
            target_regime="trending", max_turns=3,
            backtest_fn=lambda name: next(bt_calls),
        )

    assert result["accepted"] is True
    assert result["turns_used"] == 2
    assert mock_gen.call_count == 2


def test_iterative_feedback_appended_to_existing_results():
    """When the first turn fails, the next generate_strategy call must
    receive the backtest diagnostic in existing_results."""
    from strategy_generator import generate_and_iterate

    bt_results = [
        {"success": True, "total_trades": 0, "profit_total_pct": 0, "sharpe": 0},
        {"success": True, "total_trades": 20, "profit_total_pct": 1.0, "sharpe": 0.5},
    ]
    bt_calls = iter(bt_results)
    seen_existing = []

    def fake_generate_strategy(**kwargs):
        seen_existing.append(kwargs["existing_results"])
        return _success_gen()

    with patch("strategy_generator.generate_strategy", side_effect=fake_generate_strategy):
        generate_and_iterate(
            target_regime="all", max_turns=3,
            existing_results="seed context",
            backtest_fn=lambda name: next(bt_calls),
        )

    # First call sees only seed
    assert seen_existing[0] == "seed context"
    # Second call sees seed + backtest diagnostic
    assert "seed context" in seen_existing[1]
    assert "PREVIOUS ATTEMPT BACKTEST RESULT" in seen_existing[1]
    assert "ZERO TRADES" in seen_existing[1]


def test_iterative_returns_best_when_exhausted():
    """If all turns fail acceptance, return the BEST attempt seen (most trades + profit)."""
    from strategy_generator import generate_and_iterate

    bt_results = [
        {"success": True, "total_trades": 0, "profit_total_pct": 0, "sharpe": 0},      # turn 0
        {"success": True, "total_trades": 3, "profit_total_pct": -1.0, "sharpe": -0.2}, # turn 1 — better than 0
        {"success": True, "total_trades": 2, "profit_total_pct": -0.5, "sharpe": -0.1}, # turn 2 — worse than turn 1
    ]
    bt_calls = iter(bt_results)

    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=3,
            backtest_fn=lambda name: next(bt_calls),
        )

    assert result["accepted"] is False
    assert result["turns_used"] == 3
    # Best should be turn 1 (3 trades, even though unprofitable, beats 0 trades)
    assert result["mini_backtest"]["total_trades"] == 3
    assert result["turn"] == 2  # 1-indexed in the saved best


def test_iterative_bails_on_generation_failure():
    """If generate_strategy itself fails (spec parse, validation, etc.), no point looping."""
    from strategy_generator import generate_and_iterate

    fail = {"success": False, "error": "spec parse failed", "generation_id": "gen-bad"}
    with patch("strategy_generator.generate_strategy", return_value=fail) as mock_gen:
        result = generate_and_iterate(
            target_regime="all", max_turns=3,
            backtest_fn=lambda name: pytest.fail("should not be called"),
        )

    assert result["success"] is False
    assert mock_gen.call_count == 1  # bailed after first generation failure


def test_iterative_accepts_unprofitable_if_sharpe_positive():
    """profit > 0 OR sharpe > 0 — either alone is enough."""
    from strategy_generator import generate_and_iterate

    bt = {"success": True, "total_trades": 50, "profit_total_pct": -0.1, "sharpe": 0.2}
    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=3,
            backtest_fn=lambda name: bt,
        )
    assert result["accepted"] is True


def test_iterative_rejects_if_below_min_trades_floor():
    """4 trades shouldn't pass even with great profit — we want statistical signal."""
    from strategy_generator import generate_and_iterate

    bt = {"success": True, "total_trades": 4, "profit_total_pct": 10.0, "sharpe": 5.0}
    with patch("strategy_generator.generate_strategy", return_value=_success_gen()):
        result = generate_and_iterate(
            target_regime="all", max_turns=2,
            backtest_fn=lambda name: bt, accept_min_trades=5,
        )
    assert result["accepted"] is False


# ---------------------------------------------------------------------------
# generate_batch with iterative=True
# ---------------------------------------------------------------------------

_BATCH_SUCCESS = {"success": True, "filepath": Path("/tmp/MyStrategy.py")}


def test_generate_batch_iterative_routes_to_iterate():
    """When iterative=True, generate_batch calls generate_and_iterate, not generate_strategy."""
    from strategy_generator import generate_batch

    with patch("strategy_generator.generate_and_iterate", return_value=_BATCH_SUCCESS) as mock_iter, \
         patch("strategy_generator.generate_strategy", side_effect=AssertionError("should not be called")):
        results = generate_batch(count=2, iterative=True, max_turns=2)

    assert len(results) == 2
    assert mock_iter.call_count == 2


def test_generate_batch_non_iterative_uses_single_shot():
    """Default (iterative=False) preserves single-shot behavior."""
    from strategy_generator import generate_batch

    with patch("strategy_generator.generate_strategy", return_value=_BATCH_SUCCESS) as mock_gen, \
         patch("strategy_generator.generate_and_iterate", side_effect=AssertionError("should not be called")):
        results = generate_batch(count=2)

    assert len(results) == 2
    assert mock_gen.call_count == 2
