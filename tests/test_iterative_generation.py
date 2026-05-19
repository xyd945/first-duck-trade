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


def test_generate_batch_threads_attribution_per_regime():
    """get_attribution_for_regime callable must be invoked per-regime and
    its return value forwarded as attribution_patterns to generate_strategy."""
    from strategy_generator import generate_batch

    seen_regimes = []
    seen_attributions = []

    def fake_attr(regime):
        seen_regimes.append(regime)
        return f"ATTRIB_FOR_{regime}"

    def capture_call(**kwargs):
        seen_attributions.append((kwargs.get("target_regime"),
                                    kwargs.get("attribution_patterns")))
        return _BATCH_SUCCESS

    with patch("strategy_generator.generate_strategy", side_effect=capture_call):
        generate_batch(
            count=3,
            regimes=["trending", "ranging", "breakout"],
            get_attribution_for_regime=fake_attr,
        )

    assert seen_regimes == ["trending", "ranging", "breakout"]
    assert seen_attributions == [
        ("trending", "ATTRIB_FOR_trending"),
        ("ranging", "ATTRIB_FOR_ranging"),
        ("breakout", "ATTRIB_FOR_breakout"),
    ]


def test_generate_strategy_includes_attribution_patterns_in_prompt():
    """The new attribution_patterns kwarg should land in the user prompt."""
    from strategy_generator import build_generation_prompt

    prompt = build_generation_prompt(
        target_regime="trending",
        attribution_patterns="MAGIC_ATTRIBUTION_BLOCK_xyz123",
    )
    assert "MAGIC_ATTRIBUTION_BLOCK_xyz123" in prompt


def test_attribution_section_renders_above_failure_examples():
    """attribution_patterns is prescriptive ('aim here'), failure_examples is
    prohibitive ('don't do that'). The positive target should come first."""
    from strategy_generator import build_generation_prompt

    prompt = build_generation_prompt(
        target_regime="all",
        attribution_patterns="MARKER_ATTRIBUTION",
        failure_examples="MARKER_FAILURE",
    )
    attr_idx = prompt.index("MARKER_ATTRIBUTION")
    fail_idx = prompt.index("MARKER_FAILURE")
    assert attr_idx < fail_idx


# ---------------------------------------------------------------------------
# Phase 6 — generate_batch cells mode + archetype prompt injection
# ---------------------------------------------------------------------------

def test_build_prompt_injects_archetype_blurb():
    """When archetype is provided, the prompt must contain the archetype
    name and the validator-enforcement warning."""
    from strategy_generator import build_generation_prompt

    prompt = build_generation_prompt(
        target_regime="trending",
        archetype="momentum_continuation",
    )
    assert "ARCHETYPE: momentum_continuation" in prompt
    assert "spec validator will REJECT" in prompt
    # Blurb's thesis-specific content lands too
    assert "trend" in prompt.lower()


def test_build_prompt_archetype_lands_at_top():
    """Archetype is the strongest constraint — must come before regime context
    sections like failures and attribution. (Top-of-prompt has the strongest
    pull on reasoning model outputs.)"""
    from strategy_generator import build_generation_prompt

    prompt = build_generation_prompt(
        target_regime="trending",
        archetype="momentum_continuation",
        attribution_patterns="MARKER_ATTRIBUTION",
        failure_examples="MARKER_FAILURE",
    )
    archetype_idx = prompt.index("ARCHETYPE: momentum_continuation")
    attr_idx = prompt.index("MARKER_ATTRIBUTION")
    fail_idx = prompt.index("MARKER_FAILURE")
    assert archetype_idx < attr_idx
    assert archetype_idx < fail_idx


def test_build_prompt_without_archetype_is_legacy_shape():
    """Backward compat: legacy callers (no archetype) get the old prompt
    shape without the archetype block."""
    from strategy_generator import build_generation_prompt
    prompt = build_generation_prompt(target_regime="all")
    assert "ARCHETYPE:" not in prompt


def test_generate_batch_cells_mode_iterates_per_cell():
    """When cells is provided, generate_batch iterates each (archetype, regime)
    tuple and passes the archetype through to generate_strategy."""
    from strategy_generator import generate_batch

    seen = []

    def capture(**kwargs):
        seen.append({
            "target_regime": kwargs.get("target_regime"),
            "archetype": kwargs.get("archetype"),
        })
        return _BATCH_SUCCESS

    with patch("strategy_generator.generate_strategy", side_effect=capture):
        results = generate_batch(cells=[
            ("momentum_continuation", "trending"),
            ("mean_reversion", "ranging"),
            ("funding_contrarian", "all"),
        ])

    assert len(results) == 3
    assert seen == [
        {"target_regime": "trending", "archetype": "momentum_continuation"},
        {"target_regime": "ranging", "archetype": "mean_reversion"},
        {"target_regime": "all", "archetype": "funding_contrarian"},
    ]


def test_generate_batch_cells_mode_stamps_archetype_on_result():
    """The caller should be able to read back archetype + target_regime
    from the result dict without re-parsing the file."""
    from strategy_generator import generate_batch

    with patch("strategy_generator.generate_strategy", return_value=_BATCH_SUCCESS):
        results = generate_batch(cells=[("vol_squeeze", "breakout")])

    assert results[0]["archetype"] == "vol_squeeze"
    assert results[0]["target_regime"] == "breakout"


def test_generate_batch_cells_mode_routes_to_iterate_when_iterative_true():
    """cells mode + iterative=True must call generate_and_iterate, not the
    single-shot generate_strategy."""
    from strategy_generator import generate_batch

    with patch("strategy_generator.generate_and_iterate", return_value=_BATCH_SUCCESS) as mock_iter, \
         patch("strategy_generator.generate_strategy", side_effect=AssertionError("should not fire")):
        generate_batch(cells=[("mean_reversion", "ranging")], iterative=True, max_turns=2)

    assert mock_iter.call_count == 1
    call_kwargs = mock_iter.call_args.kwargs
    assert call_kwargs["archetype"] == "mean_reversion"
    assert call_kwargs["target_regime"] == "ranging"


def test_generate_batch_legacy_regimes_mode_still_works():
    """The pre-Phase-6 (count + regimes) signature must still function for
    tests and CLI uses."""
    from strategy_generator import generate_batch

    with patch("strategy_generator.generate_strategy", return_value=_BATCH_SUCCESS) as mock_gen:
        results = generate_batch(count=2, regimes=["trending", "ranging"])

    assert len(results) == 2
    assert mock_gen.call_count == 2
    # archetype is None in legacy mode
    for call in mock_gen.call_args_list:
        assert call.kwargs.get("archetype") is None
