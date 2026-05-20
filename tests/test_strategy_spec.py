"""Tests for R3: strategy spec validator + renderer."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _valid_spec(**overrides) -> dict:
    spec = {
        "name": "MyTestStrategy",
        "thesis": "Test thesis with \"quoted\" sections.",
        # Phase 6 — archetype required. mean_reversion coheres with ranging.
        "archetype": "mean_reversion",
        "target_regime": "ranging",
        "generation_id": "gen-test-v0",
        "timeframe": "1h",
        "indicators": [
            {
                "compute": "bb = ta.bbands(dataframe['close'], length=20, std=2.0)",
                "columns": [
                    {"name": "bb_lower", "source": "bb['BBL_20_2.0']"},
                    {"name": "bb_mid", "source": "bb['BBM_20_2.0']"},
                ],
            },
            {"compute": "dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)"},
        ],
        "params": [
            {"name": "rsi_oversold", "type": "int", "low": 20, "high": 40,
             "default": 30, "space": "buy"},
            {"name": "rsi_exit", "type": "int", "low": 60, "high": 80,
             "default": 70, "space": "sell"},
        ],
        "entry": {
            "core": [
                "dataframe['rsi'].shift(1) < self.rsi_oversold.value",
                "dataframe['close'].shift(1) < dataframe['bb_lower'].shift(1)",
            ],
            "macro_confidence": [
                "dataframe['fgi'] < 0",
                "dataframe['btc_funding_rate'] < 0.0003",
            ],
            "macro_min_confidence": 0.5,
        },
        "exit": {
            "core": ["dataframe['rsi'] > self.rsi_exit.value"],
        },
        "risk": {
            "stoploss": -0.05,
            "minimal_roi": {"0": 0.10, "60": 0.05, "240": 0.02},
            "max_open_trades": 3,
        },
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# validate_spec
# ---------------------------------------------------------------------------

def test_valid_spec_passes():
    from strategy_spec import validate_spec
    validate_spec(_valid_spec())  # no raise


def test_missing_required_field_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    del spec["entry"]
    with pytest.raises(SpecError, match="missing required"):
        validate_spec(spec)


def test_bad_class_name_raises():
    from strategy_spec import validate_spec, SpecError
    with pytest.raises(SpecError, match="class identifier"):
        validate_spec(_valid_spec(name="not-a-valid-name"))
    with pytest.raises(SpecError, match="class identifier"):
        validate_spec(_valid_spec(name="lowercase"))


def test_bad_regime_raises():
    from strategy_spec import validate_spec, SpecError
    with pytest.raises(SpecError, match="target_regime"):
        validate_spec(_valid_spec(target_regime="reversal"))


def test_empty_entry_core_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec["entry"]["core"] = []
    with pytest.raises(SpecError, match="entry.core"):
        validate_spec(spec)


def test_macro_min_confidence_out_of_range_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec["entry"]["macro_min_confidence"] = 1.5
    with pytest.raises(SpecError, match="macro_min_confidence"):
        validate_spec(spec)


def test_positive_stoploss_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec["risk"]["stoploss"] = 0.05  # positive — wrong sign
    with pytest.raises(SpecError, match="stoploss"):
        validate_spec(spec)


def test_bad_param_type_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec["params"][0]["type"] = "float"  # not in int/decimal/bool
    with pytest.raises(SpecError, match="param type"):
        validate_spec(spec)


# ---------------------------------------------------------------------------
# Column reference cross-check
# Both failure patterns observed in real LLM output across multiple trials.
# ---------------------------------------------------------------------------

def test_undeclared_column_in_exit_raises():
    """Trial #2 cell 1: exit referenced dataframe['rsi'] but indicators
    never computed RSI → KeyError at backtest, burning the cell."""
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    # Drop the indicator that declares 'rsi'
    spec["indicators"] = [
        {"compute": "bb = ta.bbands(dataframe['close'], length=20, std=2.0)",
         "columns": [{"name": "bb_lower", "source": "bb['BBL_20_2.0']"}]},
    ]
    # Entry/exit still reference dataframe['rsi'] and dataframe['bb_lower']
    spec["entry"]["core"] = ["dataframe['rsi'] < 30"]
    spec["exit"]["core"] = ["dataframe['rsi'] > 70"]
    with pytest.raises(SpecError, match="references dataframe.*'rsi'"):
        validate_spec(spec)


def test_undeclared_column_in_entry_core_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec["entry"]["core"] = ["dataframe['nonexistent_col'] > 0"]
    with pytest.raises(SpecError, match="entry.core references dataframe.*'nonexistent_col'"):
        validate_spec(spec)


def test_undeclared_column_in_macro_confidence_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec["entry"]["macro_confidence"] = ["dataframe['made_up_macro'] < 1"]
    with pytest.raises(SpecError, match="entry.macro_confidence references"):
        validate_spec(spec)


def test_ohlcv_columns_are_implicitly_declared():
    """open/high/low/close/volume must always be valid without an indicator."""
    from strategy_spec import validate_spec
    spec = _valid_spec()
    spec["entry"]["core"] = ["dataframe['close'] > dataframe['open']"]
    spec["exit"]["core"] = ["dataframe['rsi'] > 70"]  # rsi declared in fixture
    validate_spec(spec)  # must not raise


def test_macro_columns_are_implicitly_declared():
    """fgi/vix/funding_rate/etc are added by add_external_data() — implicit."""
    from strategy_spec import validate_spec
    spec = _valid_spec()
    spec["entry"]["macro_confidence"] = [
        "dataframe['fgi'] < 0",
        "dataframe['vix'] < 25",
        "dataframe['btc_funding_rate'] < 0.0003",
        "dataframe['eth_btc_ratio'] > 0.05",
    ]
    validate_spec(spec)


def test_inline_compute_with_columns_block_raises():
    """Trial #2 cell 2: compute had inline `dataframe['dc_upper'] = ...`
    AND a columns block with `{name: dc_upper, source: dc_upper}`. The
    renderer emitted `dataframe['dc_upper'] = dc_upper` referencing an
    undefined local → NameError at backtest."""
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec["indicators"] = [{
        "compute": "dataframe['dc_upper'] = ta.donchian(dataframe['high'], dataframe['low'], length=20)['DCU_20_20']",
        "columns": [{"name": "dc_upper", "source": "dc_upper"}],
    }]
    spec["entry"]["core"] = ["dataframe['dc_upper'] > 0"]
    spec["exit"]["core"] = ["dataframe['rsi'] > 70"]
    with pytest.raises(SpecError, match="BOTH an inline.*assignment.*AND a 'columns' block"):
        validate_spec(spec)


def test_inline_compute_without_columns_passes():
    """The renderer supports `dataframe['x'] = ta.foo(...)` as a single line.
    That's the canonical pattern — must not be rejected."""
    from strategy_spec import validate_spec
    spec = _valid_spec()
    spec["indicators"] = [
        {"compute": "dataframe['ema_20'] = ta.ema(dataframe['close'], length=20)"},
    ]
    spec["entry"]["core"] = ["dataframe['close'] > dataframe['ema_20']"]
    spec["exit"]["core"] = ["dataframe['close'] < dataframe['ema_20']"]
    validate_spec(spec)


def test_columns_block_with_local_var_compute_passes():
    """The canonical multi-column pattern: compute introduces a local var,
    columns extract from it. Renderer adds the `dataframe['x'] = ...`."""
    from strategy_spec import validate_spec
    spec = _valid_spec()
    spec["indicators"] = [{
        "compute": "macd = ta.macd(dataframe['close'])",
        "columns": [
            {"name": "macd_line", "source": "macd['MACD_12_26_9']"},
            {"name": "macd_hist", "source": "macd['MACDh_12_26_9']"},
        ],
    }]
    spec["entry"]["core"] = ["dataframe['macd_hist'] > 0", "dataframe['macd_line'] > 0"]
    spec["exit"]["core"] = ["dataframe['macd_hist'] < 0"]
    validate_spec(spec)


def test_error_message_lists_available_indicator_columns():
    """Failure message should help the LLM self-correct on the next turn."""
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()  # declares bb_lower, bb_mid, rsi
    spec["entry"]["core"] = ["dataframe['ema_50'] > 0"]
    with pytest.raises(SpecError) as exc:
        validate_spec(spec)
    msg = str(exc.value)
    # Should list what IS available so the LLM knows what to add or fix
    assert "bb_lower" in msg
    assert "rsi" in msg


# ---------------------------------------------------------------------------
# render_strategy
# ---------------------------------------------------------------------------

def test_rendered_code_compiles_as_python():
    """The renderer's output must be syntactically valid Python."""
    import ast
    from strategy_spec import render_strategy
    code = render_strategy(_valid_spec())
    ast.parse(code)  # raises SyntaxError on invalid


def test_rendered_code_inherits_base_class():
    from strategy_spec import render_strategy
    code = render_strategy(_valid_spec())
    assert "class MyTestStrategy(BaseGeneratedStrategy):" in code
    assert "from base_generated import BaseGeneratedStrategy" in code


def test_rendered_code_imports_external_data():
    from strategy_spec import render_strategy
    code = render_strategy(_valid_spec())
    assert "from indicators.external_data import add_external_data" in code
    assert "dataframe = add_external_data(dataframe)" in code


def test_rendered_code_has_all_three_populate_methods():
    from strategy_spec import render_strategy
    code = render_strategy(_valid_spec())
    assert "def populate_indicators(self" in code
    assert "def populate_entry_trend(self" in code
    assert "def populate_exit_trend(self" in code


def test_rendered_code_passes_validation_pipeline(tmp_path):
    """End-to-end: rendered code must pass the existing security/structure validator."""
    from strategy_spec import render_strategy
    from validation_pipeline import validate_strategy_file

    code = render_strategy(_valid_spec())
    fp = tmp_path / "Strategy_render_test.py"
    fp.write_text(code)
    result = validate_strategy_file(fp)
    assert result.passed, f"validation failed: {result}"


def test_rendered_macro_uses_confidence_not_and(tmp_path):
    """Macro conditions must render as a mean-confidence check, NOT as an
    additional AND gate. This is the whole point of R3."""
    from strategy_spec import render_strategy
    code = render_strategy(_valid_spec())
    assert "macro_pass" in code
    assert "macro_score" in code
    assert "macro_score >= 0.5" in code
    # Each macro condition must be wrapped with .fillna(False) to handle NaN
    assert ".fillna(False)" in code


def test_no_macro_conditions_renders_macro_pass_true():
    """An empty macro_confidence list → entry uses only core (macro_pass=True)."""
    from strategy_spec import render_strategy
    spec = _valid_spec()
    spec["entry"]["macro_confidence"] = []
    code = render_strategy(spec)
    assert "macro_pass = True" in code
    assert "macro_score" not in code


def test_params_render_correctly():
    from strategy_spec import render_strategy
    code = render_strategy(_valid_spec())
    assert "rsi_oversold = IntParameter(20, 40, default=30, space=\"buy\")" in code
    assert "rsi_exit = IntParameter(60, 80, default=70, space=\"sell\")" in code


def test_indicator_columns_assigned():
    from strategy_spec import render_strategy
    code = render_strategy(_valid_spec())
    assert "bb = ta.bbands(dataframe['close'], length=20, std=2.0)" in code
    assert "dataframe['bb_lower'] = bb['BBL_20_2.0']" in code
    assert "dataframe['bb_mid'] = bb['BBM_20_2.0']" in code
    assert "dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)" in code


def test_thesis_with_quotes_does_not_break_render():
    """The thesis has quotes — must be safely escaped in the rendered class."""
    import ast
    from strategy_spec import render_strategy
    spec = _valid_spec(thesis='Use "Bollinger" + RSI with a "twist"')
    code = render_strategy(spec)
    ast.parse(code)


# ---------------------------------------------------------------------------
# _extract_spec_json (the parser used by generate_strategy)
# ---------------------------------------------------------------------------

def test_extract_spec_json_plain():
    from strategy_generator import _extract_spec_json
    text = '{"name": "Foo", "x": 1}'
    assert _extract_spec_json(text) == {"name": "Foo", "x": 1}


def test_extract_spec_json_fenced():
    from strategy_generator import _extract_spec_json
    text = '```json\n{"name": "Foo", "x": 2}\n```'
    assert _extract_spec_json(text) == {"name": "Foo", "x": 2}


def test_extract_spec_json_with_prose():
    from strategy_generator import _extract_spec_json
    text = 'Here is the spec:\n{"name": "Foo", "nested": {"a": 1}}\n\nLet me know.'
    assert _extract_spec_json(text) == {"name": "Foo", "nested": {"a": 1}}


def test_extract_spec_json_handles_nested_braces():
    from strategy_generator import _extract_spec_json
    text = '{"a": {"b": {"c": 1}}}'
    assert _extract_spec_json(text) == {"a": {"b": {"c": 1}}}


def test_extract_spec_json_returns_none_on_garbage():
    from strategy_generator import _extract_spec_json
    assert _extract_spec_json("no json here at all") is None


def test_extract_spec_json_returns_none_on_unbalanced():
    from strategy_generator import _extract_spec_json
    assert _extract_spec_json('{"a": 1') is None


# ---------------------------------------------------------------------------
# Phase 6 — archetype field
# ---------------------------------------------------------------------------

def test_missing_archetype_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec()
    spec.pop("archetype")
    with pytest.raises(SpecError, match="missing required fields"):
        validate_spec(spec)


def test_invalid_archetype_raises():
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec(archetype="not_a_real_archetype")
    with pytest.raises(SpecError, match="archetype must be one of"):
        validate_spec(spec)


def test_incoherent_archetype_regime_pair_raises():
    """mean_reversion + trending is a category error (catching falling knives).
    Spec validator must reject it before render."""
    from strategy_spec import validate_spec, SpecError
    spec = _valid_spec(archetype="mean_reversion", target_regime="trending")
    with pytest.raises(SpecError, match="does not cohere"):
        validate_spec(spec)


def test_coherent_archetype_regime_pair_accepted():
    from strategy_spec import validate_spec
    spec = _valid_spec(archetype="momentum_continuation", target_regime="trending")
    validate_spec(spec)  # should not raise


def test_render_emits_strategy_archetype_class_attr():
    """The renderer's STRATEGY_ARCHETYPE class attr lets the orchestrator
    recover the archetype by parsing the file (defense in depth — the
    primary signal is from generate_batch's stamped result dict)."""
    from strategy_spec import render_strategy
    spec = _valid_spec(archetype="vol_squeeze", target_regime="breakout")
    code = render_strategy(spec)
    assert 'STRATEGY_ARCHETYPE = "vol_squeeze"' in code
    assert 'TARGET_REGIME = "breakout"' in code
