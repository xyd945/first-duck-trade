"""Tests for the FreqAI candidate spec path (issue #47).

Covers:
  - validate_freqai_spec accepts the committed baseline and rejects every
    class of unsafe/out-of-bounds spec
  - render_freqai_strategy emits a declarations-only subclass that passes
    validate_freqai_strategy_file (and injection via the thesis string
    can't break out of its literal)
  - render_freqai_config wires horizon/training-window/model params into
    the freqai block with a unique per-candidate identifier
  - the feature library computes what specs can reference, without NaN
    poisoning or backfill look-ahead
"""

import copy
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).parent.parent / "user_data"
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE))

from freqai_spec import (  # noqa: E402
    FreqaiSpecError,
    render_freqai_config,
    render_freqai_strategy,
    validate_freqai_spec,
)

BASELINE_SPEC_PATH = BASE / "freqai_specs" / "baseline_lgbm.json"


@pytest.fixture
def spec():
    return json.loads(BASELINE_SPEC_PATH.read_text())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_baseline_spec_is_valid(spec):
    validate_freqai_spec(spec)  # must not raise


def _expect_error(spec, msg_fragment):
    with pytest.raises(FreqaiSpecError) as exc:
        validate_freqai_spec(spec)
    assert msg_fragment in str(exc.value)


def test_rejects_missing_required_field(spec):
    del spec["thresholds"]
    _expect_error(spec, "missing required fields")


def test_rejects_wrong_spec_type(spec):
    spec["spec_type"] = "rule"
    _expect_error(spec, "spec_type")


def test_rejects_bad_class_name(spec):
    spec["name"] = "lowercase_name; import os"
    _expect_error(spec, "PascalCase")


def test_rejects_unknown_feature(spec):
    spec["features"].append("my_custom_feature")
    _expect_error(spec, "unknown feature keys")


def test_rejects_too_few_features(spec):
    spec["features"] = ["rsi"]
    _expect_error(spec, "features must be a list")


def test_rejects_duplicate_features(spec):
    spec["features"] = ["rsi", "rsi", "roc"]
    _expect_error(spec, "duplicate")


def test_rejects_unknown_model_family(spec):
    spec["model"]["family"] = "MyCustomModel"
    _expect_error(spec, "model.family")


def test_rejects_out_of_bounds_model_param(spec):
    spec["model"]["params"]["n_estimators"] = 100000
    _expect_error(spec, "outside allowed range")


def test_rejects_unknown_model_param(spec):
    spec["model"]["params"]["objective"] = "my_evil_callable"
    _expect_error(spec, "unknown param")


def test_rejects_horizon_out_of_bounds(spec):
    spec["target"]["horizon_candles"] = 500
    _expect_error(spec, "horizon_candles")


def test_rejects_non_future_return_target(spec):
    spec["target"]["type"] = "classification"
    _expect_error(spec, "future_return")


def test_rejects_exit_threshold_above_entry(spec):
    spec["thresholds"] = {"entry": 0.005, "exit": 0.01}
    _expect_error(spec, "thresholds.exit")


def test_rejects_positive_stoploss(spec):
    spec["risk"]["stoploss"] = 0.06
    _expect_error(spec, "stoploss")


def test_accepts_all_valid_entry_gates(spec):
    for gate in ({"type": "none"}, {"type": "regime_match"},
                 {"type": "ema_trend", "period": 200},
                 {"type": "di_confidence", "di_threshold": 1.5}):
        spec["entry_gate"] = gate
        validate_freqai_spec(spec)  # must not raise


def test_rejects_unknown_gate_type(spec):
    spec["entry_gate"] = {"type": "my_custom_gate"}
    _expect_error(spec, "entry_gate.type")


def test_rejects_ema_gate_without_period(spec):
    spec["entry_gate"] = {"type": "ema_trend"}
    _expect_error(spec, "entry_gate.period")


def test_rejects_out_of_bounds_gate_params(spec):
    spec["entry_gate"] = {"type": "ema_trend", "period": 5000}
    _expect_error(spec, "entry_gate.period")
    spec["entry_gate"] = {"type": "di_confidence", "di_threshold": 99}
    _expect_error(spec, "di_threshold")


def test_rejects_stray_gate_keys(spec):
    spec["entry_gate"] = {"type": "regime_match", "period": 200}
    _expect_error(spec, "unexpected keys")


def test_rendered_strategy_carries_gate(spec, tmp_path):
    from validation_pipeline import validate_freqai_strategy_file

    spec["entry_gate"] = {"type": "ema_trend", "period": 168}
    code = render_freqai_strategy(spec)
    assert 'ENTRY_GATE_TYPE = "ema_trend"' in code
    assert "ENTRY_GATE_PERIOD = 168" in code
    f = tmp_path / f"{spec['name']}.py"
    f.write_text(code)
    assert validate_freqai_strategy_file(f).passed


def test_rendered_strategy_defaults_gate_to_none(spec):
    code = render_freqai_strategy(spec)
    assert 'ENTRY_GATE_TYPE = "none"' in code


def test_di_confidence_gate_sets_config_threshold(spec):
    spec["entry_gate"] = {"type": "di_confidence", "di_threshold": 1.5}
    config = render_freqai_config(spec)
    assert config["freqai"]["feature_parameters"]["DI_threshold"] == 1.5


def test_non_di_gate_leaves_config_threshold_unset(spec):
    spec["entry_gate"] = {"type": "regime_match"}
    config = render_freqai_config(spec)
    assert "DI_threshold" not in config["freqai"]["feature_parameters"]


def test_rejects_hostile_generation_id(spec):
    spec["generation_id"] = 'x"\nimport os\ny = "'
    _expect_error(spec, "generation_id")


def test_rejects_null_model_params(spec):
    spec["model"]["params"] = None
    validate_freqai_spec(spec)  # null coerces to {} — no params is fine
    spec["model"]["params"] = ["not", "a", "dict"]
    _expect_error(spec, "model.params must be an object")


def test_rejects_non_numeric_minimal_roi_values(spec):
    spec["risk"]["minimal_roi"] = {"0": True}
    _expect_error(spec, "must be a number")


def test_rejects_non_digit_minimal_roi_keys(spec):
    spec["risk"]["minimal_roi"] = {"1h": 0.05}
    _expect_error(spec, "string of digits")


def test_rejects_train_period_out_of_bounds(spec):
    spec["freqai"]["train_period_days"] = 5
    _expect_error(spec, "train_period_days")


def test_rejects_bad_indicator_periods(spec):
    spec["freqai"]["indicator_periods_candles"] = [14, 5000]
    _expect_error(spec, "indicator_periods_candles")


# ---------------------------------------------------------------------------
# Strategy rendering
# ---------------------------------------------------------------------------

def test_rendered_strategy_is_declarations_only(spec, tmp_path):
    from validation_pipeline import validate_freqai_strategy_file

    code = render_freqai_strategy(spec)
    f = tmp_path / f"{spec['name']}.py"
    f.write_text(code)

    result = validate_freqai_strategy_file(f)
    assert result.passed, str(result)
    # No computation in the file at all
    assert "def " not in code
    assert "shift(" not in code


def test_rendered_strategy_carries_spec_values(spec):
    code = render_freqai_strategy(spec)
    assert f"class {spec['name']}(BaseFreqaiStrategy)" in code
    assert "ENTRY_THRESHOLD = 0.005" in code
    assert "EXIT_THRESHOLD = 0.0" in code
    assert "stoploss = -0.06" in code
    assert '"macro_vix"' in code


def test_thesis_injection_cannot_escape_string_literal(spec, tmp_path):
    """A hostile thesis must stay inside its string literal (json.dumps
    escaping) — and if it somehow produced code, the validator rejects it."""
    spec["thesis"] = 'x"\nimport os\nos.system("rm -rf /")\ny = "'
    code = render_freqai_strategy(spec)

    import ast
    tree = ast.parse(code)  # still valid python
    # The whole payload must live inside the STRATEGY_THESIS constant.
    imports = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1  # only the base_freqai import

    from validation_pipeline import validate_freqai_strategy_file
    f = tmp_path / f"{spec['name']}.py"
    f.write_text(code)
    assert validate_freqai_strategy_file(f).passed


def test_validator_rejects_method_definitions(tmp_path):
    """The declarations-only rule: a subclass that overrides any method is
    rejected even if the method body looks harmless."""
    from validation_pipeline import validate_freqai_strategy_file

    f = tmp_path / "Sneaky.py"
    f.write_text('''
from base_freqai import BaseFreqaiStrategy

class Sneaky(BaseFreqaiStrategy):
    STRATEGY_THESIS = "t"
    TARGET_REGIME = "all"
    GENERATION_ID = "g"
    FREQAI_FEATURES = ["rsi"]
    ENTRY_THRESHOLD = 0.005
    EXIT_THRESHOLD = 0.0

    def set_freqai_targets(self, dataframe, metadata, **kwargs):
        return dataframe
''')
    result = validate_freqai_strategy_file(f)
    assert not result.passed
    assert any("declarations-only" in e for e in result.errors)


def test_validator_rejects_wrong_base_class(tmp_path):
    from validation_pipeline import validate_freqai_strategy_file

    f = tmp_path / "WrongBase.py"
    f.write_text('''
from base_generated import BaseGeneratedStrategy

class WrongBase(BaseGeneratedStrategy):
    STRATEGY_THESIS = "t"
''')
    result = validate_freqai_strategy_file(f)
    assert not result.passed
    assert any("BaseFreqaiStrategy" in e for e in result.errors)


def test_validator_rejects_missing_attrs(tmp_path):
    from validation_pipeline import validate_freqai_strategy_file

    f = tmp_path / "Bare.py"
    f.write_text('''
from base_freqai import BaseFreqaiStrategy

class Bare(BaseFreqaiStrategy):
    STRATEGY_THESIS = "t"
''')
    result = validate_freqai_strategy_file(f)
    assert not result.passed
    assert any("Missing required class attributes" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Config rendering
# ---------------------------------------------------------------------------

def test_config_wires_spec_into_freqai_block(spec):
    config = render_freqai_config(spec)
    freqai = config["freqai"]

    assert freqai["enabled"] is True
    assert freqai["identifier"] == spec["name"]
    assert freqai["train_period_days"] == 60
    assert freqai["backtest_period_days"] == 7
    fp = freqai["feature_parameters"]
    assert fp["label_period_candles"] == 24
    assert fp["include_shifted_candles"] == 2
    assert fp["indicator_periods_candles"] == [14, 50]
    assert freqai["data_split_parameters"]["test_size"] == 0.25
    mt = freqai["model_training_parameters"]
    assert mt["n_estimators"] == 400
    assert mt["learning_rate"] == 0.05


def test_config_stays_dry_run_without_secrets(spec):
    config = render_freqai_config(spec)
    assert config["dry_run"] is True
    assert config["exchange"]["key"] == ""
    assert config["exchange"]["secret"] == ""


def test_config_identifiers_unique_per_candidate(spec):
    other = copy.deepcopy(spec)
    other["name"] = "FreqaiOtherCandidate"
    a = render_freqai_config(spec)
    b = render_freqai_config(other)
    assert a["freqai"]["identifier"] != b["freqai"]["identifier"]


def test_config_defaults_apply_when_freqai_block_omitted(spec):
    del spec["freqai"]
    config = render_freqai_config(spec)
    base = json.loads(
        (BASE / "configs" / "config-freqai-base.json").read_text()
    )
    assert config["freqai"]["train_period_days"] == base["freqai"]["train_period_days"]
    # Horizon always comes from the spec target, never the base file
    assert (config["freqai"]["feature_parameters"]["label_period_candles"]
            == spec["target"]["horizon_candles"])


# ---------------------------------------------------------------------------
# Feature library
# ---------------------------------------------------------------------------

def test_expand_features_compute_on_sample(sample_ohlcv):
    from indicators.freqai_features import EXPAND_FEATURES, add_expand_features

    df = add_expand_features(sample_ohlcv.copy(), list(EXPAND_FEATURES), 14)
    for key in EXPAND_FEATURES:
        col = f"%-{key}-period"
        assert col in df.columns, f"missing {col}"
        assert not df[col].isna().any(), f"{col} has NaN after fill"


def test_basic_and_time_features(sample_ohlcv):
    from indicators.freqai_features import add_basic_features, add_standard_features

    df = add_basic_features(sample_ohlcv.copy(), ["pct_change", "hl_range"])
    assert "%-pct_change" in df.columns
    assert "%-hl_range" in df.columns

    df = add_standard_features(df, ["time_cycle"])
    assert "%-time-week-sin" in df.columns
    assert "%-time-week-cos" in df.columns
    assert df["%-time-week-sin"].between(-1, 1).all()


def test_future_return_target_looks_forward(sample_ohlcv):
    from indicators.freqai_features import add_future_return_target

    df = add_future_return_target(sample_ohlcv.copy(), 24)
    assert "&-future_return" in df.columns
    # The last `horizon` rows have no future — must be NaN, not filled
    assert df["&-future_return"].tail(24).isna().all()
    # Spot-check: value at i is the return from close[i] to close[i+24]
    i = 10
    expected = df["close"].iloc[i + 24] / df["close"].iloc[i] - 1.0
    assert df["&-future_return"].iloc[i] == pytest.approx(expected)


def test_regime_feature_uses_detector_encoding(sample_ohlcv):
    from indicators.freqai_features import REGIME_ENCODING, add_standard_features

    df = add_standard_features(sample_ohlcv.copy(), ["regime"])
    assert "%-regime" in df.columns
    assert set(df["%-regime"].unique()).issubset(set(REGIME_ENCODING.values()))


def test_entry_gate_mask_none_allows_everything(sample_ohlcv):
    from indicators.freqai_features import entry_gate_mask

    mask = entry_gate_mask(sample_ohlcv, "none")
    assert mask.all()
    # di_confidence acts through config/do_predict, not the mask
    assert entry_gate_mask(sample_ohlcv, "di_confidence").all()


def test_entry_gate_mask_ema_trend(trending_ohlcv):
    from indicators.freqai_features import entry_gate_mask

    mask = entry_gate_mask(trending_ohlcv, "ema_trend", period=50)
    # Warm-up rows can't confirm a trend -> no buying
    assert not mask.iloc[:10].any()
    # A strongly rising series ends above its EMA -> buying allowed
    assert mask.iloc[-20:].all()


def test_entry_gate_mask_ema_trend_blocks_downtrend(trending_ohlcv):
    from indicators.freqai_features import entry_gate_mask

    falling = trending_ohlcv.copy()
    falling["close"] = falling["close"].iloc[::-1].values
    mask = entry_gate_mask(falling, "ema_trend", period=50)
    assert not mask.iloc[-20:].any()


def test_entry_gate_mask_regime_match(sample_ohlcv):
    from indicators.freqai_features import entry_gate_mask
    from indicators.regime_detector import add_regime_detection

    regimes = add_regime_detection(sample_ohlcv.copy())["regime"]
    mask = entry_gate_mask(sample_ohlcv, "regime_match",
                           target_regime="ranging")
    assert mask.equals((regimes == "ranging").fillna(False))

    # 'all' blocks only crisis bars
    mask_all = entry_gate_mask(sample_ohlcv, "regime_match",
                               target_regime="all")
    assert mask_all.equals((regimes != "crisis").fillna(False))


def test_entry_gate_mask_unknown_type_raises(sample_ohlcv):
    from indicators.freqai_features import entry_gate_mask

    with pytest.raises(ValueError, match="unknown entry gate"):
        entry_gate_mask(sample_ohlcv, "sneaky_gate")


def test_safe_fill_never_backfills():
    import pandas as pd
    from indicators.freqai_features import _safe_fill

    s = pd.Series([float("nan"), float("nan"), 5.0, float("nan"), 7.0])
    filled = _safe_fill(s)
    # Leading NaNs become 0 (not 5.0 — that would be look-ahead)
    assert filled.iloc[0] == 0.0
    assert filled.iloc[1] == 0.0
    assert filled.iloc[3] == 5.0  # ffill from the past is fine


def test_safe_fill_neutralizes_inf():
    """Zero rolling-std windows produce ±inf (volume_z, bb_width) — LightGBM
    rejects inf, and ffill must not propagate it forward."""
    import numpy as np
    import pandas as pd
    from indicators.freqai_features import _safe_fill

    s = pd.Series([1.0, np.inf, -np.inf, 2.0])
    filled = _safe_fill(s)
    assert np.isfinite(filled).all()
    assert filled.iloc[1] == 1.0  # inf → NaN → ffilled from the past


def test_expand_features_raise_loudly_on_short_input(sample_ohlcv):
    """pandas_ta returns None when the window is shorter than the indicator
    period. Silently zero-filling that would score a candidate on fabricated
    features — it must raise instead (orchestrator turns it into a clean
    FAIL_BACKTEST retirement)."""
    from indicators.freqai_features import add_expand_features

    tiny = sample_ohlcv.head(5).copy()
    with pytest.raises(ValueError, match="returned no data"):
        add_expand_features(tiny, ["adx"], 50)
    # ema_dist does arithmetic on the pandas_ta result — must bail to the
    # same loud error, not a bare float-minus-None TypeError (seen live in
    # a data-head walk-forward window during the positive-control runs)
    with pytest.raises(ValueError, match="returned no data"):
        add_expand_features(tiny, ["ema_dist"], 50)


def test_rendered_generation_id_is_json_escaped(spec):
    """Defense in depth: even though validation restricts generation_id to a
    plain token, the template must emit it via json.dumps, never raw inside
    quotes."""
    code = render_freqai_strategy(spec)
    assert 'GENERATION_ID = "freqai-spec-baseline"' in code


def test_console_scraper_parses_suffixed_sharpe_labels():
    """freqtrade 2026.x labels the rows 'Sharpe (closed trades)' — the bare
    'Sharpe │' regex silently returned 0.0 for every walk-forward window
    (console-parse path), making gate_walk_forward fail everything."""
    from backtest_runner import parse_backtest_output

    output = (
        "│ Total profit %                         │ -10.13%     │\n"
        "│ Sharpe (closed trades)                 │ -7.50       │\n"
        "│ Sortino (closed trades)                │ -8.23       │\n"
        "│ Profit factor                          │ 0.60        │\n"
    )
    r = parse_backtest_output(output, "AnyStrategy")
    assert r["sharpe"] == -7.50
    assert r["sortino"] == -8.23
    assert r["profit_factor"] == 0.60
