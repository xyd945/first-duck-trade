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


def test_safe_fill_never_backfills():
    import pandas as pd
    from indicators.freqai_features import _safe_fill

    s = pd.Series([float("nan"), float("nan"), 5.0, float("nan"), 7.0])
    filled = _safe_fill(s)
    # Leading NaNs become 0 (not 5.0 — that would be look-ahead)
    assert filled.iloc[0] == 0.0
    assert filled.iloc[1] == 0.0
    assert filled.iloc[3] == 5.0  # ffill from the past is fine
