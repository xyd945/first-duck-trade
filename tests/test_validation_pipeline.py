"""Tests for the validation pipeline — the safety net for LLM-generated code."""

import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'user_data' / 'scripts'))

from validation_pipeline import validate_strategy_file, validate_backtest_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def write_temp_strategy(code: str) -> Path:
    """Write code to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    f.write(code)
    f.close()
    return Path(f.name)


VALID_STRATEGY = '''
from freqtrade.strategy import IStrategy, IntParameter
from strategies.base_generated import BaseGeneratedStrategy
from pandas import DataFrame
import pandas_ta as ta
import numpy as np

class TestStrategy(BaseGeneratedStrategy):
    STRATEGY_THESIS = "Test strategy for validation"
    TARGET_REGIME = "all"
    GENERATION_ID = "test-001"

    fast_ema = IntParameter(10, 30, default=20, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema'] = ta.ema(dataframe['close'], length=self.fast_ema.value)
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['ema'].shift(1)) &
            (dataframe['rsi'].shift(1) < 30) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'].shift(1) > 70) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        return dataframe
'''


# ---------------------------------------------------------------------------
# Stage 1: Security checks
# ---------------------------------------------------------------------------
class TestSecurityChecks:
    def test_valid_strategy_passes(self):
        path = write_temp_strategy(VALID_STRATEGY)
        result = validate_strategy_file(path)
        assert result.passed, f"Valid strategy failed: {result}"

    def test_rejects_os_import(self):
        code = VALID_STRATEGY.replace(
            "import numpy as np",
            "import numpy as np\nimport os"
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed
        assert any("os" in e for e in result.errors)

    def test_rejects_subprocess_import(self):
        code = VALID_STRATEGY.replace(
            "import numpy as np",
            "import numpy as np\nimport subprocess"
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed

    def test_rejects_exec_call(self):
        code = VALID_STRATEGY.replace(
            "return dataframe",
            "exec('print(1)')\n        return dataframe",
            1  # Only replace first occurrence
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed
        assert any("exec" in e.lower() for e in result.errors)

    def test_rejects_eval_call(self):
        code = VALID_STRATEGY.replace(
            "return dataframe",
            "x = eval('1+1')\n        return dataframe",
            1
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed

    def test_rejects_open_call(self):
        code = "import os\nf = open('/etc/passwd')\n"
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed

    def test_rejects_requests_import(self):
        code = VALID_STRATEGY.replace(
            "import numpy as np",
            "import numpy as np\nimport requests"
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed

    def test_allows_pandas_import(self):
        result = validate_strategy_file(write_temp_strategy(VALID_STRATEGY))
        assert result.passed

    def test_allows_pandas_ta_import(self):
        result = validate_strategy_file(write_temp_strategy(VALID_STRATEGY))
        assert result.passed


# ---------------------------------------------------------------------------
# Stage 2: Look-ahead bias detection
# ---------------------------------------------------------------------------
class TestLookAheadDetection:
    def test_rejects_negative_shift(self):
        code = VALID_STRATEGY.replace(
            "dataframe['ema'].shift(1)",
            "dataframe['ema'].shift(-1)"
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed
        assert any("look-ahead" in e.lower() or "shift" in e.lower() for e in result.errors)

    def test_accepts_positive_shift(self):
        result = validate_strategy_file(write_temp_strategy(VALID_STRATEGY))
        assert result.passed

    def test_rejects_centered_rolling(self):
        code = VALID_STRATEGY.replace(
            "ta.ema(dataframe['close'], length=self.fast_ema.value)",
            "dataframe['close'].rolling(window=10, center=True).mean()"
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed
        assert any("center" in e.lower() or "rolling" in e.lower() for e in result.errors)

    def test_accepts_normal_rolling(self):
        code = VALID_STRATEGY.replace(
            "ta.ema(dataframe['close'], length=self.fast_ema.value)",
            "dataframe['close'].rolling(window=10).mean()"
        )
        result = validate_strategy_file(write_temp_strategy(code))
        assert result.passed


# ---------------------------------------------------------------------------
# Stage 3: Structure checks
# ---------------------------------------------------------------------------
class TestStructureChecks:
    def test_requires_base_class(self):
        code = VALID_STRATEGY.replace("BaseGeneratedStrategy", "IStrategy")
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed
        assert any("BaseGeneratedStrategy" in e for e in result.errors)

    def test_requires_populate_methods(self):
        # Remove populate_entry_trend
        code = VALID_STRATEGY.split("def populate_entry_trend")[0]
        result = validate_strategy_file(write_temp_strategy(code))
        assert not result.passed
        assert any("missing" in e.lower() for e in result.errors)

    def test_warns_on_missing_metadata(self):
        code = VALID_STRATEGY.replace(
            '    STRATEGY_THESIS = "Test strategy for validation"\n', ''
        )
        result = validate_strategy_file(write_temp_strategy(code))
        # Should pass but with warnings
        assert result.passed
        assert len(result.warnings) > 0

    def test_empty_file_fails(self):
        result = validate_strategy_file(write_temp_strategy(""))
        assert not result.passed

    def test_syntax_error_fails(self):
        result = validate_strategy_file(write_temp_strategy("def broken(:\n  pass"))
        assert not result.passed
        assert any("syntax" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Post-backtest validation
# ---------------------------------------------------------------------------
class TestBacktestValidation:
    def test_too_few_trades_fails(self):
        result = validate_backtest_results({"total_trades": 2, "backtest_days": 180})
        assert not result.passed

    def test_excessive_trading_fails(self):
        result = validate_backtest_results({
            "total_trades": 10000,
            "backtest_days": 30,
            "starting_balance": 1000,
            "max_drawdown_abs": 0,
        })
        assert not result.passed
        assert any("excessive" in e.lower() for e in result.errors)

    def test_excessive_drawdown_fails(self):
        result = validate_backtest_results({
            "total_trades": 50,
            "backtest_days": 180,
            "starting_balance": 1000,
            "max_drawdown_abs": 600,
        })
        assert not result.passed

    def test_normal_results_pass(self):
        result = validate_backtest_results({
            "total_trades": 50,
            "backtest_days": 180,
            "starting_balance": 1000,
            "max_drawdown_abs": 100,
        })
        assert result.passed
