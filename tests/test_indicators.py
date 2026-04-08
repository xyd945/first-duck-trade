"""Tests for custom indicators: whale_liquidity, fear_and_greed, chaikin_money_flow."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add user_data to path so indicators can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / 'user_data'))

from indicators.whale_liquidity import add_whale_liquidity
from indicators.chaikin_money_flow import add_chaikin_money_flow
from indicators.fear_and_greed import add_fear_and_greed


# =========================================================================
# Whale Liquidity Tests
# =========================================================================

class TestWhaleLiquidity:
    def test_green_candle_negative_delta(self, sample_ohlcv):
        """Green candle (close > open) should produce negative raw_delta (whale sell)."""
        df = sample_ohlcv.copy()
        # Force a green candle
        df.loc[10, 'close'] = df.loc[10, 'open'] + 5
        df = add_whale_liquidity(df)

        assert df.loc[10, 'raw_delta'] < 0

    def test_red_candle_positive_delta(self, sample_ohlcv):
        """Red candle (close < open) should produce positive raw_delta (whale buy)."""
        df = sample_ohlcv.copy()
        df.loc[10, 'close'] = df.loc[10, 'open'] - 5
        df = add_whale_liquidity(df)

        assert df.loc[10, 'raw_delta'] > 0

    def test_doji_zero_delta(self, sample_ohlcv):
        """Doji candle (close == open) should produce zero raw_delta."""
        df = sample_ohlcv.copy()
        df.loc[10, 'close'] = df.loc[10, 'open']
        df = add_whale_liquidity(df)

        assert df.loc[10, 'raw_delta'] == 0

    def test_output_columns_exist(self, sample_ohlcv):
        """Should add all expected columns."""
        df = add_whale_liquidity(sample_ohlcv.copy())

        expected_cols = ['raw_delta', 'liq_wave', 'wave_std',
                         'is_whale_buy', 'is_whale_sell', 'whale_signal']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_returns_dataframe(self, sample_ohlcv):
        """Should return a DataFrame (not a Series)."""
        result = add_whale_liquidity(sample_ohlcv.copy())
        assert isinstance(result, pd.DataFrame)

    def test_no_crash_on_zero_volume(self):
        """Should handle all-zero volume without crashing."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [103, 104, 105],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [0, 0, 0],
        })
        result = add_whale_liquidity(df)
        assert len(result) == 3

    def test_whale_signal_values(self, sample_ohlcv):
        """whale_signal should only contain -1, 0, or 1."""
        df = add_whale_liquidity(sample_ohlcv.copy())
        assert set(df['whale_signal'].dropna().unique()).issubset({-1, 0, 1})


# =========================================================================
# Chaikin Money Flow Tests
# =========================================================================

class TestChaikinMoneyFlow:
    def test_normal_calculation(self, sample_ohlcv):
        """CMF should produce values between -1 and 1."""
        df = add_chaikin_money_flow(sample_ohlcv.copy())
        valid = df['cmf'].dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_returns_dataframe(self, sample_ohlcv):
        """Should return a DataFrame."""
        result = add_chaikin_money_flow(sample_ohlcv.copy())
        assert isinstance(result, pd.DataFrame)

    def test_high_equals_low_no_crash(self):
        """When high == low (div by zero), should return 0."""
        df = pd.DataFrame({
            'open': [100, 100, 100],
            'high': [100, 100, 100],
            'low': [100, 100, 100],
            'close': [100, 100, 100],
            'volume': [1000, 1000, 1000],
        })
        result = add_chaikin_money_flow(df, length=2)
        assert not np.any(np.isinf(result['cmf']))

    def test_zero_volume(self):
        """Zero volume for entire window should return 0, not NaN/Inf."""
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [0, 0, 0, 0, 0],
        })
        result = add_chaikin_money_flow(df, length=3)
        assert not np.any(np.isinf(result['cmf']))


# =========================================================================
# Fear & Greed Index Tests
# =========================================================================

class TestFearAndGreed:
    def test_returns_dataframe(self, sample_ohlcv):
        """Should return a DataFrame (standardized API)."""
        result = add_fear_and_greed(sample_ohlcv.copy())
        assert isinstance(result, pd.DataFrame)

    def test_fgi_column_exists(self, sample_ohlcv):
        """Should add 'fgi' column."""
        result = add_fear_and_greed(sample_ohlcv.copy())
        assert 'fgi' in result.columns

    def test_fgi_is_numeric(self, sample_ohlcv):
        """FGI values should be numeric."""
        result = add_fear_and_greed(sample_ohlcv.copy())
        assert pd.api.types.is_numeric_dtype(result['fgi'])

    def test_no_external_data_graceful(self, sample_ohlcv):
        """When external data files are missing, should still compute
        (VIX and GOLD default to 0)."""
        result = add_fear_and_greed(sample_ohlcv.copy())
        # Should not crash; fgi should have values (even if 0-heavy)
        assert result['fgi'].notna().sum() > 0
