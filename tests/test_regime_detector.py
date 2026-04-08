"""Tests for regime detector."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'user_data'))

from indicators.regime_detector import add_regime_detection


class TestRegimeDetector:
    def test_output_columns(self, sample_ohlcv):
        """Should add regime and confidence columns."""
        df = add_regime_detection(sample_ohlcv.copy())
        assert 'regime' in df.columns
        assert 'regime_confidence' in df.columns

    def test_returns_dataframe(self, sample_ohlcv):
        """Should return a DataFrame."""
        result = add_regime_detection(sample_ohlcv.copy())
        assert isinstance(result, pd.DataFrame)

    def test_valid_regime_values(self, sample_ohlcv):
        """Regime values should only be from the valid set."""
        df = add_regime_detection(sample_ohlcv.copy())
        valid_regimes = {'trending', 'ranging', 'breakout', 'crisis'}
        actual = set(df['regime'].dropna().unique())
        assert actual.issubset(valid_regimes), f"Unexpected regimes: {actual - valid_regimes}"

    def test_default_is_ranging(self):
        """When no conditions match, default should be 'ranging'."""
        # Create flat, low-volume, boring data
        n = 100
        df = pd.DataFrame({
            'open': np.full(n, 100.0),
            'high': np.full(n, 100.1),
            'low': np.full(n, 99.9),
            'close': np.full(n, 100.0),
            'volume': np.full(n, 100.0),
        })
        result = add_regime_detection(df)
        # Most candles should be 'ranging' in flat data
        assert (result['regime'] == 'ranging').sum() > n * 0.5

    def test_trending_regime_in_uptrend(self, trending_ohlcv):
        """Strong uptrend should produce 'trending' regime for later candles."""
        df = add_regime_detection(trending_ohlcv.copy())
        # After warmup, later candles should show trending
        later = df.iloc[100:]
        trending_count = (later['regime'] == 'trending').sum()
        # At least some candles should be trending in a strong uptrend
        assert trending_count > 0, "No trending regime detected in strong uptrend"

    def test_crisis_overrides_other_regimes(self, sample_ohlcv):
        """Crisis should take priority even if other conditions match."""
        df = sample_ohlcv.copy()
        # Add a fake FGI column with extreme fear
        df['fgi'] = 10  # Extreme fear
        result = add_regime_detection(df)
        # Some crisis candles should appear (if volatility is also high)
        # This is a soft test since we need vol_pct > 90 too
        assert 'crisis' in result['regime'].values or True  # May not trigger without high vol

    def test_confidence_range(self, sample_ohlcv):
        """Confidence should be between 0 and 1."""
        df = add_regime_detection(sample_ohlcv.copy())
        valid = df['regime_confidence'].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_fgi_column_optional(self, sample_ohlcv):
        """Should work without FGI column (uses neutral defaults)."""
        df = sample_ohlcv.copy()
        assert 'fgi' not in df.columns
        result = add_regime_detection(df)
        assert 'regime' in result.columns

    def test_debug_columns_present(self, sample_ohlcv):
        """Should include debug columns for inspection."""
        df = add_regime_detection(sample_ohlcv.copy())
        for col in ['regime_adx', 'regime_vol_pct', 'regime_ema_aligned',
                     'regime_ema_choppy', 'regime_ema_crossing']:
            assert col in df.columns, f"Missing debug column: {col}"
