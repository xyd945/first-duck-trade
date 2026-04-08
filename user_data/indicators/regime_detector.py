"""
Regime Detector — Indicator-based market regime classification.

Classifies the current market into one of four regimes:
  - trending:  Strong directional move, EMAs aligned, ADX > 25
  - ranging:   Low momentum, choppy EMAs, neutral sentiment
  - breakout:  High volatility spike, EMAs crossing, ADX > 30
  - crisis:    Extreme volatility, fear dominant

Priority: crisis > breakout > trending > ranging (first match wins)
Default: "ranging" if no regime matches (most conservative)

Usage in Strategy:
    from indicators.regime_detector import add_regime_detection

    def populate_indicators(self, dataframe, metadata):
        dataframe = add_regime_detection(dataframe)
        # Now use: dataframe['regime'] — one of "trending", "ranging", "breakout", "crisis"
        return dataframe
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame


def add_regime_detection(
    dataframe: DataFrame,
    adx_length: int = 14,
    ema_short_length: int = 20,
    ema_long_length: int = 50,
    volatility_window: int = 30,
    ema_alignment_candles: int = 5,
    ema_chop_window: int = 20,
    ema_chop_min_crosses: int = 2,
    ema_cross_recency: int = 3,
    fgi_col: str = 'fgi',
) -> DataFrame:
    """
    Add regime detection columns to the dataframe.

    Parameters
    ----------
    dataframe : DataFrame
        OHLCV dataframe. If 'fgi' column exists, it will be used for
        Fear & Greed thresholds. Otherwise F&G conditions are skipped.
    adx_length : int
        Period for ADX calculation.
    ema_short_length : int
        Short EMA period (fast).
    ema_long_length : int
        Long EMA period (slow).
    volatility_window : int
        Rolling window for volatility percentile calculation.
    ema_alignment_candles : int
        Number of consecutive candles both EMAs must be rising/falling
        to count as "aligned".
    ema_chop_window : int
        Window to count EMA crosses for "choppy" detection.
    ema_chop_min_crosses : int
        Minimum crosses in chop_window to qualify as "choppy".
    ema_cross_recency : int
        A cross within this many candles counts as "crossing".
    fgi_col : str
        Column name for Fear & Greed index (0-100 scale).

    Returns
    -------
    DataFrame
        Original dataframe with added columns:
        - regime: str, one of "trending", "ranging", "breakout", "crisis"
        - regime_confidence: float, 0.0-1.0
    """

    # =========================================================================
    # 1. COMPUTE BASE INDICATORS
    # =========================================================================
    # ADX
    adx_result = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=adx_length)
    adx = adx_result[f'ADX_{adx_length}']

    # EMAs
    ema_short = ta.ema(dataframe['close'], length=ema_short_length)
    ema_long = ta.ema(dataframe['close'], length=ema_long_length)

    # 30-day rolling volatility percentile (using true range)
    tr = ta.true_range(dataframe['high'], dataframe['low'], dataframe['close'])
    vol_pct = tr.rolling(volatility_window).mean().rank(pct=True) * 100

    # =========================================================================
    # 2. EMA ALIGNMENT DETECTION
    # =========================================================================
    # "aligned" = EMA20 > EMA50 and both rising, OR EMA20 < EMA50 and both falling,
    # for ema_alignment_candles consecutive candles
    ema_short_rising = ema_short.diff() > 0
    ema_long_rising = ema_long.diff() > 0
    ema_short_falling = ema_short.diff() < 0
    ema_long_falling = ema_long.diff() < 0

    both_rising = ema_short_rising & ema_long_rising
    both_falling = ema_short_falling & ema_long_falling

    # Count consecutive candles of alignment
    aligned_up = (
        both_rising.rolling(ema_alignment_candles).sum() == ema_alignment_candles
    ) & (ema_short > ema_long)

    aligned_down = (
        both_falling.rolling(ema_alignment_candles).sum() == ema_alignment_candles
    ) & (ema_short < ema_long)

    is_aligned = aligned_up | aligned_down

    # =========================================================================
    # 3. EMA CROSS DETECTION
    # =========================================================================
    # Detect crosses: EMA short crosses EMA long (either direction)
    cross_above = (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))
    cross_below = (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))
    any_cross = cross_above | cross_below

    # "choppy" = 2+ crosses in the last 20 candles
    cross_count = any_cross.astype(int).rolling(ema_chop_window).sum()
    is_choppy = cross_count >= ema_chop_min_crosses

    # "crossing" = a cross within the last 3 candles
    is_crossing = any_cross.astype(int).rolling(ema_cross_recency).sum() > 0

    # =========================================================================
    # 4. FEAR & GREED THRESHOLDS
    # =========================================================================
    has_fgi = fgi_col in dataframe.columns
    if has_fgi:
        fgi = dataframe[fgi_col]
    else:
        # If no F&G data, use neutral values that don't block any regime
        fgi = pd.Series(50, index=dataframe.index)

    # =========================================================================
    # 5. REGIME CLASSIFICATION (priority order)
    # =========================================================================
    # Initialize with default
    regime = pd.Series('ranging', index=dataframe.index)
    confidence = pd.Series(0.5, index=dataframe.index)

    # --- Crisis (highest priority) ---
    is_crisis = (vol_pct > 90) & (fgi < 25)
    regime = np.where(is_crisis, 'crisis', regime)
    confidence = np.where(is_crisis, 0.8, confidence)

    # --- Breakout ---
    is_breakout = (adx > 30) & (vol_pct > 80) & is_crossing & ~is_crisis
    regime = np.where(is_breakout, 'breakout', regime)
    confidence = np.where(is_breakout, 0.7, confidence)

    # --- Trending ---
    is_trending = (adx > 25) & is_aligned & (fgi >= 50) & ~is_crisis & ~is_breakout
    regime = np.where(is_trending, 'trending', regime)
    confidence = np.where(is_trending, 0.7, confidence)

    # --- Ranging (default, but with explicit match for higher confidence) ---
    is_ranging_explicit = (adx < 20) & (vol_pct < 60) & is_choppy & (fgi >= 30) & (fgi <= 70)
    confidence = np.where(
        (regime == 'ranging') & is_ranging_explicit, 0.7, confidence
    )

    dataframe['regime'] = regime
    dataframe['regime_confidence'] = confidence.astype(float)

    # Store intermediate values for debugging/logging
    dataframe['regime_adx'] = adx
    dataframe['regime_vol_pct'] = vol_pct
    dataframe['regime_ema_aligned'] = is_aligned
    dataframe['regime_ema_choppy'] = is_choppy
    dataframe['regime_ema_crossing'] = is_crossing

    return dataframe
