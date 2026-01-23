"""
Whale Liquidity Indicator (LIQ v3: Neutral Base + Whale Spikes)

Converted from TradingView Pine Script to Python/Pandas for Freqtrade.

Original Concept:
- Detects "passive liquidity" by inverting the traditional volume delta
- If price closed higher → Crowd bought (Taker Buy) → Whales likely Sold (Passive Sell)
- If price closed lower → Crowd sold (Taker Sell) → Whales likely Bought (Passive Buy)

Outputs:
- liq_wave: Smoothed liquidity wave (ALMA smoothed)
- is_whale_buy: Boolean spike detection for whale buying activity
- is_whale_sell: Boolean spike detection for whale selling activity

Usage in Strategy:
    from indicators.whale_liquidity import add_whale_liquidity
    
    def populate_indicators(self, dataframe, metadata):
        dataframe = add_whale_liquidity(dataframe, smooth_len=40, spike_threshold=3.0)
        # Now use: dataframe['liq_wave'], dataframe['is_whale_buy'], dataframe['is_whale_sell']
        return dataframe
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame


def add_whale_liquidity(
    dataframe: DataFrame,
    smooth_len: int = 40,
    spike_threshold: float = 3.0,
    stdev_period: int = 100,
    col_prefix: str = ''
) -> DataFrame:
    """
    Add whale liquidity detection columns to the dataframe.

    This indicator detects significant liquidity movements by:
    1. Calculating "inverted delta" (passive liquidity flow)
    2. Smoothing with ALMA to create a wave
    3. Detecting "whale spikes" when the wave exceeds N standard deviations

    Parameters:
    -----------
    dataframe : DataFrame
        OHLCV dataframe with 'open', 'close', 'volume' columns
    smooth_len : int, default=40
        ALMA smoothing length for the base wave
    spike_threshold : float, default=3.0
        Strictness multiplier for whale detection (higher = fewer, more significant spikes)
    stdev_period : int, default=100
        Period for calculating standard deviation of the wave
    col_prefix : str, default=''
        Optional prefix for column names (useful if using multiple instances)

    Returns:
    --------
    DataFrame
        Original dataframe with added columns:
        - {prefix}raw_delta: Raw inverted delta (before smoothing)
        - {prefix}liq_wave: ALMA smoothed liquidity wave
        - {prefix}wave_std: Rolling standard deviation of the wave
        - {prefix}is_whale_buy: Boolean, True when whale buying detected
        - {prefix}is_whale_sell: Boolean, True when whale selling detected
        - {prefix}whale_signal: Categorical (-1=sell, 0=neutral, 1=buy)
    """

    # =========================================================================
    # 1. PASSIVE LIQUIDITY CALCULATION (Inverted Delta)
    # =========================================================================
    # If Price Closed Higher -> Crowd bought (Taker Buy) -> Whales likely Sold -> Negative Delta
    # If Price Closed Lower  -> Crowd sold (Taker Sell) -> Whales likely Bought -> Positive Delta
    # If Doji (open == close) -> Neutral

    dataframe[f'{col_prefix}raw_delta'] = np.where(
        dataframe['close'] > dataframe['open'],
        -dataframe['volume'],  # Green candle = whale passive sell
        np.where(
            dataframe['close'] < dataframe['open'],
            dataframe['volume'],  # Red candle = whale passive buy
            0  # Doji = neutral
        )
    )

    # =========================================================================
    # 2. SMOOTHING - ALMA (Arnaud Legoux Moving Average)
    # =========================================================================
    # ALMA parameters: offset=0.85, sigma=6 (Pine Script defaults)
    # pandas_ta.alma(series, length, offset, sigma)

    dataframe[f'{col_prefix}liq_wave'] = ta.alma(
        dataframe[f'{col_prefix}raw_delta'],
        length=smooth_len,
        offset=0.85,
        sigma=6
    )

    # =========================================================================
    # 3. WHALE SPIKE DETECTION
    # =========================================================================
    # Calculate rolling standard deviation of the wave
    dataframe[f'{col_prefix}wave_std'] = dataframe[f'{col_prefix}liq_wave'].rolling(
        window=stdev_period
    ).std()

    # Whale Buy: wave > (stdev * threshold) - Strong positive spike
    dataframe[f'{col_prefix}is_whale_buy'] = (
        dataframe[f'{col_prefix}liq_wave'] > (dataframe[f'{col_prefix}wave_std'] * spike_threshold)
    )

    # Whale Sell: wave < -(stdev * threshold) - Strong negative spike
    dataframe[f'{col_prefix}is_whale_sell'] = (
        dataframe[f'{col_prefix}liq_wave'] < -(dataframe[f'{col_prefix}wave_std'] * spike_threshold)
    )

    # =========================================================================
    # 4. CATEGORICAL SIGNAL (For easy strategy use)
    # =========================================================================
    # -1 = Whale Sell, 0 = Neutral, 1 = Whale Buy
    dataframe[f'{col_prefix}whale_signal'] = np.where(
        dataframe[f'{col_prefix}is_whale_buy'],
        1,
        np.where(
            dataframe[f'{col_prefix}is_whale_sell'],
            -1,
            0
        )
    )

    return dataframe


def get_whale_spike_value(dataframe: DataFrame, col_prefix: str = '') -> pd.Series:
    """
    Utility function to get the wave value only on whale spike bars.

    Useful for plotting or strategy logic where you only want the spike magnitude.

    Returns:
    --------
    Series with wave value on whale bars, NaN otherwise
    """
    wave_col = f'{col_prefix}liq_wave'
    buy_col = f'{col_prefix}is_whale_buy'
    sell_col = f'{col_prefix}is_whale_sell'

    return np.where(
        dataframe[buy_col] | dataframe[sell_col],
        dataframe[wave_col],
        np.nan
    )
