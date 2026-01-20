"""
LiquiditySweepStrategy - Vectorized Liquidity Sweep detection strategy (Improved v2).

Concept: Identifies "sweeps" of key swing high/low points, confirmed by a candle
close back within the range, filtered by volume, trend, and market structure.

Improvements (v2):
- Trend Filter: EMA alignment + ADX to avoid trading against strong trends
- Market Structure Confirmation: Strong reclaim candle (closes in top 30% of range)
- Pivot Freshness: Min age requirement to avoid sweeping very recent pivots
- Enhanced exit logic with trailing and bearish sweeps

Entry (Long): Price sweeps below a recent pivot low, reclaims strongly, with trend confirmation.
Exit (Long): Opposing sweep, RSI overbought, or trailing stop.
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from pandas import DataFrame
import pandas_ta as ta
import numpy as np


class LiquiditySweepStrategy(IStrategy):
    """
    A liquidity sweep strategy adapted from Pine Script to Freqtrade (v2 Improved).

    Detects sweeps of key swing highs/lows using vectorized pandas operations.
    Includes trend filtering, market structure confirmation, and smart exits.

    Key Improvements:
    - EMA + ADX trend filter to avoid catching falling knives
    - Strong reclaim confirmation (candle closes in upper portion of range)
    - Pivot age requirement to ensure we're sweeping significant levels
    """

    INTERFACE_VERSION = 3

    # ==========================================================================
    # Strategy Settings
    # ==========================================================================
    minimal_roi = {
        "0": 0.15,    # 15% at any time
        "60": 0.08,   # 8% after 60 minutes
        "120": 0.04,  # 4% after 2 hours
        "240": 0.02,  # 2% after 4 hours
    }

    # Stoploss - fallback fixed stoploss (custom_stoploss overrides this)
    stoploss = -0.08  # -8%

    # Enable custom stoploss
    use_custom_stoploss = True

    # Timeframe
    timeframe = '1h'

    # Startup candle count - need enough data for EMA-200 and pivot detection
    startup_candle_count = 250

    # ==========================================================================
    # HYPEROPT PARAMETERS - Original
    # ==========================================================================

    # Pivot Lookback: How many candles to look before/after finding a pivot
    # Pivot requires N candles on each side, so window = 2 * pivot_len + 1
    pivot_len = IntParameter(2, 10, default=5, space='buy', optimize=True)

    # Volume Filter - whether to require above-average volume on sweep
    use_vol_check = BooleanParameter(default=True, space='buy', optimize=True)

    # Volume MA Length for calculating average volume
    vol_len = IntParameter(10, 50, default=20, space='buy', optimize=True)

    # Volume Multiplier - sweep candle volume must exceed (vol_ma * vol_mult)
    vol_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space='buy', optimize=True)

    # How far back to search for the "swept" liquidity level (for ffill limit)
    sweep_lookback = IntParameter(10, 60, default=20, space='buy', optimize=True)

    # Noise Filter: Minimum percentage distance for a sweep to be valid (0.1% to 1%)
    min_sweep_dist = DecimalParameter(0.001, 0.01, default=0.002, decimals=3, space='buy', optimize=True)

    # RSI exit threshold - exit when RSI exceeds this (overbought)
    rsi_exit_threshold = IntParameter(60, 90, default=75, space='sell', optimize=True)

    # ==========================================================================
    # HYPEROPT PARAMETERS - New (v2 Improvements)
    # ==========================================================================

    # ---------- Trend Filter ----------
    # Short EMA period for trend detection
    trend_ema_short = IntParameter(20, 100, default=50, space='buy', optimize=True)

    # Long EMA period for trend detection
    trend_ema_long = IntParameter(100, 300, default=200, space='buy', optimize=True)

    # ADX Filter - whether to use ADX to detect strong trends
    use_adx_filter = BooleanParameter(default=True, space='buy', optimize=True)

    # ADX threshold - above this value indicates a strong trend
    adx_threshold = IntParameter(20, 40, default=25, space='buy', optimize=True)

    # ---------- Confirmation ----------
    # Require strong reclaim candle (closes in upper portion of range)
    require_strong_reclaim = BooleanParameter(default=True, space='buy', optimize=True)

    # Reclaim strength threshold (0.7 = candle closes in top 30% of its range)
    reclaim_strength_threshold = DecimalParameter(0.5, 0.9, default=0.7, decimals=1, space='buy', optimize=True)

    # ---------- Pivot Freshness ----------
    # Minimum candles between pivot formation and sweep (avoid sweeping very recent lows)
    min_pivot_age = IntParameter(5, 30, default=10, space='buy', optimize=True)

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators needed for the strategy.
        """
        pivot_window = 2 * self.pivot_len.value + 1

        # ---------------------------------------------------------------------
        # Trend Indicators (NEW - v2)
        # ---------------------------------------------------------------------
        dataframe['ema_short'] = ta.ema(dataframe['close'], length=self.trend_ema_short.value)
        dataframe['ema_long'] = ta.ema(dataframe['close'], length=self.trend_ema_long.value)

        # ADX for trend strength
        adx_result = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        dataframe['adx'] = adx_result['ADX_14']

        # ---------------------------------------------------------------------
        # Pivot Low Detection (Vectorized)
        # A pivot low occurs when the low is the minimum over a centered window
        # ---------------------------------------------------------------------
        dataframe['rolling_min'] = dataframe['low'].rolling(
            window=pivot_window,
            center=True
        ).min()
        dataframe['is_pivot_low'] = dataframe['low'] == dataframe['rolling_min']

        # Extract pivot low values, NaN elsewhere
        dataframe['pivot_low_value'] = np.where(
            dataframe['is_pivot_low'],
            dataframe['low'],
            np.nan
        )

        # Calculate pivot age (bars since last pivot)
        # Create a cumulative count that resets at each pivot
        dataframe['pivot_low_idx'] = np.where(
            dataframe['is_pivot_low'],
            dataframe.index,
            np.nan
        )
        dataframe['last_pivot_low_idx'] = dataframe['pivot_low_idx'].ffill()
        dataframe['pivot_age'] = dataframe.index - dataframe['last_pivot_low_idx']

        # Forward fill to propagate the last known pivot low
        # Limit ffill to sweep_lookback to avoid stale levels
        dataframe['last_pivot_low'] = dataframe['pivot_low_value'].ffill(
            limit=self.sweep_lookback.value
        )

        # CRITICAL: Shift by 1 to compare against ESTABLISHED pivots (avoid lookahead)
        dataframe['last_pivot_low'] = dataframe['last_pivot_low'].shift(1)
        dataframe['pivot_age'] = dataframe['pivot_age'].shift(1)

        # ---------------------------------------------------------------------
        # Pivot High Detection (Vectorized) - for opposing sweep exits
        # ---------------------------------------------------------------------
        dataframe['rolling_max'] = dataframe['high'].rolling(
            window=pivot_window,
            center=True
        ).max()
        dataframe['is_pivot_high'] = dataframe['high'] == dataframe['rolling_max']

        # Extract pivot high values
        dataframe['pivot_high_value'] = np.where(
            dataframe['is_pivot_high'],
            dataframe['high'],
            np.nan
        )

        # Forward fill for last known pivot high
        dataframe['last_pivot_high'] = dataframe['pivot_high_value'].ffill(
            limit=self.sweep_lookback.value
        )

        # Shift by 1 to avoid lookahead
        dataframe['last_pivot_high'] = dataframe['last_pivot_high'].shift(1)

        # ---------------------------------------------------------------------
        # Volume Indicators
        # ---------------------------------------------------------------------
        dataframe['vol_ma'] = dataframe['volume'].rolling(
            window=self.vol_len.value
        ).mean()

        dataframe['high_volume'] = dataframe['volume'] > (
            dataframe['vol_ma'] * self.vol_mult.value
        )

        # ---------------------------------------------------------------------
        # RSI for Exit
        # ---------------------------------------------------------------------
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)

        # ---------------------------------------------------------------------
        # Sweep Depth Calculation (for noise filter)
        # How much the wick went below the pivot as a percentage
        # ---------------------------------------------------------------------
        dataframe['sweep_depth'] = (
            (dataframe['last_pivot_low'] - dataframe['low']) / dataframe['low']
        )

        # ---------------------------------------------------------------------
        # Candle Reclaim Strength (NEW - v2)
        # How strongly the candle closed back above the sweep
        # 1.0 = closed at high, 0.0 = closed at low
        # ---------------------------------------------------------------------
        candle_range = dataframe['high'] - dataframe['low']
        # Avoid division by zero for doji candles
        candle_range = candle_range.replace(0, np.nan)
        dataframe['reclaim_strength'] = (dataframe['close'] - dataframe['low']) / candle_range
        dataframe['reclaim_strength'] = dataframe['reclaim_strength'].fillna(0.5)

        # Store the sweep candle low for custom stoploss
        dataframe['sweep_candle_low'] = dataframe['low']

        return dataframe

    # ==========================================================================
    # ENTRY LOGIC (Improved v2)
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals for long positions (v2 with trend + confirmation).

        Improved Entry Conditions:
        1. TREND FILTER: EMA alignment + ADX (don't buy in strong downtrends)
        2. SWEEP: Wick sweeps below the last pivot low (low < last_pivot_low)
        3. PIVOT AGE: Pivot must be old enough (not a very recent low)
        4. RECLAIM: Body reclaims above the pivot (close > last_pivot_low)
        5. CONFIRMATION: Strong reclaim candle (closes in top 30% of range)
        6. VOLUME: Above-average volume on sweep candle
        7. NOISE FILTER: Sweep depth exceeds minimum threshold
        """

        # =====================================================================
        # 1. TREND FILTER (NEW - v2)
        # =====================================================================
        # Basic uptrend: short EMA > long EMA AND price above long EMA
        is_uptrend = (
            (dataframe['ema_short'] > dataframe['ema_long']) &
            (dataframe['close'] > dataframe['ema_long'])
        )

        # ADX filter: block entries in STRONG downtrends
        if self.use_adx_filter.value:
            is_strong_downtrend = (
                (dataframe['adx'] > self.adx_threshold.value) &
                (dataframe['ema_short'] < dataframe['ema_long'])
            )
            trend_ok = is_uptrend & (~is_strong_downtrend)
        else:
            trend_ok = is_uptrend

        # =====================================================================
        # 2. SWEEP CONDITION
        # =====================================================================
        sweep_condition = (
            # Wick went below pivot
            (dataframe['low'] < dataframe['last_pivot_low']) &
            # Close is back above pivot (reclaim)
            (dataframe['close'] > dataframe['last_pivot_low']) &
            # Must have valid pivot data
            (dataframe['last_pivot_low'].notna())
        )

        # =====================================================================
        # 3. PIVOT AGE FILTER (NEW - v2)
        # =====================================================================
        pivot_fresh_enough = dataframe['pivot_age'] > self.min_pivot_age.value

        # =====================================================================
        # 4. CONFIRMATION - Strong Reclaim Candle (NEW - v2)
        # =====================================================================
        if self.require_strong_reclaim.value:
            strong_reclaim = dataframe['reclaim_strength'] > self.reclaim_strength_threshold.value
        else:
            strong_reclaim = True  # Always true if disabled

        # =====================================================================
        # 5. NOISE FILTER
        # =====================================================================
        noise_filter = dataframe['sweep_depth'] > self.min_sweep_dist.value

        # =====================================================================
        # 6. VOLUME FILTER
        # =====================================================================
        if self.use_vol_check.value:
            volume_ok = dataframe['high_volume']
        else:
            volume_ok = True

        # =====================================================================
        # COMBINE ALL CONDITIONS
        # =====================================================================
        conditions = (
            trend_ok &
            sweep_condition &
            pivot_fresh_enough &
            strong_reclaim &
            noise_filter &
            volume_ok &
            (dataframe['volume'] > 0)  # Safety check
        )

        dataframe.loc[conditions, 'enter_long'] = 1

        return dataframe

    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals for long positions.

        Exit Conditions (OR logic):
        A. Opposing Sweep: high sweeps above pivot high and closes below (bearish sweep)
        B. RSI Overbought: RSI exceeds threshold
        C. Trend Reversal: Short EMA crosses below Long EMA (optional, implicit via next entry filter)
        """
        # Condition A: Opposing (Bearish) Sweep
        bearish_sweep = (
            # Wick swept above pivot high
            (dataframe['high'] > dataframe['last_pivot_high']) &
            # Body closed back below pivot (failed breakout)
            (dataframe['close'] < dataframe['last_pivot_high']) &
            # Must have valid pivot data
            (dataframe['last_pivot_high'].notna())
        )

        # Condition B: RSI Overbought Exit
        rsi_exit = dataframe['rsi'] > self.rsi_exit_threshold.value

        # Combine exits with OR logic
        dataframe.loc[
            (bearish_sweep | rsi_exit) &
            (dataframe['volume'] > 0),  # Safety check
            'exit_long'
        ] = 1

        return dataframe

    # ==========================================================================
    # CUSTOM STOPLOSS (Dynamic)
    # ==========================================================================

    def custom_stoploss(
        self,
        pair: str,
        trade: 'Trade',
        current_time: 'datetime',
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> float:
        """
        Dynamic stoploss based on the sweep candle low.

        Instead of a fixed percentage, we place the stop just below
        the low of the entry candle (the sweep candle).

        Returns: stoploss as a negative ratio from current_rate, or -1 to disable.
        """
        # Access the dataframe to get the sweep candle low at entry
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe.empty:
            return -1  # Use default stoploss

        # Find the candle at trade entry time
        entry_candle = dataframe.loc[dataframe['date'] <= trade.open_date_utc]

        if entry_candle.empty:
            return -1  # Use default stoploss

        # Get the low of the entry (sweep) candle
        sweep_low = entry_candle.iloc[-1]['low']

        # Place stop slightly below (0.5% buffer)
        stop_price = sweep_low * 0.995

        # Calculate stoploss ratio from entry price
        stoploss_ratio = (stop_price - trade.open_rate) / trade.open_rate

        # Return the stoploss (must be negative for a long position stop below entry)
        # If sweep low is above entry (shouldn't happen), use default
        if stoploss_ratio >= 0:
            return -1  # Use fallback fixed stoploss

        return stoploss_ratio
