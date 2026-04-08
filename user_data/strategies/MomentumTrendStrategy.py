"""
MomentumTrendStrategy — EMA crossover trend-following strategy.

Designed to complement LiquiditySweepStrategy (which works in ranging markets)
by capturing directional moves in trending markets.

Entry (Long):
  1. EMA 20 crosses above EMA 50 (golden cross)
  2. ADX > 25 (confirming trend strength)
  3. Close above EMA 200 (higher timeframe trend alignment)
  4. Volume above average (institutional participation)

Exit (Long):
  1. EMA 20 crosses below EMA 50 (death cross)
  2. RSI > 80 (overbought)
  3. ATR-based trailing stop

Custom Stoploss:
  Dynamic stoploss at 2x ATR below entry price.
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from pandas import DataFrame
import pandas_ta as ta
import numpy as np


class MomentumTrendStrategy(IStrategy):
    """
    Trend-following strategy using EMA crossovers with ADX confirmation.
    Best in trending/breakout regimes. Complements LiquiditySweepStrategy.
    """

    INTERFACE_VERSION = 3

    # ==========================================================================
    # Strategy Settings
    # ==========================================================================
    minimal_roi = {
        "0": 0.20,    # 20% at any time
        "120": 0.10,  # 10% after 2 hours
        "360": 0.05,  # 5% after 6 hours
    }

    stoploss = -0.06  # -6% fallback
    use_custom_stoploss = True
    timeframe = '1h'
    startup_candle_count = 250

    # ==========================================================================
    # HYPEROPT PARAMETERS
    # ==========================================================================

    # EMA periods
    ema_fast = IntParameter(10, 30, default=20, space='buy', optimize=True)
    ema_slow = IntParameter(40, 80, default=50, space='buy', optimize=True)
    ema_trend = IntParameter(150, 250, default=200, space='buy', optimize=True)

    # ADX filter
    adx_threshold = IntParameter(20, 35, default=25, space='buy', optimize=True)

    # Volume filter
    use_vol_filter = BooleanParameter(default=True, space='buy', optimize=True)
    vol_ma_length = IntParameter(10, 50, default=20, space='buy', optimize=True)
    vol_multiplier = DecimalParameter(1.0, 2.5, default=1.2, decimals=1, space='buy', optimize=True)

    # ATR for stoploss
    atr_length = IntParameter(10, 30, default=14, space='buy', optimize=True)
    atr_multiplier = DecimalParameter(1.5, 3.5, default=2.0, decimals=1, space='buy', optimize=True)

    # RSI exit
    rsi_exit_threshold = IntParameter(70, 90, default=80, space='sell', optimize=True)

    # ==========================================================================
    # INDICATORS
    # ==========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # EMAs
        dataframe['ema_fast'] = ta.ema(dataframe['close'], length=self.ema_fast.value)
        dataframe['ema_slow'] = ta.ema(dataframe['close'], length=self.ema_slow.value)
        dataframe['ema_trend'] = ta.ema(dataframe['close'], length=self.ema_trend.value)

        # ADX
        adx_result = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        dataframe['adx'] = adx_result['ADX_14']

        # RSI
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)

        # ATR for dynamic stoploss
        dataframe['atr'] = ta.atr(
            dataframe['high'], dataframe['low'], dataframe['close'],
            length=self.atr_length.value
        )

        # Volume MA
        dataframe['vol_ma'] = dataframe['volume'].rolling(
            window=self.vol_ma_length.value
        ).mean()

        # EMA crossover signals (shifted by 1 to detect the cross candle)
        dataframe['ema_cross_up'] = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_fast'].shift(1) <= dataframe['ema_slow'].shift(1))
        )
        dataframe['ema_cross_down'] = (
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['ema_fast'].shift(1) >= dataframe['ema_slow'].shift(1))
        )

        return dataframe

    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = (
            # EMA golden cross occurred within last 3 candles
            (dataframe['ema_cross_up'] |
             dataframe['ema_cross_up'].shift(1) |
             dataframe['ema_cross_up'].shift(2)) &

            # ADX confirms trend strength
            (dataframe['adx'] > self.adx_threshold.value) &

            # Price above long-term trend
            (dataframe['close'] > dataframe['ema_trend']) &

            # Volume filter
            (
                ~self.use_vol_filter.value |
                (dataframe['volume'] > dataframe['vol_ma'] * self.vol_multiplier.value)
            ) &

            # Safety check
            (dataframe['volume'] > 0)
        )

        dataframe.loc[conditions, 'enter_long'] = 1
        return dataframe

    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMA death cross
        death_cross = dataframe['ema_cross_down']

        # RSI overbought
        rsi_exit = dataframe['rsi'] > self.rsi_exit_threshold.value

        dataframe.loc[
            (death_cross | rsi_exit) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1

        return dataframe

    # ==========================================================================
    # CUSTOM STOPLOSS (ATR-based)
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
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe.empty:
            return -1

        # Get ATR at entry time
        entry_candle = dataframe.loc[dataframe['date'] <= trade.open_date_utc]
        if entry_candle.empty:
            return -1

        atr_value = entry_candle.iloc[-1]['atr']
        if pd.isna(atr_value) or atr_value <= 0:
            return -1

        # Stop at entry_price - (ATR * multiplier)
        stop_price = trade.open_rate - (atr_value * self.atr_multiplier.value)
        stoploss_ratio = (stop_price - trade.open_rate) / trade.open_rate

        if stoploss_ratio >= 0:
            return -1

        return stoploss_ratio
