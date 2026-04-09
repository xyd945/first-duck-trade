from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas_ta as ta
import numpy as np
from base_generated import BaseGeneratedStrategy

class RangingBounceStrategy(BaseGeneratedStrategy):
    STRATEGY_THESIS = "Trades range-bound markets by identifying support/resistance bounces using Bollinger Bands mean reversion, RSI oversold/overbought levels, and volume confirmation for entries/exits"
    TARGET_REGIME = "ranging"
    GENERATION_ID = "gen-20260409-130604-v0"
    
    timeframe = '1h'
    startup_candle_count = 250
    
    # Hyperopt parameters
    bb_period = IntParameter(15, 25, default=20, space="buy")
    bb_std = DecimalParameter(1.8, 2.5, default=2.0, space="buy")
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_oversold = IntParameter(25, 35, default=30, space="buy")
    rsi_overbought = IntParameter(65, 75, default=70, space="sell")
    volume_sma_period = IntParameter(15, 30, default=20, space="buy")
    volume_threshold = DecimalParameter(1.1, 1.8, default=1.3, space="buy")
    cci_period = IntParameter(15, 25, default=20, space="buy")
    cci_oversold = IntParameter(-120, -80, default=-100, space="buy")
    cci_overbought = IntParameter(80, 120, default=100, space="sell")
    
    stoploss = -0.055
    minimal_roi = {
        "0": 0.06,
        "30": 0.04,
        "60": 0.02,
        "120": 0.01,
        "180": 0
    }
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands for range identification
        bb = ta.bbands(dataframe['close'], length=self.bb_period.value, std=self.bb_std.value)
        dataframe['bb_upper'] = bb['BBU_' + str(self.bb_period.value) + '_' + str(self.bb_std.value)]
        dataframe['bb_middle'] = bb['BBM_' + str(self.bb_period.value) + '_' + str(self.bb_std.value)]
        dataframe['bb_lower'] = bb['BBL_' + str(self.bb_period.value) + '_' + str(self.bb_std.value)]
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # RSI for momentum
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=self.rsi_period.value)
        
        # CCI for additional momentum confirmation
        dataframe['cci'] = ta.cci(dataframe['high'], dataframe['low'], dataframe['close'], length=self.cci_period.value)
        
        # Volume indicators
        dataframe['volume_sma'] = ta.sma(dataframe['volume'], length=self.volume_sma_period.value)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # Price position within BB
        dataframe['bb_position'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # EMA for trend context
        dataframe['ema_50'] = ta.ema(dataframe['close'], length=50)
        dataframe['ema_100'] = ta.ema(dataframe['close'], length=100)
        
        # ATR for volatility
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Condition 1: Price near lower BB (oversold in range)
                (dataframe['close'].shift(1) <= dataframe['bb_lower'].shift(1) * 1.02) &
                (dataframe['bb_position'].shift(1) < 0.2) &
                
                # Condition 2: RSI oversold but not extremely oversold
                (dataframe['rsi'].shift(1) < self.rsi_oversold.value) &
                (dataframe['rsi'].shift(1) > 20) &
                
                # Condition 3: CCI confirmation of oversold
                (dataframe['cci'].shift(1) < self.cci_oversold.value) &
                
                # Condition 4: Volume above average (interest at support)
                (dataframe['volume_ratio'].shift(1) > self.volume_threshold.value) &
                
                # Condition 5: Ranging market (narrow BB width)
                (dataframe['bb_width'].shift(1) < dataframe['bb_width'].rolling(50).mean().shift(1) * 1.1)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                # Condition 1: Price near upper BB (overbought in range)
                (dataframe['close'].shift(1) >= dataframe['bb_upper'].shift(1) * 0.98) &
                (dataframe['bb_position'].shift(1) > 0.8) &
                
                # Condition 2: RSI overbought but not extremely overbought
                (dataframe['rsi'].shift(1) > self.rsi_overbought.value) &
                (dataframe['rsi'].shift(1) < 80) &
                
                # Condition 3: CCI confirmation of overbought
                (dataframe['cci'].shift(1) > self.cci_overbought.value) &
                
                # Condition 4: Volume above average (interest at resistance)
                (dataframe['volume_ratio'].shift(1) > self.volume_threshold.value) &
                
                # Condition 5: Ranging market (narrow BB width)
                (dataframe['bb_width'].shift(1) < dataframe['bb_width'].rolling(50).mean().shift(1) * 1.1)
            ),
            'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Condition 1: Price reaches middle BB (mean reversion target)
                (dataframe['close'].shift(1) >= dataframe['bb_middle'].shift(1)) &
                
                # Condition 2: RSI back to neutral/overbought
                (dataframe['rsi'].shift(1) > 50) &
                
                # Condition 3: BB position above midpoint
                (dataframe['bb_position'].shift(1) > 0.6) &
                
                # Condition 4: Volume declining (momentum fading)
                (dataframe['volume_ratio'].shift(1) < dataframe['volume_ratio'].shift(2))
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                # Condition 1: Price reaches middle BB (mean reversion target)
                (dataframe['close'].shift(1) <= dataframe['bb_middle'].shift(1)) &
                
                # Condition 2: RSI back to neutral/oversold
                (dataframe['rsi'].shift(1) < 50) &
                
                # Condition 3: BB position below midpoint
                (dataframe['bb_position'].shift(1) < 0.4) &
                
                # Condition 4: Volume declining (momentum fading)
                (dataframe['volume_ratio'].shift(1) < dataframe['volume_ratio'].shift(2))
            ),
            'exit_short'] = 1
        
        return dataframe