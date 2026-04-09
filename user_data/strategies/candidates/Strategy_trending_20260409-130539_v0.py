from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import pandas as pd
import pandas_ta as ta
import numpy as np
from base_generated import BaseGeneratedStrategy

class TrendMomentumBreakoutStrategy(BaseGeneratedStrategy):
    
    STRATEGY_THESIS = "Multi-timeframe trend following strategy using EMA crossovers, momentum indicators, and volume confirmation to capture strong trending moves in crypto futures"
    TARGET_REGIME = "trending"
    GENERATION_ID = "gen-20260409-130539-v0"
    
    timeframe = '1h'
    startup_candle_count = 250
    
    # Risk Management
    stoploss = -0.055
    
    minimal_roi = {
        "0": 0.08,
        "120": 0.05,
        "240": 0.03,
        "480": 0.015,
        "720": 0.0
    }
    
    # Hyperopt Parameters
    ema_fast_period = IntParameter(8, 21, default=13, space='buy')
    ema_slow_period = IntParameter(34, 89, default=55, space='buy')
    rsi_period = IntParameter(10, 21, default=14, space='buy')
    rsi_entry_threshold = IntParameter(45, 65, default=55, space='buy')
    rsi_exit_threshold = IntParameter(70, 85, default=75, space='sell')
    atr_period = IntParameter(10, 20, default=14, space='buy')
    volume_ma_period = IntParameter(15, 30, default=20, space='buy')
    adx_period = IntParameter(12, 20, default=14, space='buy')
    adx_threshold = DecimalParameter(20.0, 35.0, decimals=1, default=25.0, space='buy')
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # EMAs for trend identification
        dataframe['ema_fast'] = ta.ema(dataframe['close'], length=self.ema_fast_period.value)
        dataframe['ema_slow'] = ta.ema(dataframe['close'], length=self.ema_slow_period.value)
        dataframe['ema_200'] = ta.ema(dataframe['close'], length=200)
        
        # RSI for momentum
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=self.rsi_period.value)
        
        # ADX for trend strength
        adx_data = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=self.adx_period.value)
        dataframe['adx'] = adx_data['ADX_' + str(self.adx_period.value)]
        dataframe['di_plus'] = adx_data['DMP_' + str(self.adx_period.value)]
        dataframe['di_minus'] = adx_data['DMN_' + str(self.adx_period.value)]
        
        # ATR for volatility
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period.value)
        
        # Volume analysis
        dataframe['volume_ma'] = ta.sma(dataframe['volume'], length=self.volume_ma_period.value)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # MACD for momentum confirmation
        macd_data = ta.macd(dataframe['close'], fast=12, slow=26, signal=9)
        dataframe['macd'] = macd_data['MACD_12_26_9']
        dataframe['macdsignal'] = macd_data['MACDs_12_26_9']
        dataframe['macdhist'] = macd_data['MACDh_12_26_9']
        
        # Bollinger Bands for volatility and mean reversion signals
        bb_data = ta.bbands(dataframe['close'], length=20, std=2)
        dataframe['bb_upper'] = bb_data['BBU_20_2.0']
        dataframe['bb_middle'] = bb_data['BBM_20_2.0']
        dataframe['bb_lower'] = bb_data['BBL_20_2.0']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # Price momentum
        dataframe['price_change_5'] = dataframe['close'].pct_change(5)
        dataframe['price_change_10'] = dataframe['close'].pct_change(10)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = []
        
        # Condition 1: EMA trend alignment (fast > slow > 200)
        conditions.append(
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_slow'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_fast'])
        )
        
        # Condition 2: Strong trend with ADX and directional movement
        conditions.append(
            (dataframe['adx'] > self.adx_threshold.value) &
            (dataframe['di_plus'] > dataframe['di_minus'])
        )
        
        # Condition 3: Momentum confirmation with RSI and MACD
        conditions.append(
            (dataframe['rsi'] > self.rsi_entry_threshold.value) &
            (dataframe['rsi'] < 80) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1))
        )
        
        # Condition 4: Volume confirmation
        conditions.append(
            (dataframe['volume_ratio'] > 1.1) &
            (dataframe['volume'] > dataframe['volume'].shift(1))
        )
        
        # Condition 5: Price momentum and volatility
        conditions.append(
            (dataframe['price_change_5'] > 0.01) &
            (dataframe['bb_width'] > dataframe['bb_width'].rolling(20).mean()) &
            (dataframe['close'] > dataframe['bb_middle'])
        )
        
        # All conditions must be met for entry
        dataframe.loc[
            (conditions[0] & conditions[1] & conditions[2] & conditions[3] & conditions[4]),
            'enter_long'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = []
        
        # Condition 1: EMA crossover reversal
        conditions.append(
            (dataframe['ema_fast'] <= dataframe['ema_slow']) |
            (dataframe['close'] < dataframe['ema_slow'])
        )
        
        # Condition 2: RSI overbought
        conditions.append(
            (dataframe['rsi'] > self.rsi_exit_threshold.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1))
        )
        
        # Condition 3: MACD bearish divergence
        conditions.append(
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macdhist'] < 0)
        )
        
        # Condition 4: Weakening trend strength
        conditions.append(
            (dataframe['adx'] < dataframe['adx'].shift(1)) &
            (dataframe['di_minus'] > dataframe['di_plus'])
        )
        
        # Exit if any condition is met
        dataframe.loc[
            (conditions[0] | conditions[1] | conditions[2] | conditions[3]),
            'exit_long'
        ] = 1
        
        return dataframe