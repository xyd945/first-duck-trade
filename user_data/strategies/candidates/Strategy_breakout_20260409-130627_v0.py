from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import pandas as pd
import pandas_ta as ta
import numpy as np
from base_generated import BaseGeneratedStrategy

class BreakoutMomentumStrategy(BaseGeneratedStrategy):
    
    STRATEGY_THESIS = "Breakout strategy that identifies strong momentum moves above dynamic resistance levels with volume confirmation and ADX trend strength filtering"
    TARGET_REGIME = "breakout"
    GENERATION_ID = "gen-20260409-130627-v0"
    
    timeframe = '1h'
    startup_candle_count = 250
    can_short = True
    
    # Risk Management
    stoploss = -0.055
    minimal_roi = {
        "0": 0.15,
        "30": 0.08,
        "60": 0.04,
        "120": 0.02,
        "240": 0.01,
        "480": 0
    }
    
    # Hyperopt Parameters
    donchian_length = IntParameter(15, 35, default=20, space='buy')
    adx_period = IntParameter(12, 20, default=14, space='buy')
    adx_threshold = DecimalParameter(20.0, 35.0, default=25.0, space='buy')
    rsi_period = IntParameter(12, 18, default=14, space='buy')
    rsi_breakout_threshold = DecimalParameter(55.0, 70.0, default=60.0, space='buy')
    volume_ma_period = IntParameter(18, 30, default=20, space='buy')
    volume_multiplier = DecimalParameter(1.3, 2.2, default=1.5, space='buy')
    
    # Exit parameters
    exit_rsi_high = DecimalParameter(75.0, 85.0, default=80.0, space='sell')
    exit_adx_low = DecimalParameter(15.0, 25.0, default=20.0, space='sell')
    trailing_stop = DecimalParameter(0.02, 0.06, default=0.035, space='sell')
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Donchian Channels for breakout detection
        donchian = ta.donchian(dataframe['high'], dataframe['low'], length=self.donchian_length.value)
        dataframe['dc_upper'] = donchian['DCU_' + str(self.donchian_length.value)]
        dataframe['dc_lower'] = donchian['DCL_' + str(self.donchian_length.value)]
        dataframe['dc_middle'] = donchian['DCM_' + str(self.donchian_length.value)]
        
        # ADX for trend strength
        adx_data = ta.adx(dataframe['high'], dataframe['low'], dataframe['close'], length=self.adx_period.value)
        dataframe['adx'] = adx_data['ADX_' + str(self.adx_period.value)]
        dataframe['di_plus'] = adx_data['DMP_' + str(self.adx_period.value)]
        dataframe['di_minus'] = adx_data['DMN_' + str(self.adx_period.value)]
        
        # RSI for momentum confirmation
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=self.rsi_period.value)
        
        # Volume indicators
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=self.volume_ma_period.value).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # EMA for trend direction
        dataframe['ema_21'] = ta.ema(dataframe['close'], length=21)
        dataframe['ema_50'] = ta.ema(dataframe['close'], length=50)
        
        # ATR for volatility
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        
        # Price momentum
        dataframe['price_change_pct'] = dataframe['close'].pct_change(periods=3) * 100
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                # Condition 1: Breakout above Donchian upper channel
                (dataframe['close'] > dataframe['dc_upper'].shift(1)) &
                (dataframe['close'].shift(1) <= dataframe['dc_upper'].shift(2)) &
                
                # Condition 2: Strong trend strength with ADX
                (dataframe['adx'] > self.adx_threshold.value) &
                (dataframe['di_plus'] > dataframe['di_minus']) &
                
                # Condition 3: RSI momentum confirmation
                (dataframe['rsi'] > self.rsi_breakout_threshold.value) &
                (dataframe['rsi'].shift(1) <= self.rsi_breakout_threshold.value) &
                
                # Condition 4: Volume surge confirmation
                (dataframe['volume_ratio'] > self.volume_multiplier.value) &
                
                # Condition 5: Price above key EMAs for trend alignment
                (dataframe['close'] > dataframe['ema_21']) &
                (dataframe['ema_21'] > dataframe['ema_50']) &
                
                # Additional filter: positive price momentum
                (dataframe['price_change_pct'] > 1.0) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                # Short conditions (inverse breakout logic)
                # Condition 1: Breakout below Donchian lower channel
                (dataframe['close'] < dataframe['dc_lower'].shift(1)) &
                (dataframe['close'].shift(1) >= dataframe['dc_lower'].shift(2)) &
                
                # Condition 2: Strong trend strength with ADX (bearish)
                (dataframe['adx'] > self.adx_threshold.value) &
                (dataframe['di_minus'] > dataframe['di_plus']) &
                
                # Condition 3: RSI momentum confirmation (oversold bounce down)
                (dataframe['rsi'] < (100 - self.rsi_breakout_threshold.value)) &
                (dataframe['rsi'].shift(1) >= (100 - self.rsi_breakout_threshold.value)) &
                
                # Condition 4: Volume surge confirmation
                (dataframe['volume_ratio'] > self.volume_multiplier.value) &
                
                # Condition 5: Price below key EMAs for bearish trend
                (dataframe['close'] < dataframe['ema_21']) &
                (dataframe['ema_21'] < dataframe['ema_50']) &
                
                # Additional filter: negative price momentum
                (dataframe['price_change_pct'] < -1.0) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                # Exit long conditions
                # Condition 1: RSI extremely overbought
                (dataframe['rsi'] > self.exit_rsi_high.value) |
                
                # Condition 2: ADX weakening (trend losing strength)
                (dataframe['adx'] < self.exit_adx_low.value) |
                
                # Condition 3: Price falling back into Donchian channel
                (dataframe['close'] < dataframe['dc_middle']) |
                
                # Condition 4: Bearish crossover of DI lines
                ((dataframe['di_minus'] > dataframe['di_plus']) & 
                 (dataframe['di_minus'].shift(1) <= dataframe['di_plus'].shift(1)))
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                # Exit short conditions
                # Condition 1: RSI extremely oversold
                (dataframe['rsi'] < (100 - self.exit_rsi_high.value)) |
                
                # Condition 2: ADX weakening (trend losing strength)
                (dataframe['adx'] < self.exit_adx_low.value) |
                
                # Condition 3: Price rising back into Donchian channel
                (dataframe['close'] > dataframe['dc_middle']) |
                
                # Condition 4: Bullish crossover of DI lines
                ((dataframe['di_plus'] > dataframe['di_minus']) & 
                 (dataframe['di_plus'].shift(1) <= dataframe['di_minus'].shift(1)))
            ),
            'exit_short'] = 1
        
        return dataframe