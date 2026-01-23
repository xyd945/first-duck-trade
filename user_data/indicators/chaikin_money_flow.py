"""
Chaikin Money Flow (CMF) Indicator

Converted from TradingView Pine Script to Python/Pandas for Freqtrade.

Original Concept:
- Measures the amount of Money Flow Volume over a specific period.
- CMF Sum(AD, n) / Sum(Vol, n)
- Where AD = ((2*Close - Low - High) / (High - Low)) * Volume

Outputs:
- cmf: Chaikin Money Flow values

Usage in Strategy:
    from indicators.chaikin_money_flow import add_chaikin_money_flow
    
    def populate_indicators(self, dataframe, metadata):
        dataframe = add_chaikin_money_flow(dataframe, length=20)
        return dataframe
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

def add_chaikin_money_flow(
    dataframe: DataFrame,
    length: int = 20,
    col_name: str = 'cmf'
) -> DataFrame:
    """
    Add Chaikin Money Flow (CMF) to the dataframe.

    Parameters:
    -----------
    dataframe : DataFrame
        OHLCV dataframe with 'close', 'high', 'low', 'volume' columns
    length : int, default=20
        The period for the rolling sum
    col_name : str, default='cmf'
        The name of the output column

    Returns:
    --------
    DataFrame
        Original dataframe with added 'cmf' column
    """
    
    # 1. Calculate Money Flow Multiplier (AD term without volume)
    # Formula: ((2*Close - Low - High) / (High - Low))
    # Handle division by zero where High == Low
    
    high_low_diff = dataframe['high'] - dataframe['low']
    
    # Avoid division by zero
    # If high == low, the multiplier is 0
    adj_close = (2 * dataframe['close'] - dataframe['low'] - dataframe['high'])
    
    mf_multiplier = np.where(
        high_low_diff == 0, 
        0, 
        adj_close / high_low_diff
    )
    
    # 2. Calculate Money Flow Volume
    mf_volume = mf_multiplier * dataframe['volume']
    
    # 3. Calculate CMF
    # CMF = Sum(MF Volume, n) / Sum(Volume, n)
    
    rolling_mf_volume = dataframe['volume'].copy() # Placeholder to match index type if needed, but we'll overwrite
    
    # Use pandas rolling sum
    # rolling() requires a Series or DataFrame
    mf_volume_series = pd.Series(mf_volume, index=dataframe.index)
    
    sum_mf_volume = mf_volume_series.rolling(window=length).sum()
    sum_volume = dataframe['volume'].rolling(window=length).sum()
    
    # Handle division by zero for the final calculation (if sum_volume is 0)
    dataframe[col_name] = np.where(
        sum_volume == 0,
        0,
        sum_mf_volume / sum_volume
    )
    
    return dataframe
