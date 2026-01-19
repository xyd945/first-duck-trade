"""
MyFirstStrategy - A simple RSI-based buy/sell strategy.

Entry: RSI < 30 (oversold)
Exit: RSI > 70 (overbought)
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta


class MyFirstStrategy(IStrategy):
    """
    A simple mean-reversion strategy based on RSI.
    Buys when RSI dips below 30 (oversold) and sells when RSI rises above 70 (overbought).
    """

    INTERFACE_VERSION = 3

    # Minimal ROI (Take Profit targets)
    minimal_roi = {
        "0": 0.20,   # 20% profit at any time
        "30": 0.05,  # 5% profit after 30 minutes
        "60": 0.01,  # 1% profit after 60 minutes
    }

    # Stoploss (Fixed)
    stoploss = -0.10  # -10%

    # Timeframe
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate technical indicators and append them to the DataFrame.
        """
        # Calculate RSI with a 14-period lookback
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals (buy conditions).
        Vectorized logic sets 'enter_long' column to 1 when conditions are met.
        """
        dataframe.loc[
            (
                # Signal: RSI is below 30 (oversold)
                (dataframe['rsi'] < 30) &
                # Safety: Volume must exist
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals (sell conditions).
        Vectorized logic sets 'exit_long' column to 1 when conditions are met.
        """
        dataframe.loc[
            (
                # Signal: RSI is above 70 (overbought)
                (dataframe['rsi'] > 70) &
                # Safety: Volume must exist
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
