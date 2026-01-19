"""
MyFirstStrategy - A simple RSI-based buy/sell strategy with Hyperopt support.

Entry: RSI < rsi_entry threshold (oversold)
Exit: RSI > rsi_exit threshold (overbought)
"""

from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import pandas_ta as ta


class MyFirstStrategy(IStrategy):
    """
    A simple mean-reversion strategy based on RSI.
    Buys when RSI dips below threshold (oversold) and sells when RSI rises above threshold (overbought).
    
    Hyperopt-optimizable parameters:
    - rsi_length: RSI calculation period
    - rsi_entry: RSI threshold for entry (buy when RSI < this)
    - rsi_exit: RSI threshold for exit (sell when RSI > this)
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

    # Startup candle count - ensures indicators have enough data before trading
    startup_candle_count = 30

    # ========== HYPEROPT PARAMETERS ==========
    # RSI period: search between 7 and 21 (default 14)
    rsi_length = IntParameter(7, 21, default=14, space="buy", optimize=True)

    # RSI entry threshold: search between 20 and 40 (default 30)
    # Buy when RSI drops below this value
    rsi_entry = IntParameter(20, 40, default=30, space="buy", optimize=True)

    # RSI exit threshold: search between 60 and 80 (default 70)
    # Sell when RSI rises above this value
    rsi_exit = IntParameter(60, 80, default=70, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate technical indicators and append them to the DataFrame.
        """
        # Calculate RSI with hyperopt-optimizable period
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=self.rsi_length.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals (buy conditions).
        Vectorized logic sets 'enter_long' column to 1 when conditions are met.
        """
        dataframe.loc[
            (
                # Signal: RSI is below entry threshold (oversold)
                (dataframe['rsi'] < self.rsi_entry.value) &
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
                # Signal: RSI is above exit threshold (overbought)
                (dataframe['rsi'] > self.rsi_exit.value) &
                # Safety: Volume must exist
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
