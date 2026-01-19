---
trigger: always_on
---

# Project Context: Freqtrade Crypto Bot

## 1. Core Architecture & Locations
- **Framework:** Freqtrade (Latest Stable).
- **Language:** Python 3.10+.
- **Data Structure:** Pandas DataFrame.
    - **Columns:** `date`, `open`, `high`, `low`, `close`, `volume`.
    - **Logic:** Vectorized (Column-based). **NEVER** Iterative (Row-based).
- **File Locations:**
    - Strategies: `user_data/strategies/`
    - Real Config: `user_data/config.json` (Do NOT edit directly without reference)
    - **Safe Config Ref:** `user_data/config_cheatsheet.json`

## 2. STRICT Prohibitions (Critical)
**Violating these rules will cause the bot to crash.**

1.  **NO `self.buy()` or `self.sell()`**: These methods do not exist in the strategy scope. You MUST set the `enter_long` or `exit_long` column to `1`.
2.  **NO `for` loops**: Do not iterate over dataframe rows. Use vectorized pandas operations.
    - *Wrong:* `if row['rsi'] < 30:`
    - *Right:* `dataframe.loc[dataframe['rsi'] < 30, 'enter_long'] = 1`
3.  **NO Index modification**: Never reset or drop the dataframe index.
4.  **NO Column Dropping**: Never use `df.drop()`. Freqtrade requires all original columns to persist.

## 3. Configuration Protocol
**When asked to update settings (whitelist, stoploss, stakes):**
1.  **READ** `user_data/config_cheatsheet.json` FIRST to see the allowed structure and valid field names.
2.  **APPLY** the changes to `user_data/config.json` matching the structure found in the cheatsheet.
3.  **DO NOT** invent new settings keys that are not present in the cheatsheet or official docs.

## 4. Coding & Library Standards
- **Primary Library:** `pandas_ta`.
    - Import as: `import pandas_ta as ta`
    - Usage: `dataframe.ta.rsi(length=14, append=True)`
    - **Note:** Do NOT use `talib` unless absolutely necessary.
- **Look-Ahead Bias (Crucial):**
    - Backtesting passes the *entire* timeframe at once.
    - You **MUST** access previous candles using `.shift()`.
    - *Example:* To check if the *previous* candle closed green: 
      `dataframe['close'].shift(1) > dataframe['open'].shift(1)`

## 5. Strategy Template
All strategy files must follow this class structure exactly:

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
import talib.abstract as tl  # Only if needed

class MyStrategy(IStrategy):
    INTERFACE_VERSION = 3
    
    # Minimal ROI (Target Profit)
    minimal_roi = {"0": 0.2, "30": 0.05, "60": 0.01}
    
    # Stoploss (Fixed)
    stoploss = -0.10
    
    # Timeframe
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Example: Calculate RSI and append to dataframe
        # Note: 'append=True' is often handled by pandas_ta automatically, 
        # but assignment is safer for clarity.
        dataframe['rsi'] = dataframe.ta.rsi(length=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &  # Signal
                (dataframe['volume'] > 0)  # Safety check
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 70),
            'exit_long'] = 1
        return dataframe