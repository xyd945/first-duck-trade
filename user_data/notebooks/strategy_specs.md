# Freqtrade Strategy Technical Specification

## 1. Overview
This document outlines the technical requirements and syntax for creating a custom trading strategy in Freqtrade. It serves as a strict guide for the "Builder" agent to ensure compliance with Freqtrade's architecture and performance standards.

## 2. Base Class Architecture
All strategies must inherit from `IStrategy`.

**Import Path:**
```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
import talib.abstract as tl
```

**Class Definition:**
```python
class MyCustomStrategy(IStrategy):
    INTERFACE_VERSION = 3
    # ... strategy settings ...
```

## 3. Core Strategy Methods
The strategy logic is divided into three mandatory methods. Each method operates on the full DataFrame and must return the modified DataFrame.

### 3.1. `populate_indicators`
Calculates technical indicators and appends them as new columns to the DataFrame.

**Syntax:**
```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Example: Calculate RSI using pandas-ta
    dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
    
    # Example: Calculate SMA
    dataframe['sma_200'] = ta.sma(dataframe['close'], length=200)
    
    return dataframe
```

### 3.2. `populate_entry_trend`
Defines entry signals (buying). Uses vectorized operations to set the `enter_long` (or `enter_short`) column to `1`.

**Syntax:**
```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            # Signal Condition: RSI < 30
            (dataframe['rsi'] < 30) &
            # Safety Check: Previous candle closed green
            (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &
            # Safety Check: Volume exists
            (dataframe['volume'] > 0)
        ),
        'enter_long'] = 1  # Buy Signal
        
    return dataframe
```

### 3.3. `populate_exit_trend`
Defines exit signals (selling). Uses vectorized operations to set the `exit_long` (or `exit_short`) column to `1`.

**Syntax:**
```python
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            # Signal Condition: RSI > 70
            (dataframe['rsi'] > 70) &
            # Volume safety check
            (dataframe['volume'] > 0)
        ),
        'exit_long'] = 1  # Sell Signal
        
    return dataframe
```

## 4. Avoiding Look-Ahead Bias
**Critical Rule:** You must never use data from the *current* candle to make a decision if that candle has not closed yet, or if you are backtesting. However, Freqtrade's backtesting engine assumes you are making decisions at the *close* of the candle.

To be explicitly safe and refer to confirmed past data (e.g., "confirm the *previous* candle closed green"):

**Syntax:**
Use `.shift(n)` to access previous rows.
- `dataframe['close'].shift(1)` = Close price of the previous candle.
- `dataframe['open'].shift(1)` = Open price of the previous candle.

**Incorrect (Look-Ahead risk if reasoning about 'current' formation in a way that implies knowledge of future close):**
```python
# Dangerous if you think this means "current candle IS green" while it's still forming
(dataframe['close'] > dataframe['open']) 
```

**Correct (Confirmed previous state):**
```python
# "The candle that just finished was green"
(dataframe['close'].shift(1) > dataframe['open'].shift(1))
```

## 5. Summary Checkpoints for Builder
1.  **Inheritance:** Must inherit `IStrategy`.
2.  **Vectorization:** No `for` loops. Use `dataframe.loc`.
3.  **Columns:** Assign `1` to `enter_long` / `exit_long`.
4.  **Shift:** Use `.shift(1)` for confirming patterns on closed candles.
