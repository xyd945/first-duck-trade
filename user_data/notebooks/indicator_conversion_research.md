# Research: Converting TradingView Indicators to Freqtrade (Python)

## 1. Objective
Establish a standard workflow for porting Pine Script indicators to Python so they can be **reused** across multiple Freqtrade strategies without code duplication.

## 2. Recommended Folder Structure
Freqtrade doesn't have a rigid "indicators" folder standard, but we can enforce a Pythonic library structure.

**Proposed Structure:**
```text
user_data/
├── strategies/
│   ├── MyStrategy.py
│   └── ...
├── indicators/              <-- NEW FOLDER
│   ├── __init__.py          <-- Makes it a package
│   ├── liquidity_sweeps.py  <-- Specific indicator logic
│   └── custom_rsi.py        <-- Another indicator
```

**Importing in Strategy:**
Since `user_data` is mounted, we can add it to the system path to ensure imports work reliably in Docker or local envs.

```python
import sys
import os
from pathlib import Path

# Add user_data to path if not present
user_data_path = Path(__file__).parent.parent
if str(user_data_path) not in sys.path:
    sys.path.append(str(user_data_path))

# Now import clean and reusable
from indicators.liquidity_sweeps import detect_liquidity_sweeps
```

## 3. Pine Script vs. Python (Pandas) Conversion Table

| Concept | Pine Script | Python (pandas / pandas_ta) |
| :--- | :--- | :--- |
| **Series** | `close`, `high`, `low` | `dataframe['close']`, `dataframe['high']` |
| **Shift/Offset** | `close[1]` (Previous candle) | `dataframe['close'].shift(1)` |
| **Missing Data** | `na` | `nan` (via `numpy.nan`) |
| **Condition** | `iff(cond, true, false)` | `np.where(cond, val_true, val_false)` |
| **RSI** | `ta.rsi(close, 14)` | `ta.rsi(dataframe['close'], length=14)` |
| **Rolling Max** | `ta.highest(high, 20)` | `dataframe['high'].rolling(20).max()` |
| **Crossover** | `ta.crossover(a, b)` | `qtpylib.crossed_above(a, b)` |
| **Variables** | `var float x = 0.0` | Requires `custom_indicator` (slower) or vectorization tricks |

## 4. Conversion Workflow

### Step 1: Isolate the Logic
Separate the "Visuals" (Plotting) from the "Calculation" (Logic). Freqtrade only needs the calculation.

**Pine:**
```json
// Logic
rsi = ta.rsi(close, 14)
signal = rsi < 30
// Visual
plot(rsi)
bgcolor(signal ? color.green : na)
```

### Step 2: Create Python Function
Write a function that accepts the full `dataframe` and returns the specific series or modifies the dataframe in place.

**Python (`user_data/indicators/my_cool_indicator.py`):**
```python
import pandas_ta as ta

def add_my_cool_indicator(dataframe, rsi_len=14, oversold=30):
    """
    Reusable indicator: Calculates RSI and adds 'buy_signal' column.
    """
    # 1. Calculate Core Indicator
    dataframe['my_rsi'] = ta.rsi(dataframe['close'], length=rsi_len)
    
    # 2. Return the Series (or modify dataframe)
    # Returning series is more "functional" and reusable
    return dataframe['my_rsi']
```

### Step 3: Use in Strategy (`populate_indicators`)
```python
from indicators.my_cool_indicator import add_my_cool_indicator

class MyStrategy(IStrategy):
    def populate_indicators(self, dataframe, metadata):
        # Reuse the logic
        dataframe['rsi_custom'] = add_my_cool_indicator(dataframe, rsi_len=14)
        return dataframe
```

## 5. Handling Advanced Pine (Loops & State)
Pine Script is excellent at handling state (variables that keep value between bars).
*   **Vectorization First:** Try to convert loops to rolling windows.
*   **Numpy:** Use `numpy.where` for conditional logic.
*   **Last Resort:** Use `dataframe.apply()` (Slow! Avoid if possible).

## 6. Conclusion
Yes, converting is highly recommended.
1.  **Extract** Pine logic.
2.  **Vectorize** into Pandas/Numpy.
3.  **Package** into `user_data/indicators/`.
4.  **Import** into strategies.
