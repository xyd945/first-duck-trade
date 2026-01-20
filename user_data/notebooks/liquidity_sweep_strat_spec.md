# Liquidity Sweeps Strategy Specification (Vectorized)

## 1. Overview
This strategy adapts the "Liquidity Sweeps [xyd945] + Volume" Pine Script indicator into a **vectorized** Freqtrade strategy.
The core concept is to identify "sweeps" of key swing high/low points, confirmed by a candle close back within the range, and filtered by volume.

## 2. Key Challenges & Solutions
**Challenge:** The Pine Script uses an array of pivots and loops backward to find "unmitigated" levels.
**Solution (Vectorized):**
Since Freqtrade prohibits loops, we will implement a "Recent Liquidity" approach.
1.  **Pivot Detection:** Calculate Pivot Highs and Lows using a rolling window (similar to `ta.pivothigh`).
2.  **Vectorized "Active" Levels:** instead of an array of unmitigated levels, we will check if the *current* price sweeps the *most recent* significant pivot within a lookback window (e.g., last 10-20 candles, or the last confirmed pivot).
3.  **Strict Vectorization:** We will use `rolling().min()` / `.max()` or `idxmin()` / `idxmax()` logic to find the local extrema that serve as liquidity pools.

## 3. Configuration & Inputs (Hyperopt Ready)
In Freqtrade, parameters that should be optimized must be defined as class attributes using `IntParameter`, `DecimalParameter`, or `CategoricalParameter`.

```python
from freqtrade.strategy import IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter

class LiquiditySweepsStrategy(IStrategy):
    # ... other settings ...

    # Hyperoptable Parameters
    # -----------------------
    # Pivot Lookback: How many candles to look before/after finding a pivot
    pivot_len = IntParameter(2, 10, default=5, space='buy', optimize=True)
    
    # Volume Filter
    # Note: Boolean optimization is essentially Categorical([True, False])
    use_vol_check = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Volume MA Length
    vol_len = IntParameter(10, 50, default=20, space='buy', optimize=True)
    
    # Volume Multiplier relative to MA
    vol_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space='buy', optimize=True)
    
    # How far back to search for the "swept" liquidity level
    sweep_lookback = IntParameter(10, 60, default=20, space='buy', optimize=True)

    # Noise Filter: Minimum percentage distance for a sweep to be valid (0.1% to 1%)
    min_sweep_dist = DecimalParameter(0.001, 0.01, default=0.002, decimals=3, space='buy', optimize=True)

    # RSI Risk Management
    rsi_exit_threshold = IntParameter(60, 90, default=75, space='sell', optimize=True)


    # ...
```

**Instruction for Builder:**
*   Do NOT use a static dictionary method for these numbers.
*   Access these values in your logic using `self.pivot_len.value`, `self.vol_len.value`, etc.
*   Ensure the `optimize=True` flag is set so the user can run `freqtrade hyperopt`.


## 4. Logic Specification for Builder

### 4.1. Helper: Pivot Detection (Vectorized)
Implement a helper (or use `pandas_ta` if available, otherwise raw pandas):
*   **Pivot Low:** A candle causing a local minimum over `2 * pivot_len + 1` bars.
    *   `df['pivot_low']` = Price where `low` is minimum in window.
    *   Mark `NaN` if not a pivot.

### 4.2. Helper: Find "Target" Pivot
To detect a sweep of a *previous* pivot:
*   We need the *value* of the most recent confirmed Pivot Low that is *older* than the current swing.
*   **Vectorized Trick:** `df['recent_pivot_low'] = df['low'].rolling(window=swings_lookback).min().shift(1)`
    *   *Refinement:* This finds the lowest low. The logic requires sweeping a *specific pivot*, not just the lowest low.
    *   **Better Approach for Spec:** Use `scipy.signal.argrelextrema` (allowed import) OR pandas simple rolling pivots.
    *   **Simplest Robust Method:**
        1. Identify Pivot candidates: `is_pivot = (low == low.rolling(center=True, window=N).min())`.
        2. Forward fill the *value* of the last pivot: `df['last_pivot_low'] = df.loc[is_pivot, 'low']`.
        3. `df['last_pivot_low'] = df['last_pivot_low'].ffill()`
        4. **CRITICAL:** `shift(1)` the `last_pivot_low` column so we are comparing to the *established* pivot, not creating one right now.

### 4.3. Entry Signal (Long) - "Bullish Sweep"
Condition to set `enter_long = 1`:

1.  **Sweep Condition:**
    *   `df['low'] < df['last_pivot_low']` (Wick goes below previous pivot)
    *   `df['close'] > df['last_pivot_low']` (Body closes back above - Reclaim)
2.  **Volume Filter (Confirmation):**
    *   `df['volume'] > (df['volume'].rolling(vol_len).mean() * vol_mult)`
3.  **Noise Filter (New):**
    *   To filter out "noisy" small sweeps, the sweep depth (pivot - low) must be significant enough.
    *   `sweep_depth = (df['last_pivot_low'] - df['low']) / df['low']`
    *   `min_sweep_threshold = 0.002` (0.2%, adjustable via Hyperopt `min_sweep_dist`)
    *   Condition: `sweep_depth > min_sweep_threshold`

### 4.4. Exit Signal (Long)
We will combine multiple exit mechanisms to avoid stagnation.

**A. Opposing Sweep (Standard):**
*   If `df['high'] > df['last_pivot_high']` and `df['close'] < df['last_pivot_high']` -> `exit_long = 1`

**B. "Time-Based" Stagnation Exit:**
*   If price hasn't moved X% in Y candles, exit.
*   Freqtrade has built-in `unfilledtimeout` and `ignore_roi_if_entry_signal`.
*   *Custom Implementation:*
    *   `dataframe['date_diff'] = (dataframe['date'] - dataframe['date'].shift(timeout_candles))`
    *   (Too complex for vectorized basic, rely on ROI for now or simple "Close if RSI > 80").

**C. RSI Overbought Exit (Fast Exit):**
*   `dataframe['rsi'] > rsi_exit_threshold` (e.g., 75)

### 4.5. Risk Management (Dynamic)
*   **Stoploss:** Instead of fixed, use recent swing low.
    *   *Note:* Freqtrade's `stoploss` is fixed percentage from entry. For dynamic stoploss (at the sweep low), we need `custom_stoploss`.
    *   **Builder Instruction:** Implement `custom_stoploss` method.
        *   Stop price = `dataframe['low']` of the sweep candle (or slightly below).


## 5. Implementation Guidance for Builder
*   **Imports:** `from scipy.signal import argrelextrema` is helpful for faster pivot detection, OR use `df['low'].rolling(window=n, center=True).min() == df['low']`.
*   **Forward Fill:** Use `.ffill()` effectively to propagate the "level to sweep" forward in time.
*   **Shift:** ALways compare against `.shift(1)` of the pivot levels to avoid lookahead/current-bar bias.
