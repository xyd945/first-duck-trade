# Liquidity Sweep Strategy Improvement Plan

## 1. Problem Analysis
The initial backtest results (31.8% win rate) indicate that the strategy is catching "knives" rather than valid reversals.
1.  **Fighting the Trend:** Buying sweeps in a strong downtrend.
2.  **Weak Pivots:** Treating every local low as a liquidity zone.
3.  **No Market Structure:** Buying blindly without confirmation of a shift.

## 2. Improved Logic Specification

### 2.1. Trend Filter (The "Do Not Trade" Filter)
We will add a higher-level trend filter. If the trend is bearish, we **disable** long entries.

*   **Filter 1: EMA alignment (Hyperoptable)**
    *   `EMA_short` (e.g., 50) > `EMA_long` (e.g., 200).
    *   Price > `EMA_200`.
*   **Filter 2: ADX (Trend Strength)**
    *   If ADX > 25, the trend is strong.
    *   *Rule:* If ADX > 25 AND Short EMA < Long EMA, strictly FORBID buying (strong downtrend).

### 2.2. Market Structure Confirmation (The "Reversal" Check)
Instead of just "wick below, close above", we demand a **Break of Structure (BOS)** on a specific confirming candle.

*   **Logic:**
    1.  **Sweep:** Wick < Pivot Low.
    2.  **Reclaim:** Close > Pivot Low.
    3.  **Confirmation (New):**
        *   Option A (Conservative): Close > High of the *sweep candle*.
        *   Option B (Aggressive): Close > Pivot Low + `ATR * N`.
    *   **Implementation:** We will use Option A. The signal triggers only when a candle closes above the high of the sweep candle.

### 2.3. Robust Pivot Detection (The "Real Zone" Check)
Mechanical rolling pivots are too noisy. We need "significant" levels.

*   **Improvement:** "Fractal" or "Time-Filtered" Pivots.
    *   Require the pivot to be the lowest point in a wider window (e.g., 20 candles).
    *   **Volume Filter for Pivots:** The candle forming the pivot low should ideally have higher-than-average volume (indicating a stopping volume event previously). *Optional, maybe too strict.*
    *   **Spacing:** Ensure the new sweep is at least `N` candles away from the pivot (avoid sweeping a low formed 2 bars ago). `sweep_lookback` min value should be higher.

---

## 3. Configuration Update (Hyperopt)

```python
class LiquiditySweepStrategy(IStrategy):

    # ... existing params ...

    # 1. Trend Filter
    trend_ema_short = IntParameter(20, 100, default=50, space='buy', optimize=True)
    trend_ema_long = IntParameter(100, 300, default=200, space='buy', optimize=True)
    
    # 2. ADX Filter
    use_adx_filter = BooleanParameter(default=True, space='buy', optimize=True)
    adx_threshold = IntParameter(20, 40, default=25, space='buy', optimize=True)

    # 3. Confirmation
    # Require close > high of sweep candle?
    require_candle_break = BooleanParameter(default=True, space='buy', optimize=True)

    # 4. Pivot Freshness
    # Minimum bars between pivot formation and sweep
    min_pivot_age = IntParameter(5, 30, default=10, space='buy', optimize=True)
```

## 4. Implementation Steps for Builder

1.  **Trend Indicators:** Add EMA_50, EMA_200, ADX in `populate_indicators`.
2.  **Pivot Age Logic:**
    *   When storing `last_pivot_low`, also store `last_pivot_index` (or calculate age vector).
    *   `pivot_age = current_index - last_pivot_index`.
    *   Condition: `pivot_age > min_pivot_age`.
3.  **Confirmation Logic:**
    *   Identify the sweep candle.
    *   Signal triggers on the *next* candle that closes above `sweep_candle_high`.
    *   *Note:* This might be hard to vectorize perfectly in one step.
    *   *Simplified Vectorized Approach:*
        *   `is_sweep_candle = (low < last_pivot) & (close > last_pivot)`
        *   `sweep_high = df['high']` where `is_sweep_candle` is True, ffill().
        *   `entry_signal = (close > sweep_high) & (close.shift(1) <= sweep_high)`... this is complex.
    *   *Alternative:* Just ensure the *sweep candle itself* is strong.
        *   `close > (high + low) / 2` (Closes in upper half).
        *   `body_size > wick_size * 0.5` (Significant body).

## 5. Revised Vectorized Entry Logic
```python
# 1. Trend Filter
is_uptrend = (df['ema_short'] > df['ema_long']) & (df['close'] > df['ema_long'])
if self.use_adx_filter.value:
    # If strong downtrend, block
    is_strong_downtrend = (df['adx'] > self.adx_threshold.value) & (df['ema_short'] < df['ema_long'])
    trend_ok = is_uptrend & (~is_strong_downtrend)
else:
    trend_ok = is_uptrend

# 2. Sweep & Freshness
# ... calculate last_pivot_low and pivot_age ...
sweep_condition = (df['low'] < df['last_pivot_low']) & (pivot_age > self.min_pivot_age.value)

# 3. Candle Confirmation
# Candle closes strongly back inside OR closes above sweep candle high
# Simplified: Sweep candle must close in top 30% of its range (strong rejection)
candle_range = df['high'] - df['low']
reclaim_strength = (df['close'] - df['low']) / candle_range
is_strong_reclaim = reclaim_strength > 0.7  # Hyperoptable?

entry = trend_ok & sweep_condition & is_strong_reclaim
```
