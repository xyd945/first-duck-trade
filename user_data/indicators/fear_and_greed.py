import pandas as pd
import pandas_ta as ta
import numpy as np
import json
from pathlib import Path

def resolve_target_index(dataframe: pd.DataFrame) -> pd.Index:
    """Return an index whose labels match the dataframe's rows in time.

    Freqtrade hands strategies a RangeIndex with a ``date`` column. Manual
    tests and notebooks typically use a true DatetimeIndex. External-data
    reindex calls must align on timestamps either way — if we reindex onto
    a RangeIndex when the source has a DatetimeIndex, every label misses
    and the result is silently all-NaN (which silently disabled vix, gold,
    dxy, spx, btc_funding_rate, btc_oi for every Freqtrade backtest until
    this was discovered).
    """
    if isinstance(dataframe.index, pd.DatetimeIndex):
        return dataframe.index
    if 'date' in dataframe.columns:
        return pd.DatetimeIndex(dataframe['date'])
    return dataframe.index


def load_external_dataframe(pair_name: str, timeframe: str = '1d', data_dir: str = 'user_data/data/binance') -> pd.DataFrame:
    """
    Load external data (VIX/USDT, GOLD/USDT) from JSON files fetched by fetch_extra_data.py.
    """
    pair_filename = pair_name.replace("/", "_")
    file_path = Path(f"{data_dir}/{pair_filename}-{timeframe}.json")
    
    if not file_path.exists():
        # Fallback: Look in project root relative path if run from there
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / data_dir / f"{pair_filename}-{timeframe}.json"
        
    if not file_path.exists():
        print(f"WARNING: External data file not found: {file_path}")
        return pd.DataFrame()
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Freqtrade format: [[timestamp, open, high, low, close, volume], ...]
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"ERROR loading external data {file_path}: {e}")
        return pd.DataFrame()

def add_fear_and_greed(dataframe: pd.DataFrame, fast_length: int = 21, slow_length: int = 144, smooth_length: int = 5) -> pd.DataFrame:
    """
    Fear & Greed Index by DGT (Python Port) with External Data (VIX, GOLD).
    
    Ported from Pine Script: https://www.tradingview.com/script/0l30Y20J-Trading-Psychology-Fear-Greed-Index-by-DGT/
    
    Components:
    1. PMACD (Price Convergence)
    2. RoR (Rate of Return)
    3. Money Flow (Chaikin / Accumulation Distribution)
    4. VIX (Volatility Index - External)
    5. GOLD (Safe Haven Demand - External)
    """
    df = dataframe.copy()
    
    # --- 1. PMACD (Price Convergence/Divergence) ---
    # Pine: pmacd = (close / ta.ema(close, slowLength) - 1) * 100
    ema_slow = ta.ema(df['close'], length=slow_length)
    # pandas_ta returns None when the input has fewer non-NaN values than
    # `length` — Freqtrade's mini-backtest can hit this if startup candles
    # haven't loaded yet. Fall back to a zero-centered Series so the rest
    # of the indicator pipeline still produces something.
    if ema_slow is None:
        ema_slow = pd.Series(df['close'].mean(), index=df.index)
    pmacd = (df['close'] / ema_slow - 1) * 100
    
    # --- 2. RoR (Rate of Return) ---
    # Pine: ror = (close - close[slowLength]) / close[slowLength] * 100
    ror = df['close'].pct_change(slow_length) * 100
    
    # --- 3. Money Flow ---
    # Pine: accDist = close == high and close == low or high == low ? 0 : (2 * close - low - high) / (high - low)
    hl_diff = df['high'] - df['low']
    hl_diff = hl_diff.replace(0, np.nan) 
    
    acc_dist = (2 * df['close'] - df['low'] - df['high']) / hl_diff
    acc_dist = acc_dist.fillna(0)
    
    nz_volume = df['volume'].fillna(0)
    
    # Pine: moneyFlow = math.sum(accDist * nzVolume, fastLength) / math.sum(nzVolume, fastLength) * 100
    mf_num = (acc_dist * nz_volume).rolling(fast_length).sum()
    mf_denom = nz_volume.rolling(fast_length).sum()
    
    # Avoid div by zero
    money_flow = (mf_num / mf_denom.replace(0, np.nan)) * 100
    money_flow = money_flow.fillna(0)

    # --- 4. VIX (Volatility Index) ---
    # Pine: vix = request.security('VIX', timeframe.period, -(close / ta.ema(close, slowLength) - 1) * 100)
    # We fetch daily VIX data.
    # Logic: -(vix_close / ema(vix_close, slow_length) - 1) * 100.
    # Note on inverse logic: High VIX = Fear. Low VIX = Greed? 
    # Pine script formula: `-(close / ta.ema(close, slowLength) - 1) * 100`. 
    # If VIX is high (above EMA), result is negative (Fear). 
    # If VIX is low (below EMA), result is positive (Greed).
    
    vix_df = load_external_dataframe('VIX/USDT', '1d')
    vix_val = pd.Series(0, index=df.index) # Default 0
    
    if not vix_df.empty:
        # Calculate metric on VIX dataframe first
        vix_ema = ta.ema(vix_df['close'], length=slow_length)
        if vix_ema is None:
            # Too few VIX rows to compute EMA — fall back to neutral signal
            vix_ema = pd.Series(vix_df['close'].mean(), index=vix_df.index)
        vix_dbq = -(vix_df['close'] / vix_ema - 1) * 100
        
        # Merge to main dataframe using ffill (Daily data to lower timeframe)
        # We reindex to match main dataframe
        # Combine index to handle missing timestamps
        # IMPORTANT: Avoid lookahead bias. reindex(method='ffill') is mostly safe if timestamps align.
        # Ideally, we verify timestamp.
        # Shift by 1 day to avoid look-ahead bias: use yesterday's VIX close,
        # not today's (today's close isn't known until end of day)
        vix_dbq = vix_dbq.shift(1)
        try:
            vix_val = vix_dbq.reindex(df.index, method='ffill').fillna(0)
        except (TypeError, ValueError):
            # Index types incompatible or duplicate labels
            vix_val = pd.Series(0, index=df.index)

    # --- 5. GOLD (Safe Haven Demand) ---
    # Pine: gold = request.security('GOLD', timeframe.period, -(1 - close[fastLength] / close) * 100)
    # Logic: -(1 - close[fast] / close) * 100 = - ( (close - close[fast]) / close ) * 100 ??? 
    # Let's simplify algebra: -(1 - prev/curr) = curr/curr - prev/curr = (curr - prev)/curr ? NO. 
    # It is roughly negative Rate of Return or Momentum.
    # If Gold goes UP, Fear goes UP (so index should go DOWN/Negative).
    # If Gold goes UP, `close` > `close[fast]`. `close[fast]/close` < 1. `1 - ratio` > 0.
    # So `-(positive)` is negative (Fear). Correct.
    
    gold_df = load_external_dataframe('GOLD/USDT', '1d')
    gold_val = pd.Series(0, index=df.index)
    
    if not gold_df.empty:
        # close[fast] is shift(fast)
        gold_fast = gold_df['close'].shift(fast_length)
        # Avoid div by zero
        gold_curr = gold_df['close'].replace(0, np.nan)
        
        gold_metric = -(1 - gold_fast / gold_curr) * 100
        # Shift by 1 day to avoid look-ahead bias (same as VIX)
        gold_metric = gold_metric.shift(1)
        try:
            gold_val = gold_metric.reindex(df.index, method='ffill').fillna(0)
        except (TypeError, ValueError):
            gold_val = pd.Series(0, index=df.index)

    # --- Cycle Calculation ---
    # Pine: cycle_raw = nzVolume ? math.avg(pmacd, ror, moneyFlow, vix, gold) : math.avg(pmacd, ror, vix, gold)
    # We essentially average all available components.
    
    # If volume is missing (often in crypto volume is mostly present, but let's follow logic)
    # In pandas, mean(axis=1) handles NaNs by ignoring them? No, we need to be explicit.
    
    # Create DataFrame of components
    components = pd.DataFrame({
        'pmacd': pmacd,
        'ror': ror,
        'mf': money_flow,
        'vix': vix_val,
        'gold': gold_val
    })
    
    # If volume is 0 or nan, exclude MF?
    # Pine: nzVolume ? avg(...) : avg(no MF)
    # We can just average them. If MF is nan (due to missing volume), pandas mean skips it.
    # Our money_flow calculation fills 0, so it is always included unless we intentionally define it as NaN.
    # For simplicity, we average all 5.
    
    cycle_raw = components.mean(axis=1)
    
    # --- Smoothing ---
    # Pine: cycle = ta.rma(cycle_raw, smoothLength)
    # RMA is EMA with alpha = 1/length
    cycle = cycle_raw.ewm(alpha=1/smooth_length, adjust=False).mean()
    
    dataframe['fgi'] = cycle.values

    return dataframe
