"""
Fetch External Data Script

This script fetches correlation data (VIX, Gold, etc.) from Yahoo Finance (yfinance)
and saves it as Freqtrade-compatible OHLCV JSON files ("Fake Pairs").

Requirements:
    pip install yfinance pandas

Usage:
    python3 user_data/scripts/fetch_extra_data.py

Output:
    Saves files to: user_data/data/binance/ (or target exchange)
    Format: PAIR-TIMEFRAME.json (e.g. VIX_USDT-1d.json)
"""

import sys
from pathlib import Path

# Ensure user_data is in path if needed, though this script is standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TARGET_EXCHANGE = "binance"  # Where to save the "fake pair" data
DATA_DIR = PROJECT_ROOT / "user_data" / "data" / TARGET_EXCHANGE

# Map External Symbol -> Freqtrade Pair Name
# Note: we append /USDT to make Freqtrade happy, even though it's not a real pair
PAIRS_TO_FETCH = {
    "^VIX": "VIX/USDT",       # Volatility Index
    "GC=F": "GOLD/USDT",      # Gold Futures
    "^GSPC": "SPX/USDT",      # S&P 500
    "DX-Y.NYB": "DXY/USDT",   # US Dollar Index
}

TIMEFRAME = "1d"  # Daily data is usually best for macro
PERIOD = "2y"     # How far back to fetch

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_data_dir(directory: Path):
    if not directory.exists():
        print(f"Creating directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

def fetch_yahoo_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    print(f"Fetching {symbol} from Yahoo Finance...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"WARNING: No data found for {symbol}")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"ERROR fetching {symbol}: {e}")
        return pd.DataFrame()

def format_for_freqtrade(df: pd.DataFrame) -> list:
    """
    Convert Yahoo DataFrame to Freqtrade JSON format.
    Freqtrade expects a list of lists: [[timestamp, open, high, low, close, volume], ...]
    Timestamp should be in milliseconds.
    """
    # Convert index to UTC timestamps in milliseconds
    timestamps_ms = (df.index.tz_convert(timezone.utc)
                     .astype('int64') // 10**6)

    # Fill NaN volumes (common for indices like VIX)
    volumes = df['Volume'].fillna(0.0)

    # Build array: [timestamp, open, high, low, close, volume]
    result = pd.DataFrame({
        'ts': timestamps_ms,
        'open': df['Open'].astype(float),
        'high': df['High'].astype(float),
        'low': df['Low'].astype(float),
        'close': df['Close'].astype(float),
        'volume': volumes.astype(float),
    })

    # Sort by timestamp and return as list of lists
    result = result.sort_values('ts')
    return result.values.tolist()

def save_to_json(data: list, pair_name: str, timeframe: str):
    # Sanitize pair name for filename (VIX/USDT -> VIX_USDT)
    filename_pair = pair_name.replace("/", "_")
    filename = f"{filename_pair}-{timeframe}.json"
    file_path = DATA_DIR / filename
    
    ensure_data_dir(DATA_DIR)
    
    print(f"Saving {len(data)} candles to {file_path}...")
    
    with open(file_path, 'w') as f:
        json.dump(data, f)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"--- External Data Fetcher ---")
    print(f"Target Directory: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        # Try creating it, assuming user has basic setup
        # If exchange dir doesn't exist, this creates it
        ensure_data_dir(DATA_DIR)

    for yf_symbol, ft_pair in PAIRS_TO_FETCH.items():
        df = fetch_yahoo_data(yf_symbol, period=PERIOD, interval=TIMEFRAME)
        
        if not df.empty:
            freqtrade_data = format_for_freqtrade(df)
            save_to_json(freqtrade_data, ft_pair, TIMEFRAME)
            print(f"✅ Successfully saved {ft_pair}")
        else:
            print(f"❌ Failed to process {ft_pair}")
            
    print("--- Done ---")

if __name__ == "__main__":
    main()
