# External Data Pipeline Specification (3rd Party Data)

## 1. Problem
Freqtrade is designed for crypto exchanges. It natively fetches OHLCV data for pairs (e.g., `BTC/USDT`).
However, we need external correlation data (Macro, On-chain, Sentiment) which is not available via the crypto exchange API.

**Examples:**
*   **VIX (Volatility Index):** Traditional Finance (Yahoo Finance).
*   **Gold (XAU):** Commodities (Yahoo/Metals API).
*   **Fear & Greed:** Alternative.me API.
*   **On-Chain:** Glassnode/CryptoQuant (API).

## 2. Solution: The "Fake Pair" Architecture
To ensure **Backtesting Compatibility**, we cannot just "fetch URL" inside the strategy. The data must exist on disk in Freqtrade's native format (`json` or `feather`) so the backtesting engine can load it time-sequentially alongside crypto data.

**The Workflow:**
1.  **Fetcher Script:** A standalone Python script runs (cronjob/manual) to fetch 3rd party data.
2.  **Normalizer:** Converts data into Freqtrade OHLCV format (`date`, `open`, `high`, `low`, `close`, `volume`).
    *   *Note:* If data is single-value (e.g., VIX=20), set Open=High=Low=Close=20.
3.  **Storage:** Saves as a "Fake Pair" in `user_data/data/binance` (or whichever exchange).
    *   Example: `VIX/USDT` (even though it doesn't exist on Binance).
4.  **Strategy:** Uses `self.dp.get_pair_dataframe('VIX/USDT', self.timeframe)` to access it.

## 3. Development Spec for `user_data/scripts/fetch_extra_data.py`

### 3.1. Requirements
*   **Libraries:** `yfinance` (for TradFi), `requests` (for APIs), `pandas`.
*   **Output:** strictly formatted headers: `date`, `open`, `high`, `low`, `close`, `volume`.

### 3.2. Implementation Logic (Draft)

```python
import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_vix():
    # Fetch VIX from Yahoo
    vix = yf.download("^VIX", period="2y", interval="1d")
    
    # Format to Freqtrade (Snake case columns)
    df = pd.DataFrame()
    df['date'] = vix.index
    df['open'] = vix['Open']
    df['high'] = vix['High']
    df['low'] = vix['Low']
    df['close'] = vix['Close']
    df['volume'] = vix['Volume'].fillna(0) # VIX often has no volume
    
    return df

def save_to_freqtrade(df, pair_name, timeframe='1d'):
    # Save as JSON or Feather
    # Filename structure: pair_timeframe.json
    # pair_name: VIX_USDT
    file_path = f"user_data/data/binance/{pair_name}-{timeframe}.json"
    
    # Convert date to timestamp (ms) as expected by Freqtrade
    df['date'] = df['date'].astype(int) // 10**6 
    
    df.to_json(file_path, orient='values') # Check Freqtrade specific JSON format structure
```

*Note: Freqtrade JSON format is actually a list of lists `[[timestamp, open, ...], ...]`. The script must respect this.*

## 4. Strategy Spec (How to use it)

### 4.1. Configuration
Add the fake pairs to `config.json` -> `exchange` -> `pair_whitelist`? NO.
**Do NOT** add them to the whitelist, or the bot will try to trade them (and fail).

### 4.2. Strategy Code
Use the DataProvider (`self.dp`) to request the data.

```python
class MyStrategy(IStrategy):
    def informative_pairs(self):
        # Tell Freqtrade to load this data from disk during backtest
        return [
            ("VIX/USDT", "1d"),
            ("GOLD/USDT", "1d"),
        ]

    def populate_indicators(self, dataframe, metadata):
        # 1. Fetch the data
        if self.dp:
            vix_df = self.dp.get_pair_dataframe(pair="VIX/USDT", timeframe="1d")
            gold_df = self.dp.get_pair_dataframe(pair="GOLD/USDT", timeframe="1d")
            
            # 2. Merge (Advanced)
            # You must merge this Informative Tuple onto the main timeframe
            dataframe = merge_informative_pair(dataframe, vix_df, self.timeframe, "1d", ffill=True)
            
            # Now you have dataframe['close_1d_VIX_USDT'] available!
            
        return dataframe
```

## 5. Summary of Workflows
1.  **Macro Analysis:** Use `VIX` to detect high-volatility regimes (Panic selling).
2.  **Correlation:** Use `GOLD` inverse correlation for Bitcoin.
3.  **On-Chain:** Fetch "Exchange Inflow" (Fear) -> save as `INFLOW/USDT`.

## 6. Action Plan
1.  Create `user_data/scripts/fetch_data.py` (The Pipeline).
2.  Run strict cronjob to update daily.
3.  Update Strategy to consume clean OHLCV data.
