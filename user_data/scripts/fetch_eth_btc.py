"""
Fetch ETH/USDT and BTC/USDT daily candles from Binance public klines API
for the BTC-dominance proxy (R2c).

Two files, both written in Freqtrade OHLCV JSON format so the existing
`load_external_dataframe()` loader picks them up unchanged:

  data/binance/ETH_USDT-1d.json   ~1000 days of ETH spot
  data/binance/BTC_USDT-1d.json   ~1000 days of BTC spot

Why Binance and not OKX (our trading venue): we trade BTC on OKX, but the
indicator computes ETH/BTC by dividing two series row-by-row, so the two
must come from the same exchange to guarantee aligned timestamps. Binance
spot has the deepest free public-API limit (1000 daily candles per call,
no auth) and the price discovery is essentially identical to OKX at daily
cadence.

The alt_strength indicator does the look-ahead shift (1 day) at join time;
we write the raw history here.
"""

import json
import logging
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger("fetch_eth_btc")

BASE = "https://api.binance.com"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "binance"

PAIRS = [
    ("ETHUSDT", "ETH_USDT"),
    ("BTCUSDT", "BTC_USDT"),
]


def fetch_klines(symbol: str, interval: str = "1d", limit: int = 1000) -> list:
    """Return [[ts_ms, open, high, low, close, volume], ...] sorted ascending.

    Binance kline response is a list of 12-element arrays — we keep the first
    six (timestamp, OHLCV) which is exactly the Freqtrade JSON shape.
    """
    r = requests.get(
        f"{BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json()
    return [
        [int(row[0]), float(row[1]), float(row[2]), float(row[3]),
         float(row[4]), float(row[5])]
        for row in rows
    ]


def write_json(out: Path, rows: list) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    log.info(f"Wrote {len(rows)} rows to {out}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rc = 0
    for symbol, ft_pair in PAIRS:
        try:
            klines = fetch_klines(symbol)
            write_json(OUT_DIR / f"{ft_pair}-1d.json", klines)
        except Exception as e:
            log.error(f"{symbol} fetch failed: {e}")
            rc = 1
        # Polite gap between API calls
        time.sleep(0.5)

    return rc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
