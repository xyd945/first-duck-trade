"""
Fetch BTC perpetual market signals from Binance Futures public API.

Two series, both written as fake-OHLCV JSON next to the macro feeds so
`load_external_dataframe()` can pick them up unchanged:

  data/binance/BTCFUND_USDT-8h.json   funding rate (rate stored in close)
  data/binance/BTCOI_USDT-1d.json     open interest (USD value in close)

Why Binance and not OKX (our trading venue):
  OKX's funding-rate history is paginated 100 at a time (~33 days/call), and
  there is no public OI-history endpoint that gives more than ~5 days. Binance
  futures public endpoints return 333 days of funding and 500 days of daily OI
  in a single call each. The BTCUSDT-PERP funding rate is essentially the same
  signal on both venues — we trade on OKX but we read positioning from Binance.

Look-ahead handling lives in the indicator layer (perp_metrics.add_perp_metrics
shifts both series by 1 settlement before joining), so this script does NOT
shift here — we want the raw history on disk.
"""

import json
import logging
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger("fetch_perp_data")

BASE = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "binance"


def fetch_funding_rate_history(symbol: str = SYMBOL, limit: int = 1000) -> list:
    """Return [[ms_timestamp, rate], ...] sorted ascending. Up to 1000 entries
    (~333 days of 8h funding).
    """
    r = requests.get(
        f"{BASE}/fapi/v1/fundingRate",
        params={"symbol": symbol, "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json()
    # Binance returns dicts with fundingTime (ms) + fundingRate (str decimal)
    return sorted(
        [(int(row["fundingTime"]), float(row["fundingRate"])) for row in rows]
    )


def fetch_open_interest_history(
    symbol: str = SYMBOL, period: str = "1d", limit: int = 500
) -> list:
    """Return [[ms_timestamp, oi_usd], ...] sorted ascending. 1d cadence gives
    ~500 days of history in a single call.

    Binance returns `sumOpenInterestValue` in USD which is what we want.
    """
    r = requests.get(
        f"{BASE}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json()
    return sorted(
        [(int(row["timestamp"]), float(row["sumOpenInterestValue"])) for row in rows]
    )


def to_freqtrade_ohlcv(rows: list) -> list:
    """Wrap a [[ts, value], ...] series into Freqtrade's [[ts, o, h, l, c, v], ...]
    format so the existing load_external_dataframe loader can read it.
    `close` carries the metric; o/h/l mirror it; volume is a constant.
    """
    return [[ts, val, val, val, val, 1.0] for ts, val in rows]


def write_json(out: Path, rows: list) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    log.info(f"Wrote {len(rows)} rows to {out}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        funding = fetch_funding_rate_history()
        write_json(OUT_DIR / "BTCFUND_USDT-8h.json", to_freqtrade_ohlcv(funding))
    except Exception as e:
        log.error(f"Funding fetch failed: {e}")
        return 1

    # Polite gap between API calls
    time.sleep(0.5)

    try:
        oi = fetch_open_interest_history(period="1d", limit=500)
        write_json(OUT_DIR / "BTCOI_USDT-1d.json", to_freqtrade_ohlcv(oi))
    except Exception as e:
        log.error(f"OI fetch failed: {e}")
        return 2

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
