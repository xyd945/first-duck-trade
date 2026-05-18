"""
External macro data injection for generated strategies.

`add_external_data(dataframe)` is a single entry point that adds macro
columns to the strategy dataframe:

  - fgi                    composite Fear & Greed (PMACD + RoR + Money Flow + VIX + Gold)
  - vix                    CBOE Volatility Index close (1d, ffilled to strategy timeframe)
  - gold                   Gold futures close
  - dxy                    US Dollar Index close
  - spx                    S&P 500 close
  - btc_funding_rate       BTC perpetual funding rate (positioning signal)
  - btc_oi                 BTC futures open interest in USD
  - btc_oi_pct_change_24h  1-day % change in OI
  - eth_btc_ratio          ETH/BTC price ratio (BTC dominance proxy)
  - eth_btc_change_7d      7-day % change in ETH/BTC (alt momentum)
  - alt_strength_zscore_30d  30d rolling z-score of ETH/BTC (regime signal)

All raw series are shifted forward in time before reindexing to prevent
look-ahead bias (daily closes need a 1-day shift, 8h funding needs an 8h
shift, etc.). See each indicator module for the exact offset. Missing data
files produce columns of NaN rather than exceptions; strategies should
handle NaN with `.fillna()` or explicit guards.
"""

import pandas as pd

from .alt_strength import add_alt_strength
from .fear_and_greed import add_fear_and_greed, load_external_dataframe
from .perp_metrics import add_perp_metrics


def _attach_external_close(
    dataframe: pd.DataFrame,
    pair: str,
    column: str,
    timeframe: str = "1d",
) -> None:
    """Reindex an external close series onto the strategy dataframe index.

    Writes to `dataframe[column]` in place. Shifts by 1 day to avoid using a
    daily close before the day has ended. If the source file is missing or
    unreadable, writes NaN — never raises.
    """
    src = load_external_dataframe(pair, timeframe)
    if src.empty:
        dataframe[column] = pd.NA
        return
    try:
        series = src["close"].shift(1).reindex(dataframe.index, method="ffill")
        dataframe[column] = series.values
    except (TypeError, ValueError):
        dataframe[column] = pd.NA


def add_external_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Inject all available external macro columns + fgi into the dataframe.

    Idempotent: calling twice is safe. Generated strategies should call this
    as the FIRST line of populate_indicators so the columns are guaranteed
    present before any entry/exit logic references them.
    """
    dataframe = add_fear_and_greed(dataframe)
    _attach_external_close(dataframe, "VIX/USDT", "vix")
    _attach_external_close(dataframe, "GOLD/USDT", "gold")
    _attach_external_close(dataframe, "DXY/USDT", "dxy")
    _attach_external_close(dataframe, "SPX/USDT", "spx")
    dataframe = add_perp_metrics(dataframe)
    dataframe = add_alt_strength(dataframe)
    return dataframe
