"""
External macro data injection for generated strategies.

`add_external_data(dataframe)` is a single entry point that adds five macro
columns to the strategy dataframe:

  - fgi   composite Fear & Greed index (PMACD + RoR + Money Flow + VIX + Gold)
  - vix   CBOE Volatility Index close (1d, ffilled to strategy timeframe)
  - gold  Gold futures close
  - dxy   US Dollar Index close
  - spx   S&P 500 close

All four raw series (vix/gold/dxy/spx) are shifted by 1 day before reindexing
to the strategy dataframe — daily closes are only known after the day ends, so
shifting prevents look-ahead bias. Missing data files produce a column of NaN
rather than an exception; strategies should handle NaN with `.fillna(0)` or
explicit guards.

The fgi component already lives in fear_and_greed.py; this module reuses its
loader and adds the raw closes the LLM can build cross-asset conditions from
(e.g. "enter when vix below 18 and dxy weakening").
"""

import pandas as pd

from .fear_and_greed import add_fear_and_greed, load_external_dataframe


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
    return dataframe
