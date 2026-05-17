"""
BTC perpetual market signals attached to the strategy dataframe.

Three columns:
  btc_funding_rate            Last published 8h funding rate (decimal, e.g. 0.0001 = 1bp).
                              Positive = longs paying shorts = market is long-loaded.
  btc_oi                      BTC futures open interest in USD.
  btc_oi_pct_change_24h       1-day % change in OI. Positive = positions building,
                              negative = de-leveraging.

Both source series are shifted forward in time before joining to prevent
look-ahead: an hourly strategy bar at time T sees the funding rate that
settled at or before T (NOT the rate currently accruing, which won't
publish until the end of its 8h window), and the OI snapshot from at
least one full day before T.
"""

import pandas as pd

from .fear_and_greed import load_external_dataframe


def _join_external_metric(
    dataframe: pd.DataFrame,
    pair: str,
    timeframe: str,
    column: str,
    lookback_offset: pd.Timedelta,
) -> pd.Series:
    """Load a metric series from disk and align it onto `dataframe.index`.

    `lookback_offset` is added to every source timestamp before reindexing —
    that's the look-ahead guard. For an 8h-cadence series we add 8h so a
    bar at 09:00 sees the funding rate that settled at 00:00 (8h earlier),
    not 08:00 (which a naive ffill would otherwise expose).
    """
    src = load_external_dataframe(pair, timeframe)
    if src.empty:
        return pd.Series(pd.NA, index=dataframe.index)
    shifted = src["close"].copy()
    shifted.index = shifted.index + lookback_offset
    try:
        return shifted.reindex(dataframe.index, method="ffill")
    except (TypeError, ValueError):
        return pd.Series(pd.NA, index=dataframe.index)


def add_perp_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Inject btc_funding_rate, btc_oi, btc_oi_pct_change_24h columns.

    Safe to call when the source files are missing — columns become NaN
    rather than raising. Idempotent.
    """
    dataframe["btc_funding_rate"] = _join_external_metric(
        dataframe,
        pair="BTCFUND/USDT",
        timeframe="8h",
        column="btc_funding_rate",
        # Settlement publishes at end of the 8h window. Adding 8h means a bar
        # at 09:00 reads the rate from 00:00 — which was settled at 08:00 and
        # therefore is the most recently *published* rate as of 09:00.
        lookback_offset=pd.Timedelta(hours=8),
    ).values

    btc_oi = _join_external_metric(
        dataframe,
        pair="BTCOI/USDT",
        timeframe="1d",
        column="btc_oi",
        # Daily OI snapshot — shift by 1 day so a bar at any hour on day T
        # sees the snapshot from day T-1, not the in-progress day T value.
        lookback_offset=pd.Timedelta(days=1),
    )
    dataframe["btc_oi"] = btc_oi.values

    # 24h pct change is computed on the already-shifted series, so it carries
    # the same look-ahead guarantee.
    dataframe["btc_oi_pct_change_24h"] = btc_oi.pct_change(periods=24).values
    return dataframe
