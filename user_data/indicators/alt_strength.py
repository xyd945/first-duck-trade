"""
ETH/BTC ratio as a BTC-dominance proxy (R2c).

Three columns injected into the strategy dataframe:

  eth_btc_ratio              ETH/USDT close ÷ BTC/USDT close.
                             Rising = alts outperforming = BTC dominance
                             falling; falling = capital flight to BTC.

  eth_btc_change_7d          7-day % change in the ratio. Captures
                             short-term alt-strength momentum (faster than
                             a 30d z-score, slower than a daily diff).

  alt_strength_zscore_30d    z-score of eth_btc_ratio over a 30-day rolling
                             window. Normalized regime-style signal:
                               > +1.5  alt-season conditions (extreme)
                               > +0.5  alts taking share from BTC
                               -0.5..+0.5  neutral
                               < -0.5  BTC taking share from alts
                               < -1.5  capitulation into BTC (crisis-adjacent)

Why ETH/BTC and not real BTC.D market cap: free historical BTC.D data
requires CoinGecko Pro or scraping TradingView. ETH/BTC correlates with
the inverse of BTC.D around -0.85 because ETH is ~17% of total crypto
market cap and the dominant alt — it captures the majority of the
dominance signal at zero cost.

Both source series are shifted forward 1 day before join to prevent
look-ahead: a strategy bar at any hour on day T sees the daily close
from day T-1, not the in-progress day T close.
"""

import numpy as np
import pandas as pd

from .fear_and_greed import load_external_dataframe


def add_alt_strength(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Inject eth_btc_ratio, eth_btc_change_7d, alt_strength_zscore_30d columns.

    Safe when source files are missing — columns become NaN rather than
    raising. Idempotent.
    """
    eth = load_external_dataframe("ETH/USDT", "1d")
    btc = load_external_dataframe("BTC/USDT", "1d")

    if eth.empty or btc.empty:
        dataframe["eth_btc_ratio"] = pd.NA
        dataframe["eth_btc_change_7d"] = pd.NA
        dataframe["alt_strength_zscore_30d"] = pd.NA
        return dataframe

    # Inner-join on date so the ratio is only defined for days both series
    # actually published a close. Drop any zero/missing BTC closes to avoid
    # divide-by-zero polluting the series.
    joined = pd.DataFrame({
        "eth_close": eth["close"],
        "btc_close": btc["close"].replace(0, np.nan),
    }).dropna()
    if joined.empty:
        dataframe["eth_btc_ratio"] = pd.NA
        dataframe["eth_btc_change_7d"] = pd.NA
        dataframe["alt_strength_zscore_30d"] = pd.NA
        return dataframe

    ratio = joined["eth_close"] / joined["btc_close"]
    change_7d = ratio.pct_change(periods=7) * 100

    # Rolling z-score: (x - mean) / std over 30 days. min_periods=10 so we
    # produce *something* once we have a third of the window — early bars
    # in a short backtest still get a signal rather than universal NaN.
    rolling_mean = ratio.rolling(window=30, min_periods=10).mean()
    rolling_std = ratio.rolling(window=30, min_periods=10).std()
    # Avoid div-by-zero when the ratio has been flat (std=0); fall back to NaN
    # in that case rather than emitting ±inf.
    zscore = (ratio - rolling_mean) / rolling_std.replace(0, np.nan)

    # Shift +1 day so a bar at any hour on day T reads day T-1's value.
    # The shift uses the source index spacing (daily), so 1 row == 1 day.
    ratio_shifted = ratio.shift(1)
    change_shifted = change_7d.shift(1)
    zscore_shifted = zscore.shift(1)

    try:
        dataframe["eth_btc_ratio"] = ratio_shifted.reindex(
            dataframe.index, method="ffill"
        ).values
        dataframe["eth_btc_change_7d"] = change_shifted.reindex(
            dataframe.index, method="ffill"
        ).values
        dataframe["alt_strength_zscore_30d"] = zscore_shifted.reindex(
            dataframe.index, method="ffill"
        ).values
    except (TypeError, ValueError):
        # Index types incompatible — fail soft so a single bad reindex
        # doesn't break the whole indicator pipeline.
        dataframe["eth_btc_ratio"] = pd.NA
        dataframe["eth_btc_change_7d"] = pd.NA
        dataframe["alt_strength_zscore_30d"] = pd.NA

    return dataframe
