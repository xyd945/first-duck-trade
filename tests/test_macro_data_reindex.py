"""Regression tests for the macro-data reindex bug.

Background: Freqtrade hands strategies a dataframe with a RangeIndex and a
'date' column. The external-data indicators were doing
``src.reindex(dataframe.index, method='ffill')`` against source series whose
index is a DatetimeIndex. With a RangeIndex target every label misses and
the result is silently all-NaN — every macro filter using vix/gold/dxy/spx/
btc_funding_rate/btc_oi was a no-op for every Freqtrade backtest until this
was discovered.

These tests assert the fix: each indicator must populate non-NaN values on
both a DatetimeIndex dataframe AND a RangeIndex+date-column dataframe, and
must align the same values to the same timestamps in either shape.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data"))


def _datetime_indexed_ohlcv(n: int = 240, start: str = "2026-04-01 00:00") -> pd.DataFrame:
    """Manual/notebook shape — DatetimeIndex, no 'date' column."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100 + np.linspace(0, 5, n)
    return pd.DataFrame({
        "open": close - 0.1, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


def _freqtrade_shape_ohlcv(n: int = 240, start: str = "2026-04-01 00:00") -> pd.DataFrame:
    """Freqtrade shape — RangeIndex, 'date' column holds timestamps."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100 + np.linspace(0, 5, n)
    df = pd.DataFrame({
        "date": idx,
        "open": close - 0.1, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": np.full(n, 1000.0),
    })
    assert isinstance(df.index, pd.RangeIndex)
    return df


def _daily_close_series(start: str, vals: list[float]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(vals), freq="1D", tz="UTC")
    return pd.DataFrame({
        "open": vals, "high": vals, "low": vals, "close": vals, "volume": vals,
    }, index=idx)


@pytest.fixture
def mock_external_loader(monkeypatch):
    from indicators import external_data as ed
    from indicators import fear_and_greed as fag
    from indicators import perp_metrics as pm
    from indicators import alt_strength as als

    store: dict[str, pd.DataFrame] = {}

    def fake_loader(pair, timeframe="1d"):
        return store.get(pair, pd.DataFrame()).copy()

    for mod in (ed, fag, pm, als):
        monkeypatch.setattr(mod, "load_external_dataframe", fake_loader)
    return store


# ---------------------------------------------------------------------------
# resolve_target_index — the helper that powers the fix
# ---------------------------------------------------------------------------

def test_resolve_target_index_passes_through_datetime_index():
    from indicators.fear_and_greed import resolve_target_index
    df = _datetime_indexed_ohlcv()
    out = resolve_target_index(df)
    assert isinstance(out, pd.DatetimeIndex)
    assert out.equals(df.index)


def test_resolve_target_index_extracts_date_column_when_rangeindex():
    from indicators.fear_and_greed import resolve_target_index
    df = _freqtrade_shape_ohlcv()
    out = resolve_target_index(df)
    assert isinstance(out, pd.DatetimeIndex)
    # Order preserved, same timestamps
    assert (out == pd.DatetimeIndex(df["date"])).all()


def test_resolve_target_index_falls_through_when_no_signal():
    """No DatetimeIndex, no 'date' column — return the index as-is. Callers
    handle the failure in their existing try/except."""
    from indicators.fear_and_greed import resolve_target_index
    df = pd.DataFrame({"close": [1.0, 2.0]})
    out = resolve_target_index(df)
    assert out is df.index


# ---------------------------------------------------------------------------
# vix / gold / dxy / spx via _attach_external_close (external_data.py)
# ---------------------------------------------------------------------------

def test_vix_populates_for_freqtrade_shape(mock_external_loader):
    from indicators.external_data import add_external_data
    n_days = 200
    mock_external_loader["VIX/USDT"] = _daily_close_series(
        "2026-01-01", [10.0 + d for d in range(n_days)]
    )
    df = _freqtrade_shape_ohlcv(n=240, start="2026-04-04 00:00")
    out = add_external_data(df)

    # Regression: this was 0% before the fix.
    assert out["vix"].notna().sum() > 200, "vix must populate on RangeIndex+date dataframes"


def test_vix_alignment_matches_across_index_shapes(mock_external_loader):
    """Same VIX series, same hourly window — DatetimeIndex and RangeIndex+date
    must produce identical values. If they don't, look-ahead behavior differs
    between manual analysis and live backtests."""
    from indicators.external_data import add_external_data
    n_days = 200
    mock_external_loader["VIX/USDT"] = _daily_close_series(
        "2026-01-01", [10.0 + d for d in range(n_days)]
    )

    df_dt = _datetime_indexed_ohlcv(n=240, start="2026-04-04 00:00")
    df_ft = _freqtrade_shape_ohlcv(n=240, start="2026-04-04 00:00")

    out_dt = add_external_data(df_dt)
    out_ft = add_external_data(df_ft)

    np.testing.assert_array_equal(out_dt["vix"].values, out_ft["vix"].values)


def test_vix_no_lookahead_on_freqtrade_shape(mock_external_loader):
    """The look-ahead guarantee (T sees T-1's daily close) must survive the fix."""
    from indicators.external_data import add_external_data
    n_days = 200
    mock_external_loader["VIX/USDT"] = _daily_close_series(
        "2026-01-01", [10.0 + d for d in range(n_days)]
    )
    df = _freqtrade_shape_ohlcv(n=240, start="2026-04-04 00:00")
    out = add_external_data(df)

    # Apr 4 = day index 93 → VIX[93] = 103.0. Shifted by 1 → Apr 4 sees VIX[92] = 102.0.
    apr4_noon = out[out["date"] == pd.Timestamp("2026-04-04 12:00+00:00")]
    apr5_midnight = out[out["date"] == pd.Timestamp("2026-04-05 00:00+00:00")]
    assert apr4_noon["vix"].iloc[0] == 102.0
    assert apr5_midnight["vix"].iloc[0] == 103.0


# ---------------------------------------------------------------------------
# btc_funding_rate / btc_oi via _join_external_metric (perp_metrics.py)
# ---------------------------------------------------------------------------

def test_btc_funding_populates_for_freqtrade_shape(mock_external_loader):
    from indicators.external_data import add_external_data
    # 200 8h funding rate samples = ~67 days of coverage
    n = 200
    mock_external_loader["BTCFUND/USDT"] = pd.DataFrame({
        "open": np.linspace(0.0001, 0.0005, n),
        "high": np.linspace(0.0001, 0.0005, n),
        "low":  np.linspace(0.0001, 0.0005, n),
        "close": np.linspace(0.0001, 0.0005, n),
        "volume": np.full(n, 1.0),
    }, index=pd.date_range("2026-01-01", periods=n, freq="8h", tz="UTC"))

    df = _freqtrade_shape_ohlcv(n=240, start="2026-04-04 00:00")
    out = add_external_data(df)

    # Regression: this was 0 before the fix.
    assert out["btc_funding_rate"].notna().sum() > 200


def test_btc_oi_populates_for_freqtrade_shape(mock_external_loader):
    from indicators.external_data import add_external_data
    n = 200
    mock_external_loader["BTCOI/USDT"] = _daily_close_series(
        "2026-01-01", [1e9 + d * 1e6 for d in range(n)]
    )
    df = _freqtrade_shape_ohlcv(n=240, start="2026-04-04 00:00")
    out = add_external_data(df)

    assert out["btc_oi"].notna().sum() > 200
    # 24h pct change derives from the same series — must also populate.
    assert out["btc_oi_pct_change_24h"].notna().sum() > 100


# ---------------------------------------------------------------------------
# eth_btc_ratio / change_7d / zscore via add_alt_strength
# ---------------------------------------------------------------------------

def test_alt_strength_populates_for_freqtrade_shape(mock_external_loader):
    from indicators.external_data import add_external_data
    n = 200
    mock_external_loader["ETH/USDT"] = _daily_close_series(
        "2026-01-01", [2000.0 + d * 5 for d in range(n)]
    )
    mock_external_loader["BTC/USDT"] = _daily_close_series(
        "2026-01-01", [40000.0 + d * 50 for d in range(n)]
    )

    df = _freqtrade_shape_ohlcv(n=240, start="2026-04-04 00:00")
    out = add_external_data(df)

    assert out["eth_btc_ratio"].notna().sum() > 200
    assert out["eth_btc_change_7d"].notna().sum() > 100
    # z-score needs 10+ days; should populate over most of the window
    assert out["alt_strength_zscore_30d"].notna().sum() > 100
