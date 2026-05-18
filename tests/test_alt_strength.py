"""Tests for R2c: ETH/BTC ratio as a BTC-dominance proxy."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data"))
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def test_fetch_eth_btc_formats_klines_as_freqtrade_ohlcv():
    """Each row must be [ts_ms (int), o, h, l, c, v (floats)] — Freqtrade format."""
    import fetch_eth_btc

    fake_response = [
        # Binance kline format: 12 fields, we keep first 6
        [1735689600000, "3000.5", "3050.1", "2980.0", "3020.7", "12345.6",
         1735775999999, "37200000", 1500, "6000", "18000000", "0"],
        [1735776000000, "3020.7", "3100.0", "3010.0", "3080.2", "13456.7",
         1735862399999, "41000000", 1600, "6500", "20000000", "0"],
    ]

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return fake_response

    with patch("fetch_eth_btc.requests.get", return_value=FakeResp()):
        rows = fetch_eth_btc.fetch_klines("ETHUSDT")

    assert len(rows) == 2
    assert isinstance(rows[0][0], int)
    assert all(isinstance(x, float) for x in rows[0][1:])
    assert rows[0] == [1735689600000, 3000.5, 3050.1, 2980.0, 3020.7, 12345.6]


def test_fetch_eth_btc_writes_both_pairs(tmp_path, monkeypatch):
    """main() must write ETH_USDT-1d.json and BTC_USDT-1d.json."""
    import fetch_eth_btc

    monkeypatch.setattr(fetch_eth_btc, "OUT_DIR", tmp_path)
    call_log = []

    def fake_fetch(symbol, interval="1d", limit=1000):
        call_log.append(symbol)
        return [[1735689600000, 1.0, 1.0, 1.0, 1.0, 1.0]]

    monkeypatch.setattr(fetch_eth_btc, "fetch_klines", fake_fetch)
    monkeypatch.setattr(fetch_eth_btc.time, "sleep", lambda *_a, **_kw: None)

    rc = fetch_eth_btc.main()
    assert rc == 0
    assert call_log == ["ETHUSDT", "BTCUSDT"]
    assert (tmp_path / "ETH_USDT-1d.json").exists()
    assert (tmp_path / "BTC_USDT-1d.json").exists()


def test_fetch_eth_btc_partial_failure_returns_nonzero(tmp_path, monkeypatch):
    """If one of the two pairs fails, main() should report non-zero rc but
    still try the other pair."""
    import fetch_eth_btc

    monkeypatch.setattr(fetch_eth_btc, "OUT_DIR", tmp_path)
    monkeypatch.setattr(fetch_eth_btc.time, "sleep", lambda *_a, **_kw: None)

    def flaky_fetch(symbol, interval="1d", limit=1000):
        if symbol == "ETHUSDT":
            raise RuntimeError("network blip")
        return [[1735689600000, 1.0, 1.0, 1.0, 1.0, 1.0]]

    monkeypatch.setattr(fetch_eth_btc, "fetch_klines", flaky_fetch)
    rc = fetch_eth_btc.main()
    assert rc == 1
    assert (tmp_path / "BTC_USDT-1d.json").exists()
    assert not (tmp_path / "ETH_USDT-1d.json").exists()


# ---------------------------------------------------------------------------
# Indicator
# ---------------------------------------------------------------------------

def _make_synthetic_pair_df(prices: list, start: str = "2025-01-01") -> pd.DataFrame:
    """Build a fear_and_greed.load_external_dataframe-shaped df from a price list."""
    dates = pd.date_range(start, periods=len(prices), freq="1D", tz="UTC")
    df = pd.DataFrame({
        "open": prices, "high": prices, "low": prices,
        "close": prices, "volume": [1.0] * len(prices),
    }, index=dates)
    df.index.name = "date"
    return df


def test_alt_strength_computes_ratio_from_eth_and_btc():
    """eth_btc_ratio should equal ETH close / BTC close (with 1d look-ahead shift)."""
    from indicators.alt_strength import add_alt_strength

    eth_prices = [3000.0] * 40
    btc_prices = [60000.0] * 40
    eth_df = _make_synthetic_pair_df(eth_prices)
    btc_df = _make_synthetic_pair_df(btc_prices)

    # Strategy bar at hourly cadence over the same dates
    strategy_idx = pd.date_range("2025-01-15", periods=24, freq="1h", tz="UTC")
    dataframe = pd.DataFrame(index=strategy_idx)

    with patch("indicators.alt_strength.load_external_dataframe") as mock_load:
        mock_load.side_effect = lambda pair, tf: eth_df if pair == "ETH/USDT" else btc_df
        out = add_alt_strength(dataframe)

    assert "eth_btc_ratio" in out.columns
    assert "eth_btc_change_7d" in out.columns
    assert "alt_strength_zscore_30d" in out.columns
    # Constant prices → ratio = 0.05, change_7d = 0, zscore undefined (std=0 → NaN)
    assert out["eth_btc_ratio"].iloc[-1] == pytest.approx(0.05)
    assert out["eth_btc_change_7d"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert pd.isna(out["alt_strength_zscore_30d"].iloc[-1])


def test_alt_strength_detects_rising_ratio():
    """Rising ETH while BTC stays flat → ratio rises → positive 7d change + positive z-score."""
    from indicators.alt_strength import add_alt_strength

    eth_prices = list(np.linspace(3000.0, 3500.0, 40))  # +16.7% over 40 days
    btc_prices = [60000.0] * 40
    eth_df = _make_synthetic_pair_df(eth_prices)
    btc_df = _make_synthetic_pair_df(btc_prices)

    strategy_idx = pd.date_range("2025-02-05", periods=24, freq="1h", tz="UTC")
    dataframe = pd.DataFrame(index=strategy_idx)

    with patch("indicators.alt_strength.load_external_dataframe") as mock_load:
        mock_load.side_effect = lambda pair, tf: eth_df if pair == "ETH/USDT" else btc_df
        out = add_alt_strength(dataframe)

    assert out["eth_btc_ratio"].iloc[-1] > 0.05  # final ratio above starting
    assert out["eth_btc_change_7d"].iloc[-1] > 0  # 7d change positive
    assert out["alt_strength_zscore_30d"].iloc[-1] > 0  # rising → positive z


def test_alt_strength_detects_falling_ratio():
    """Falling ETH/BTC ratio → negative 7d change + negative z-score (BTC dominance rising)."""
    from indicators.alt_strength import add_alt_strength

    eth_prices = list(np.linspace(3000.0, 2500.0, 40))  # ETH down 16.7%
    btc_prices = [60000.0] * 40
    eth_df = _make_synthetic_pair_df(eth_prices)
    btc_df = _make_synthetic_pair_df(btc_prices)

    strategy_idx = pd.date_range("2025-02-05", periods=24, freq="1h", tz="UTC")
    dataframe = pd.DataFrame(index=strategy_idx)

    with patch("indicators.alt_strength.load_external_dataframe") as mock_load:
        mock_load.side_effect = lambda pair, tf: eth_df if pair == "ETH/USDT" else btc_df
        out = add_alt_strength(dataframe)

    assert out["eth_btc_ratio"].iloc[-1] < 0.05
    assert out["eth_btc_change_7d"].iloc[-1] < 0
    assert out["alt_strength_zscore_30d"].iloc[-1] < 0


def test_alt_strength_handles_missing_data():
    """Missing source files → NaN columns, no exception."""
    from indicators.alt_strength import add_alt_strength

    dataframe = pd.DataFrame(index=pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC"))

    with patch("indicators.alt_strength.load_external_dataframe", return_value=pd.DataFrame()):
        out = add_alt_strength(dataframe)

    assert "eth_btc_ratio" in out.columns
    assert out["eth_btc_ratio"].isna().all()
    assert out["eth_btc_change_7d"].isna().all()
    assert out["alt_strength_zscore_30d"].isna().all()


def test_alt_strength_handles_one_missing_pair():
    """If ETH or BTC alone is missing, the indicator must still degrade gracefully."""
    from indicators.alt_strength import add_alt_strength

    eth_df = _make_synthetic_pair_df([3000.0] * 30)
    dataframe = pd.DataFrame(index=pd.date_range("2025-01-15", periods=10, freq="1h", tz="UTC"))

    def loader(pair, tf):
        return eth_df if pair == "ETH/USDT" else pd.DataFrame()

    with patch("indicators.alt_strength.load_external_dataframe", side_effect=loader):
        out = add_alt_strength(dataframe)

    assert out["eth_btc_ratio"].isna().all()


def test_alt_strength_avoids_lookahead():
    """Bar at day T must see ratio from day T-1, not day T (1-day shift)."""
    from indicators.alt_strength import add_alt_strength

    # Two distinct closes on consecutive days so we can verify which one wins
    dates = pd.date_range("2025-01-01", periods=40, freq="1D", tz="UTC")
    eth_close = np.full(40, 3000.0)
    eth_close[-1] = 9999.0  # spike on last day — should NOT leak into the bar at that day
    btc_close = np.full(40, 60000.0)

    eth_df = pd.DataFrame({"open": eth_close, "high": eth_close, "low": eth_close,
                            "close": eth_close, "volume": 1.0}, index=dates)
    btc_df = pd.DataFrame({"open": btc_close, "high": btc_close, "low": btc_close,
                            "close": btc_close, "volume": 1.0}, index=dates)
    eth_df.index.name = "date"
    btc_df.index.name = "date"

    # Strategy bar on the same day as the spike
    spike_day = dates[-1]
    strategy_idx = pd.DatetimeIndex([spike_day])
    dataframe = pd.DataFrame(index=strategy_idx)

    with patch("indicators.alt_strength.load_external_dataframe") as mock_load:
        mock_load.side_effect = lambda pair, tf: eth_df if pair == "ETH/USDT" else btc_df
        out = add_alt_strength(dataframe)

    # The bar on the spike day must reflect the PRE-spike ratio (0.05), not 0.166
    assert out["eth_btc_ratio"].iloc[0] == pytest.approx(0.05)


def test_alt_strength_handles_btc_zero_close():
    """Zero BTC close (data glitch) must not crash with div-by-zero."""
    from indicators.alt_strength import add_alt_strength

    eth_df = _make_synthetic_pair_df([3000.0] * 30)
    btc_prices = [60000.0] * 30
    btc_prices[10] = 0.0  # one bad row
    btc_df = _make_synthetic_pair_df(btc_prices)

    dataframe = pd.DataFrame(index=pd.date_range("2025-01-20", periods=24, freq="1h", tz="UTC"))

    with patch("indicators.alt_strength.load_external_dataframe") as mock_load:
        mock_load.side_effect = lambda pair, tf: eth_df if pair == "ETH/USDT" else btc_df
        out = add_alt_strength(dataframe)

    # Last value should still be a sensible ratio, not inf/NaN
    assert np.isfinite(out["eth_btc_ratio"].iloc[-1])
    assert out["eth_btc_ratio"].iloc[-1] == pytest.approx(0.05)


def test_external_data_includes_alt_strength_columns():
    """The umbrella add_external_data must include the new columns."""
    from indicators.external_data import add_external_data

    df = pd.DataFrame({
        "open": [100.0] * 30, "high": [100.0] * 30,
        "low": [100.0] * 30, "close": [100.0] * 30,
        "volume": [1000.0] * 30,
    }, index=pd.date_range("2025-01-01", periods=30, freq="1h", tz="UTC"))

    out = add_external_data(df)
    assert "eth_btc_ratio" in out.columns
    assert "eth_btc_change_7d" in out.columns
    assert "alt_strength_zscore_30d" in out.columns
