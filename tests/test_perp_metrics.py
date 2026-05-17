"""Tests for R2b: BTC perpetual signals (funding rate + open interest)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))
sys.path.insert(0, str(ROOT / "user_data"))


def _hourly_ohlcv(n: int = 250, start: str = "2026-04-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100 + np.linspace(0, 5, n)
    return pd.DataFrame({
        "open":   close - 0.1,
        "high":   close + 0.3,
        "low":    close - 0.3,
        "close":  close,
        "volume": np.full(n, 1000.0),
    }, index=idx)


def _series_to_close_df(start: str, vals: list, freq: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(vals), freq=freq, tz="UTC")
    return pd.DataFrame({
        "open": vals, "high": vals, "low": vals, "close": vals, "volume": vals,
    }, index=idx)


@pytest.fixture
def mock_loader(monkeypatch):
    from indicators import perp_metrics as pm

    store: dict[str, pd.DataFrame] = {}

    def fake_loader(pair: str, timeframe: str = "1d") -> pd.DataFrame:
        return store.get(pair, pd.DataFrame()).copy()

    monkeypatch.setattr(pm, "load_external_dataframe", fake_loader)
    return store


# ---------------------------------------------------------------------------
# Column attachment + idempotency
# ---------------------------------------------------------------------------

def test_add_perp_metrics_adds_columns(mock_loader):
    from indicators.perp_metrics import add_perp_metrics

    df = _hourly_ohlcv()
    out = add_perp_metrics(df)
    for col in ("btc_funding_rate", "btc_oi", "btc_oi_pct_change_24h"):
        assert col in out.columns


def test_add_perp_metrics_missing_files_no_crash(mock_loader):
    """Empty store → all three columns become NaN, no exception."""
    from indicators.perp_metrics import add_perp_metrics

    df = _hourly_ohlcv()
    out = add_perp_metrics(df)
    assert out["btc_funding_rate"].isna().all()
    assert out["btc_oi"].isna().all()
    assert out["btc_oi_pct_change_24h"].isna().all()


def test_add_perp_metrics_is_idempotent(mock_loader):
    from indicators.perp_metrics import add_perp_metrics

    mock_loader["BTCFUND/USDT"] = _series_to_close_df(
        "2026-03-01 00:00", [0.0001, 0.0002, 0.00015] * 100, freq="8h"
    )
    df = _hourly_ohlcv(n=250, start="2026-04-01 00:00")
    out1 = add_perp_metrics(df)
    out2 = add_perp_metrics(out1)
    pd.testing.assert_series_equal(out1["btc_funding_rate"], out2["btc_funding_rate"])


# ---------------------------------------------------------------------------
# Look-ahead protection
# ---------------------------------------------------------------------------

def test_funding_rate_no_lookahead(mock_loader):
    """A bar at hour H within an 8h window must see the rate from the
    PREVIOUS settlement, not the one currently accruing.

    Settlements at 00:00, 08:00, 16:00 UTC. We mock four sequential rates.
    A bar at 09:00 UTC is inside the 08:00-16:00 window — the 08:00 rate
    is NOT yet settled, so the bar must see the 00:00 rate.
    """
    from indicators.perp_metrics import add_perp_metrics

    # 30 days of 8h funding settlements starting Mar 1 — distinct values
    n = 90
    rates = [0.0001 * i for i in range(n)]
    mock_loader["BTCFUND/USDT"] = _series_to_close_df(
        "2026-03-01 00:00", rates, freq="8h"
    )

    # Build strategy frame: one bar inside each 8h window on Apr 4
    df = pd.DataFrame(
        {"open": [1] * 3, "high": [1] * 3, "low": [1] * 3, "close": [1] * 3, "volume": [1] * 3},
        index=pd.to_datetime([
            "2026-04-04 03:00",  # inside 00:00-08:00 window → should see Apr 3 16:00 settlement
            "2026-04-04 09:00",  # inside 08:00-16:00 window → should see Apr 4 00:00 settlement
            "2026-04-04 17:00",  # inside 16:00-24:00 window → should see Apr 4 08:00 settlement
        ], utc=True),
    )

    out = add_perp_metrics(df)

    # Index of "2026-04-04 00:00" within the 8h series:
    # Mar 1 00:00 + i*8h = Apr 4 00:00 → i = (34 days * 3) = 102. But n=90 so
    # let's just compute: number of 8h settlements between Mar 1 00:00 and Apr 4 00:00 = 34*3 = 102.
    # That's outside our 90-rate range. Use a longer series.
    # Re-mock with more rates so Apr 4 indices are real.
    rates_long = [0.0001 * i for i in range(200)]
    mock_loader["BTCFUND/USDT"] = _series_to_close_df(
        "2026-03-01 00:00", rates_long, freq="8h"
    )
    out = add_perp_metrics(df)

    # Apr 4 00:00 = settlement index 102 → rate = 0.0102
    # Apr 4 08:00 = index 103 → rate = 0.0103
    # Apr 3 16:00 = index 101 → rate = 0.0101
    # With +8h shift: source ts X is observable from X+8h onward.
    # Bar at 03:00 sees: latest source ts where (source_ts + 8h) <= 03:00
    #   → source_ts <= Apr 3 19:00. Latest 8h settlement on/before that is Apr 3 16:00 (index 101).
    assert out.loc["2026-04-04 03:00+00:00", "btc_funding_rate"] == pytest.approx(0.0101)
    # Bar at 09:00 sees source_ts <= Apr 4 01:00 → Apr 4 00:00 (index 102).
    assert out.loc["2026-04-04 09:00+00:00", "btc_funding_rate"] == pytest.approx(0.0102)
    # Bar at 17:00 sees source_ts <= Apr 4 09:00 → Apr 4 08:00 (index 103).
    assert out.loc["2026-04-04 17:00+00:00", "btc_funding_rate"] == pytest.approx(0.0103)


def test_oi_no_lookahead(mock_loader):
    """Daily OI snapshot must be shifted by 1 day before joining."""
    from indicators.perp_metrics import add_perp_metrics

    # Daily OI: each day's value = 1_000_000 * day_index
    n = 30
    vals = [1_000_000.0 * i for i in range(n)]
    mock_loader["BTCOI/USDT"] = _series_to_close_df(
        "2026-04-01 00:00", vals, freq="1D"
    )

    df = pd.DataFrame(
        {"open": [1] * 2, "high": [1] * 2, "low": [1] * 2, "close": [1] * 2, "volume": [1] * 2},
        index=pd.to_datetime([
            "2026-04-05 12:00",  # day 4 of the source — should see day 3's value (3_000_000)
            "2026-04-06 03:00",  # day 5 — should see day 4 (4_000_000)
        ], utc=True),
    )

    out = add_perp_metrics(df)
    assert out.loc["2026-04-05 12:00+00:00", "btc_oi"] == 3_000_000.0
    assert out.loc["2026-04-06 03:00+00:00", "btc_oi"] == 4_000_000.0


# ---------------------------------------------------------------------------
# Prompt + package wiring
# ---------------------------------------------------------------------------

def test_prompt_documents_perp_columns():
    from strategy_generator import SYSTEM_PROMPT

    assert "CRYPTO POSITIONING" in SYSTEM_PROMPT
    for col in ("btc_funding_rate", "btc_oi", "btc_oi_pct_change_24h"):
        assert f"dataframe['{col}']" in SYSTEM_PROMPT, f"prompt missing {col}"


def test_indicators_package_exports_add_perp_metrics():
    import indicators
    assert "add_perp_metrics" in indicators.__all__


def test_add_external_data_includes_perp_columns(monkeypatch):
    """End-to-end: the single add_external_data entry point must produce
    the perp columns alongside fgi/vix/gold/dxy/spx."""
    from indicators import external_data as ed
    from indicators import fear_and_greed as fag
    from indicators import perp_metrics as pm

    empty = pd.DataFrame()
    monkeypatch.setattr(ed, "load_external_dataframe", lambda *a, **k: empty)
    monkeypatch.setattr(fag, "load_external_dataframe", lambda *a, **k: empty)
    monkeypatch.setattr(pm, "load_external_dataframe", lambda *a, **k: empty)

    from indicators.external_data import add_external_data

    df = _hourly_ohlcv()
    out = add_external_data(df)
    for col in ("fgi", "vix", "gold", "dxy", "spx",
                "btc_funding_rate", "btc_oi", "btc_oi_pct_change_24h"):
        assert col in out.columns, f"missing {col} after add_external_data"
