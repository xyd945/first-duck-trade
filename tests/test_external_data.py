"""Tests for R2a: external data injection into generated strategies."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))
sys.path.insert(0, str(ROOT / "user_data"))


# ---------------------------------------------------------------------------
# add_external_data
# ---------------------------------------------------------------------------

def _hourly_ohlcv(n: int = 250, start: str = "2026-04-01 00:00") -> pd.DataFrame:
    """Default length >=200 so fear_and_greed's slow EMA (length=144) has data."""
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    # Use a gentle uptrend so EMA/RoR have something to compute (not flat)
    close = 100 + np.linspace(0, 5, n)
    return pd.DataFrame({
        "open":   close - 0.1,
        "high":   close + 0.3,
        "low":    close - 0.3,
        "close":  close,
        "volume": np.full(n, 1000.0),
    }, index=idx)


def _daily_close_series(start: str, vals: list[float]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(vals), freq="1D", tz="UTC")
    return pd.DataFrame({
        "open": vals, "high": vals, "low": vals, "close": vals, "volume": vals,
    }, index=idx)


@pytest.fixture
def mock_external_loader(monkeypatch):
    """Replace load_external_dataframe with an in-memory mock."""
    from indicators import external_data as ed

    store: dict[str, pd.DataFrame] = {}

    def fake_loader(pair: str, timeframe: str = "1d") -> pd.DataFrame:
        return store.get(pair, pd.DataFrame()).copy()

    monkeypatch.setattr(ed, "load_external_dataframe", fake_loader)
    # add_fear_and_greed uses its own import; patch that too
    from indicators import fear_and_greed as fag
    monkeypatch.setattr(fag, "load_external_dataframe", fake_loader)
    return store


def test_add_external_data_adds_all_columns(mock_external_loader):
    from indicators.external_data import add_external_data

    df = _hourly_ohlcv()
    out = add_external_data(df)

    for col in ("fgi", "vix", "gold", "dxy", "spx"):
        assert col in out.columns, f"missing column {col}"


def test_add_external_data_no_lookahead(mock_external_loader):
    """A 1h row at day T must reflect external close at day T-1, not T."""
    from indicators.external_data import add_external_data

    # 200 days of mock VIX so add_fear_and_greed's slow_length=144 EMA has
    # enough data. Values are start_value + day_index so each day is distinct.
    n_days = 200
    mock_external_loader["VIX/USDT"] = _daily_close_series(
        "2026-01-01", [10.0 + d for d in range(n_days)]
    )

    # Hourly strategy frame spanning a slice well inside the VIX history
    df = _hourly_ohlcv(n=240, start="2026-04-04 00:00")

    out = add_external_data(df)

    # Apr 4 = day index (Apr 4 - Jan 1) = 93 → VIX[93] = 103.0.
    # Shifted by 1 (yesterday's close) → Apr 4 sees VIX[92] = 102.0.
    assert out.loc["2026-04-04 12:00+00:00", "vix"] == 102.0
    assert out.loc["2026-04-04 23:00+00:00", "vix"] == 102.0
    # Apr 5 must see VIX[93] = 103.0
    assert out.loc["2026-04-05 00:00+00:00", "vix"] == 103.0
    assert out.loc["2026-04-05 12:00+00:00", "vix"] == 103.0


def test_add_external_data_missing_files_dont_crash(mock_external_loader):
    """Empty store means every loader returns an empty df — columns must be NA."""
    from indicators.external_data import add_external_data

    df = _hourly_ohlcv()
    out = add_external_data(df)

    # vix/gold/dxy/spx should be all-NaN; fgi from fear_and_greed falls back to 0
    assert out["vix"].isna().all()
    assert out["gold"].isna().all()
    assert out["dxy"].isna().all()
    assert out["spx"].isna().all()


def test_add_external_data_is_idempotent(mock_external_loader):
    """Calling twice should not error and should produce the same columns."""
    from indicators.external_data import add_external_data

    mock_external_loader["GOLD/USDT"] = _daily_close_series(
        "2026-04-01", [2000.0 + i for i in range(20)]
    )
    df = _hourly_ohlcv(n=250, start="2026-04-02 00:00")
    out1 = add_external_data(df)
    out2 = add_external_data(out1)
    pd.testing.assert_series_equal(out1["gold"], out2["gold"])


# ---------------------------------------------------------------------------
# Prompt integration
# ---------------------------------------------------------------------------

def test_system_prompt_documents_external_data():
    from strategy_generator import SYSTEM_PROMPT

    assert "EXTERNAL DATA" in SYSTEM_PROMPT
    # Every documented column must appear
    for col in ("fgi", "vix", "gold", "dxy", "spx"):
        assert f"dataframe['{col}']" in SYSTEM_PROMPT, f"prompt missing dataframe['{col}']"
    # The mandatory first-line instruction must be present
    assert "FIRST line of populate_indicators MUST be" in SYSTEM_PROMPT
    assert "dataframe = add_external_data(dataframe)" in SYSTEM_PROMPT
    # And the import line must appear in the CORRECT PATTERN example
    assert "from indicators.external_data import add_external_data" in SYSTEM_PROMPT


def test_indicators_package_exports_external_data():
    import indicators
    assert "add_external_data" in indicators.__all__
    assert "add_fear_and_greed" in indicators.__all__
