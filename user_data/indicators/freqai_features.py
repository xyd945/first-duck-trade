"""
FreqAI feature library — the ONLY place FreqAI feature computation lives.

FreqAI candidates (issue #47) are rendered from declarative specs: the
generated strategy file contains a feature-key list, thresholds, and risk
numbers — never computation code. Every feature a spec can request maps to
a function here, hand-written and reviewed once, so the "LLM proposes an ML
experiment" path can never smuggle in arbitrary Python or look-ahead bias.

Three feature groups, mirroring FreqAI's three expansion hooks:

  EXPAND features   period-parameterized; FreqAI calls the hook once per
                    period in feature_parameters.indicator_periods_candles
                    (and per timeframe / shifted candle) and suffixes the
                    column names itself. Column names here are constant
                    ("%-rsi-period" etc.) by FreqAI convention.
  BASIC features    single-shot price/volume transforms, no period arg.
  STANDARD features time-of-week cycle + the external macro columns that
                    add_external_data() already exposes to rule-based
                    strategies (funding, OI, ETH/BTC, FGI, VIX). All
                    external series are shifted forward at load time by
                    their source modules, so reusing them here inherits
                    the same anti-look-ahead guarantees.

NaN policy: forward-fill only, then fill remaining leading NaNs with 0.
Never backfill — bfill would leak future values into early rows. A source
file that's missing entirely produces a constant-0 column, which is inert
for tree models rather than fatal.

Pure pandas / pandas_ta — importable (and tested) in the orchestrator
container, which has no freqtrade installed.
"""

import pandas as pd


def _safe_fill(series: pd.Series) -> pd.Series:
    """ffill then zero-fill the leading gap. No bfill — see module docstring.

    ±inf (e.g. a zero rolling-std in a flat window dividing a z-score)
    becomes NaN first — LightGBM rejects inf inputs, and ffill would
    otherwise propagate it forward across the window.
    """
    import numpy as np
    # np.nan (not pd.NA) keeps the series float-dtyped for LightGBM
    return series.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# EXPAND features (period-parameterized)
# ---------------------------------------------------------------------------

def _f_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    import pandas_ta as ta
    return ta.rsi(df["close"], length=period)


def _f_ema_dist(df: pd.DataFrame, period: int) -> pd.Series:
    import pandas_ta as ta
    ema = ta.ema(df["close"], length=period)
    return (df["close"] - ema) / ema


def _f_natr(df: pd.DataFrame, period: int) -> pd.Series:
    import pandas_ta as ta
    return ta.natr(df["high"], df["low"], df["close"], length=period)


def _f_adx(df: pd.DataFrame, period: int) -> pd.Series:
    import pandas_ta as ta
    adx = ta.adx(df["high"], df["low"], df["close"], length=period)
    # pandas_ta returns a frame (or None on too-short input — _computed
    # in the caller raises the loud error); the ADX line is ADX_<period>
    return adx[f"ADX_{period}"] if adx is not None else None


def _f_bb_width(df: pd.DataFrame, period: int) -> pd.Series:
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return (4.0 * std) / mid  # (upper-lower)/mid at 2 sigma


def _f_roc(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].pct_change(periods=period)


def _f_volume_z(df: pd.DataFrame, period: int) -> pd.Series:
    mean = df["volume"].rolling(period).mean()
    std = df["volume"].rolling(period).std()
    return (df["volume"] - mean) / std


EXPAND_FEATURES = {
    "rsi": _f_rsi,
    "ema_dist": _f_ema_dist,
    "natr": _f_natr,
    "adx": _f_adx,
    "bb_width": _f_bb_width,
    "roc": _f_roc,
    "volume_z": _f_volume_z,
}


# ---------------------------------------------------------------------------
# BASIC features (no period)
# ---------------------------------------------------------------------------

def _f_pct_change(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change()


def _f_hl_range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]) / df["close"]


BASIC_FEATURES = {
    "pct_change": _f_pct_change,
    "hl_range": _f_hl_range,
}


# ---------------------------------------------------------------------------
# STANDARD features (time cycle + external macro columns)
# ---------------------------------------------------------------------------

# Spec feature key -> column produced by indicators.external_data.add_external_data
EXTERNAL_FEATURES = {
    "funding": "btc_funding_rate",
    "oi_change": "btc_oi_pct_change_24h",
    "eth_btc": "eth_btc_change_7d",
    "alt_strength": "alt_strength_zscore_30d",
    "macro_fgi": "fgi",
    "macro_vix": "vix",
}

TIME_FEATURES = ("time_cycle",)

ALL_FEATURE_KEYS = (
    tuple(EXPAND_FEATURES)
    + tuple(BASIC_FEATURES)
    + tuple(EXTERNAL_FEATURES)
    + TIME_FEATURES
)


def _computed(key: str, value, period=None) -> pd.Series:
    """Guard a feature function's output. pandas_ta returns None (not an
    empty series) when the input is shorter than the indicator period —
    silently zero-filling that would score a candidate on fabricated
    features, so fail loudly instead; the orchestrator turns the crash
    into a clean FAIL_BACKTEST retirement."""
    if value is None:
        detail = f" (period={period})" if period is not None else ""
        raise ValueError(
            f"feature {key!r} returned no data{detail} — "
            f"input window too short for the indicator"
        )
    return value


def add_expand_features(
    dataframe: pd.DataFrame, features: list[str], period: int
) -> pd.DataFrame:
    """Compute the selected period-parameterized features.

    Column names follow FreqAI's expand_all convention ("%-<key>-period");
    FreqAI itself appends the concrete period / timeframe / pair suffixes.
    Unknown keys are ignored here — the spec validator is the gate that
    rejects them loudly.
    """
    for key in features:
        fn = EXPAND_FEATURES.get(key)
        if fn is not None:
            dataframe[f"%-{key}-period"] = _safe_fill(
                _computed(key, fn(dataframe, period), period)
            )
    return dataframe


def add_basic_features(dataframe: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute the selected single-shot features ("%-<key>" columns)."""
    for key in features:
        fn = BASIC_FEATURES.get(key)
        if fn is not None:
            dataframe[f"%-{key}"] = _safe_fill(_computed(key, fn(dataframe)))
    return dataframe


def add_standard_features(dataframe: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute time-cycle and external macro features.

    External columns are injected via add_external_data (idempotent, safely
    shifted at source) and then copied under "%-ext-<key>" names so FreqAI
    treats them as model features. The raw columns stay available for any
    non-feature use.
    """
    wanted_external = [k for k in features if k in EXTERNAL_FEATURES]
    if wanted_external:
        from .external_data import add_external_data
        dataframe = add_external_data(dataframe)
        for key in wanted_external:
            col = EXTERNAL_FEATURES[key]
            if col in dataframe.columns:
                dataframe[f"%-ext-{key}"] = _safe_fill(
                    pd.to_numeric(dataframe[col], errors="coerce")
                )
            else:
                dataframe[f"%-ext-{key}"] = 0.0

    if "time_cycle" in features and "date" in dataframe.columns:
        import numpy as np
        dates = pd.to_datetime(dataframe["date"], utc=True)
        hour_of_week = dates.dt.dayofweek * 24 + dates.dt.hour
        dataframe["%-time-week-sin"] = np.sin(2 * np.pi * hour_of_week / 168.0)
        dataframe["%-time-week-cos"] = np.cos(2 * np.pi * hour_of_week / 168.0)

    return dataframe


def add_future_return_target(
    dataframe: pd.DataFrame, horizon_candles: int, column: str = "&-future_return"
) -> pd.DataFrame:
    """Label: forward return over `horizon_candles`.

    THE one legitimate use of shift(-N) in the FreqAI path — targets must
    look forward; FreqAI trims the unlabeled tail before training. Features
    must never call this. Kept here (trusted, hand-written module) so
    rendered candidate files contain no shift(-N) and stay rejectable by
    the look-ahead validator if a template bug ever leaks one in.
    """
    dataframe[column] = (
        dataframe["close"].shift(-horizon_candles) / dataframe["close"] - 1.0
    )
    return dataframe
