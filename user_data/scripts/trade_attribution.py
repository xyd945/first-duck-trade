"""
R2d: per-trade attribution.

For every closed trade in a backtest, look up the macro context that was
in effect at entry time (fgi, vix, funding, oi-momentum, alt-strength),
bucket those values, and aggregate win-rate per bucket. The result tells
us which macro conditions actually favor wins vs losses for a given
strategy — feedback the reflector and the LLM generator can use to
sharpen the next round of strategies.

Why bucket macro context rather than the strategy's own entry conditions:
- the strategies generated from spec don't persist their exact entry
  predicate set in a machine-readable way once rendered
- macro context is strategy-agnostic — same buckets apply across the
  pool so we can compare attribution across candidates
- the failure memory loop already shows the LLM that "Strategy X lost
  money"; attribution sharpens that to "Strategy X loses when funding
  is frothy AND alt-strength is in capitulation"

Bucket definitions are deliberate, not data-driven:
  fgi                       fear / neutral / greed
  vix                       low (<18) / mid / high (>25)
  btc_funding_rate          shorts_pay (<0) / neutral / frothy (>3bp/8h)
  btc_oi_pct_change_24h     deleverage (<-5%) / stable / building (>5%)
  alt_strength_zscore_30d   btc_dominant (<-1) / neutral / alt_season (>1)

"Lift" = bucket win-rate minus overall win-rate. Positive lift = this
bucket is over-represented in wins. We only report buckets with at least
min(3, 20% of trades) samples to avoid celebrating noise from one-off
buckets.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

log = logging.getLogger("trade_attribution")

# Each entry: (upper_bound, label). Last bucket should have upper=inf.
BUCKETS: dict[str, list[tuple[float, str]]] = {
    "fgi": [(-10, "fear"), (10, "neutral"), (float("inf"), "greed")],
    "vix": [(18, "low"), (25, "mid"), (float("inf"), "high")],
    "btc_funding_rate": [(0, "shorts_pay"), (0.0003, "neutral"), (float("inf"), "frothy")],
    "btc_oi_pct_change_24h": [(-5, "deleverage"), (5, "stable"), (float("inf"), "building")],
    "alt_strength_zscore_30d": [(-1, "btc_dominant"), (1, "neutral"), (float("inf"), "alt_season")],
}

ATTRIBUTED_COLUMNS = tuple(BUCKETS.keys())


def bucket_value(name: str, value) -> str | None:
    """Map a numeric macro value to its bucket label, e.g. 'vix_high'.
    Returns None for unknown signals or NaN."""
    if value is None or pd.isna(value):
        return None
    cuts = BUCKETS.get(name)
    if not cuts:
        return None
    for upper, label in cuts:
        if value < upper:
            return f"{name}_{label}"
    return None


# ---------------------------------------------------------------------------
# Freqtrade trade-export loader
# ---------------------------------------------------------------------------

def load_trades_from_zip(zip_path: str | Path, strategy_name: str) -> list[dict]:
    """Read a strategy's trade list from Freqtrade's exported .zip.

    Freqtrade writes a zip containing several files; the main one is the
    JSON not ending in `_config.json`. Inside that JSON, trades live at
    `strategy.<StrategyName>.trades`.

    Returns [] on any error (missing file, malformed zip, etc.).
    """
    p = Path(zip_path)
    if not p.exists():
        log.warning(f"trade export missing: {p}")
        return []
    try:
        with zipfile.ZipFile(p) as zf:
            # Take the first .json that isn't the config sidecar
            names = [n for n in zf.namelist()
                     if n.endswith(".json") and "_config" not in n]
            if not names:
                return []
            with zf.open(names[0]) as f:
                data = json.load(f)
    except (zipfile.BadZipFile, json.JSONDecodeError, OSError) as e:
        log.warning(f"could not read trade export {p}: {e}")
        return []

    strat = data.get("strategy", {})
    if strategy_name in strat:
        return strat[strategy_name].get("trades", []) or []
    # Fallback: single-strategy export with different naming
    vals = list(strat.values())
    return vals[0].get("trades", []) if vals else []


# ---------------------------------------------------------------------------
# Macro context snapshot builder
# ---------------------------------------------------------------------------

def build_macro_snapshots() -> pd.DataFrame:
    """Daily dataframe of macro signals indexed by date (UTC).

    Reads the same external data files the strategies consume and runs
    them through `add_external_data` on a daily-cadence BTC dataframe.
    Returns only the columns we attribute on (see ATTRIBUTED_COLUMNS).
    Empty dataframe if the source BTC daily file is missing.
    """
    import sys
    indicators_path = Path(__file__).resolve().parent.parent
    if str(indicators_path) not in sys.path:
        sys.path.insert(0, str(indicators_path))

    from indicators.fear_and_greed import load_external_dataframe
    from indicators.external_data import add_external_data

    btc = load_external_dataframe("BTC/USDT", "1d")
    if btc.empty:
        log.warning("no BTC daily data — attribution macro context unavailable")
        return pd.DataFrame()

    df = btc.copy()
    try:
        df = add_external_data(df)
    except Exception as e:
        log.warning(f"add_external_data failed during macro snapshot build: {e}")
        return pd.DataFrame()

    keep = [c for c in ATTRIBUTED_COLUMNS if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Core attribution
# ---------------------------------------------------------------------------

def attribute_trades(
    trades: list[dict],
    macro_df: pd.DataFrame,
    min_bucket_sample: int = 3,
) -> dict:
    """Bucket each trade's entry-time macro context and aggregate win rates.

    Returns:
      {
        "total_trades": int,
        "overall_win_rate": float (0-1),
        "buckets": {bucket_label: {"trades": N, "wins": K,
                                    "win_rate": float, "lift": float}},
        "top_positive_lift": [bucket_label, ...],   # up to 3, eligible only
        "top_negative_lift": [bucket_label, ...],   # up to 3, eligible only
      }

    Eligibility for top-N reporting: bucket must have >=
    max(min_bucket_sample, total_trades // 5) samples — avoids loud
    one-trade buckets dominating the ranking.
    """
    total = len(trades)
    if total == 0 or macro_df.empty:
        return {
            "total_trades": total,
            "overall_win_rate": 0.0,
            "buckets": {},
            "top_positive_lift": [],
            "top_negative_lift": [],
        }

    bucket_stats: dict = defaultdict(lambda: {"trades": 0, "wins": 0})
    overall_wins = 0

    for t in trades:
        try:
            entry_time = pd.to_datetime(t["open_date"], utc=True)
        except (KeyError, ValueError):
            continue
        won = float(t.get("profit_ratio", 0)) > 0
        if won:
            overall_wins += 1

        # Find the latest macro snapshot at or before entry time
        mask = macro_df.index <= entry_time
        if not mask.any():
            continue
        row = macro_df[mask].iloc[-1]

        for name in macro_df.columns:
            b = bucket_value(name, row[name])
            if b is None:
                continue
            bucket_stats[b]["trades"] += 1
            if won:
                bucket_stats[b]["wins"] += 1

    overall_wr = overall_wins / total
    buckets = {}
    for b, s in bucket_stats.items():
        wr = s["wins"] / s["trades"] if s["trades"] else 0.0
        buckets[b] = {
            "trades": s["trades"],
            "wins": s["wins"],
            "win_rate": round(wr, 3),
            "lift": round(wr - overall_wr, 3),
        }

    min_sample = max(min_bucket_sample, total // 5)
    eligible = [(b, s) for b, s in buckets.items() if s["trades"] >= min_sample]
    # Sign-filter so the same bucket can't appear in both lists (e.g. a
    # bucket with lift +0.02 isn't a "loser" just because it ranks third
    # from the bottom).
    top_pos = [b for b, s in sorted(eligible, key=lambda x: x[1]["lift"], reverse=True)
               if s["lift"] > 0][:3]
    top_neg = [b for b, s in sorted(eligible, key=lambda x: x[1]["lift"])
               if s["lift"] < 0][:3]

    return {
        "total_trades": total,
        "overall_win_rate": round(overall_wr, 3),
        "buckets": buckets,
        "top_positive_lift": top_pos,
        "top_negative_lift": top_neg,
    }


def summarize_attribution(attr: dict) -> str:
    """Human-readable rendering for logs and the reflector prompt."""
    total = attr.get("total_trades", 0)
    if total == 0:
        return "No trades to attribute."

    overall_wr = attr.get("overall_win_rate", 0)
    lines = [f"Trades: {total}, overall win rate: {overall_wr:.0%}"]

    if attr.get("top_positive_lift"):
        lines.append("Conditions favoring WINS:")
        for b in attr["top_positive_lift"]:
            s = attr["buckets"][b]
            lines.append(
                f"  {b}: {s['win_rate']:.0%} "
                f"({s['wins']}/{s['trades']}, lift {s['lift']:+.2f})"
            )

    if attr.get("top_negative_lift"):
        lines.append("Conditions favoring LOSSES:")
        for b in attr["top_negative_lift"]:
            s = attr["buckets"][b]
            lines.append(
                f"  {b}: {s['win_rate']:.0%} "
                f"({s['wins']}/{s['trades']}, lift {s['lift']:+.2f})"
            )

    return "\n".join(lines)
