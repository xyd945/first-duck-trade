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


def format_attributions_for_reflector(
    attributions: list[dict],
    max_chars: int = 2500,
) -> str:
    """Render multiple strategies' attribution as a compact reflector-prompt
    section.

    Each input row is a `get_recent_attributions` dict (name, target_regime,
    status, total_trades, attribution=...). Output is a per-strategy summary
    showing top-positive and top-negative lift buckets. The reflector LLM
    sees this and can reason about cross-strategy patterns ("3 of the last
    4 momentum strategies win in fgi_fear — generate more contrarian
    momentum filters next round").

    Returns "" when given an empty list so callers can unconditionally drop
    the section into the prompt.
    """
    if not attributions:
        return ""

    header = (
        f"PER-STRATEGY MACRO ATTRIBUTION (last {len(attributions)} backtests with"
        f" sufficient sample size).\n"
        "Each strategy shows which macro buckets at trade-entry time correlated\n"
        "with wins vs losses. 'lift +0.10' = bucket win-rate is 10 percentage\n"
        "points above the strategy's overall win-rate. Use this to spot which\n"
        "macro filters work or don't across the pool.\n"
    )

    lines = [header]
    for d in attributions:
        attr = d.get("attribution", {})
        buckets = attr.get("buckets", {})
        pos = attr.get("top_positive_lift", [])
        neg = attr.get("top_negative_lift", [])
        wr = attr.get("overall_win_rate", 0)
        tot = attr.get("total_trades", 0)

        lines.append(
            f"\n{d['name']} (regime={d.get('target_regime', '?')}, "
            f"status={d.get('status', '?')}, {tot} trades, "
            f"{wr:.0%} WR, profit={d.get('profit_total_pct', 0):.2f}%):"
        )
        if pos:
            wins = ", ".join(f"{b} {buckets[b]['lift']:+.2f}" for b in pos)
            lines.append(f"  Wins favored when: {wins}")
        if neg:
            losses = ", ".join(f"{b} {buckets[b]['lift']:+.2f}" for b in neg)
            lines.append(f"  Losses favored when: {losses}")
        if not pos and not neg:
            lines.append("  (no buckets cleared the sample-size threshold)")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    return text


def aggregate_attributions_by_bucket(
    rows: list[dict],
    regime: str | None = None,
    min_strategies: int = 2,
) -> dict:
    """Roll multiple per-strategy attributions up into bucket-wise stats.

    The reflector reads per-strategy attribution one row at a time; the
    generator wants the cross-strategy view — "fgi_fear was a top winner
    in 4/6 recent ranging strategies". That's much sharper signal for
    the LLM building the NEXT strategy than five individual data blobs.

    Filtering:
      regime=None        consider every row
      regime="ranging"   keep only rows whose target_regime is "ranging"

      Strict — we don't widen 'ranging' to include target_regime='all'.
      A target='all' strategy's wins aren't specifically attributable to
      ranging conditions, so mixing it in would muddy per-regime guidance.
      Caller is expected to fall back to the pool-wide aggregate when
      a regime is thin (use regime=None for that).

    Per-bucket fields:
      appears_positive   strategies where the bucket landed in top_positive_lift
      appears_negative   strategies where it landed in top_negative_lift
      n_with_data        strategies where the bucket had ANY data at all
                         (i.e. trades occurred in that bucket)
      avg_lift           mean lift across the n_with_data strategies

    Buckets with fewer than `min_strategies` appearances are excluded
    from top-N ranking so we don't promote single-strategy quirks as
    "consistent patterns".

    Returns:
      {
        "n_strategies": int,             # number of rows that contributed
        "regime": str|None,              # filter that was applied
        "buckets": {name: {...}},
        "top_consistent_winners": [name, ...],  # net positive, sorted
        "top_consistent_losers":  [name, ...],
      }
    """
    if regime is not None and regime != "all":
        rows = [r for r in rows if r.get("target_regime") == regime]

    if not rows:
        return {
            "n_strategies": 0, "regime": regime, "buckets": {},
            "top_consistent_winners": [], "top_consistent_losers": [],
        }

    # bucket name → running aggregation
    agg: dict = {}
    for row in rows:
        attr = row.get("attribution", {})
        pos = set(attr.get("top_positive_lift", []))
        neg = set(attr.get("top_negative_lift", []))
        buckets = attr.get("buckets", {})

        # Use the union of (top-listed buckets) ∪ (buckets-with-data) so
        # both contribute to aggregates correctly.
        for name in pos | neg | set(buckets.keys()):
            a = agg.setdefault(
                name,
                {"appears_positive": 0, "appears_negative": 0,
                 "n_with_data": 0, "_lift_sum": 0.0},
            )
            if name in pos:
                a["appears_positive"] += 1
            if name in neg:
                a["appears_negative"] += 1
            if name in buckets:
                a["n_with_data"] += 1
                a["_lift_sum"] += buckets[name].get("lift", 0.0)

    # Finalize: compute avg_lift and drop the running sum
    for name, a in agg.items():
        a["avg_lift"] = round(a["_lift_sum"] / a["n_with_data"], 3) if a["n_with_data"] else 0.0
        del a["_lift_sum"]

    # Rank — winners need NET positive appearances AND positive avg lift,
    # losers need NET negative appearances AND negative avg lift. The
    # double-condition keeps a bucket that lifts +0.01 in 3 strategies and
    # -0.10 in 2 strategies out of the winners list.
    def _net(b):
        return b["appears_positive"] - b["appears_negative"]

    winners = [
        (name, b) for name, b in agg.items()
        if _net(b) > 0 and b["avg_lift"] > 0
        and (b["appears_positive"] + b["appears_negative"]) >= min_strategies
    ]
    losers = [
        (name, b) for name, b in agg.items()
        if _net(b) < 0 and b["avg_lift"] < 0
        and (b["appears_positive"] + b["appears_negative"]) >= min_strategies
    ]
    winners.sort(key=lambda x: (-_net(x[1]), -x[1]["avg_lift"]))
    losers.sort(key=lambda x: (_net(x[1]), x[1]["avg_lift"]))

    return {
        "n_strategies": len(rows),
        "regime": regime,
        "buckets": agg,
        "top_consistent_winners": [n for n, _ in winners[:5]],
        "top_consistent_losers": [n for n, _ in losers[:5]],
    }


def format_aggregate_for_generator(
    agg: dict,
    target_regime: str,
    max_chars: int = 1200,
) -> str:
    """Render an aggregated-attribution dict as a generator prompt section.

    Returns "" when the aggregate has no strategies or no rankable buckets —
    callers can unconditionally drop the result into the prompt.

    `target_regime` is what the generator is being asked to build; we use
    it to caption the section and to flag when we fell back to pool-wide
    data (agg["regime"] != target_regime).
    """
    n = agg.get("n_strategies", 0)
    if n == 0:
        return ""

    winners = agg.get("top_consistent_winners", [])
    losers = agg.get("top_consistent_losers", [])
    if not winners and not losers:
        return ""

    scope = agg.get("regime")
    if scope is None or scope == "all" or scope != target_regime:
        scope_label = f"pool-wide; insufficient {target_regime!r}-regime data" \
            if target_regime not in (None, "all", scope) else "pool-wide across regimes"
    else:
        scope_label = f"{scope}-regime strategies"

    lines = [
        f"HISTORICAL ATTRIBUTION PATTERNS ({n} prior strategies, {scope_label}):",
        "Macro buckets that empirically separated wins from losses in recent backtests.",
    ]

    if winners:
        lines.append("\nConsistently favored WINS:")
        for name in winners:
            b = agg["buckets"][name]
            net = b["appears_positive"] - b["appears_negative"]
            lines.append(
                f"  {name}: top-positive in {b['appears_positive']}/{n} strategies, "
                f"net +{net}, avg lift {b['avg_lift']:+.2f}"
            )

    if losers:
        lines.append("\nConsistently favored LOSSES:")
        for name in losers:
            b = agg["buckets"][name]
            net = b["appears_negative"] - b["appears_positive"]
            lines.append(
                f"  {name}: top-negative in {b['appears_negative']}/{n} strategies, "
                f"net +{net}, avg lift {b['avg_lift']:+.2f}"
            )

    lines.append(
        "\nSTRONGLY consider gating entries to favor the winning conditions and "
        "filter out the losing ones in your macro_confidence block. These came "
        "from real trades, not priors."
    )

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated]"
    return text


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
