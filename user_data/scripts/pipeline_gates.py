"""
R7: Pipeline gates between backtest and promotion.

Each gate is a pure function: takes backtest output (and reference data),
returns a verdict dict. Gates never mutate DB state — the orchestrator
decides what to do with verdicts.

Gates landed in this round:
  regime_conditional_floor   adjust the min-trades floor by how much of the
                             backtest window was actually in the target regime
                             (a breakout strategy in a 90% ranging quarter
                             shouldn't be penalized for sitting out)
  beat_buyhold               strategy profit must beat BTC HODL OR have
                             materially lower drawdown over the same period
  walk_forward               OOS robustness: median sub-window sharpe must be
                             positive AND not catastrophically degrade vs full
  correlation                stubbed — requires per-trade exports we don't
                             have yet (Freqtrade backtests are run with
                             --export none); revisit when trade logs land

Verdict shape:
  {"passed": bool,
   "verdict": str,      # short code, e.g. "PASS_REGIME", "FAIL_BUYHOLD"
   "reason": str,       # one-line human summary
   "details": dict}     # the numbers that informed the decision
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, pstdev
from typing import Callable

log = logging.getLogger("pipeline_gates")


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def _pass(verdict: str, reason: str, **details) -> dict:
    return {"passed": True, "verdict": verdict, "reason": reason, "details": details}


def _fail(verdict: str, reason: str, **details) -> dict:
    return {"passed": False, "verdict": verdict, "reason": reason, "details": details}


def _skip(verdict: str, reason: str, **details) -> dict:
    """Gate didn't actually evaluate (missing data, n/a regime, degenerate
    inputs, etc.). The verdict carries ``passed=True`` for legacy compatibility
    with non-strict aggregation, but ``skipped=True`` lets a strict caller
    treat it as a fail. In production we want a strategy promoted ONLY when
    every expected gate produced a real pass — not because the reference
    data was missing and the gate auto-shrugged."""
    return {"passed": True, "verdict": verdict, "reason": reason,
            "details": details, "skipped": True}


def is_strict_pass(verdict: dict) -> bool:
    """A verdict that should count as PASS even when ``STRICT_PROMOTION_GATES``
    is on: passed and not skipped. The single source of truth for what
    "really passed" means — orchestrator + tests both go through this so
    we can't accidentally diverge."""
    return verdict.get("passed", False) and not verdict.get("skipped", False)


# ---------------------------------------------------------------------------
# Regime-conditional floor
# ---------------------------------------------------------------------------

def compute_regime_fractions(
    btc_df, lookback_days: int, end_date: datetime = None
) -> dict[str, float]:
    """Compute the fraction of the lookback window spent in each regime.

    Runs `add_regime_detection` over the BTC candles, then aggregates
    `regime` per-bar to fractions. Returns a dict with keys
    trending/ranging/breakout/crisis (zeros for regimes that never fired).

    `end_date` anchors the lookback window (default: now). Research runs
    that evaluate candidates against a historical window (issue #47
    positive-control experiments) pass the window's end so the regime
    floor reflects the SAME market the backtest saw, not today's.
    """
    import pandas as pd

    # Add regime_detector path — when called from orchestrator the
    # indicators package is already on PYTHONPATH; when called from tests
    # this import works directly.
    import sys
    indicators_path = Path(__file__).resolve().parent.parent
    if str(indicators_path) not in sys.path:
        sys.path.insert(0, str(indicators_path))
    from indicators.regime_detector import add_regime_detection

    df = btc_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        anchor = end_date or datetime.now(timezone.utc)
        cutoff = anchor - timedelta(days=lookback_days)
        df = df[(df["date"] >= cutoff) & (df["date"] <= anchor)]
    if len(df) < 60:
        # Not enough candles to classify — return uniform priors so the
        # floor adjustment is a no-op.
        return {"trending": 0.25, "ranging": 0.25, "breakout": 0.25, "crisis": 0.25}

    df = df.reset_index(drop=True)
    df = add_regime_detection(df)
    counts = df["regime"].value_counts(normalize=True).to_dict()
    return {
        "trending": float(counts.get("trending", 0.0)),
        "ranging": float(counts.get("ranging", 0.0)),
        "breakout": float(counts.get("breakout", 0.0)),
        "crisis": float(counts.get("crisis", 0.0)),
    }


def gate_regime_conditional_floor(
    bt: dict,
    target_regime: str,
    regime_fractions: dict[str, float],
    base_min_trades: int = 20,
    absolute_min_trades: int = 5,
) -> dict:
    """If the target regime was rare in the backtest window, lower the trade-count
    floor proportionally. A breakout strategy that only had 15% of the window
    in breakout regime should pass with ~3 trades, not 20.

    Strategies with target_regime='all' don't get any adjustment — they're
    expected to trade across all regimes, so the UNADJUSTED base floor
    applies as a real pass/fail. This used to be a skip verdict, which
    strict mode counts as "no evidence" — meaning no 'all'-regime candidate
    could ever promote under STRICT_PROMOTION_GATES, however profitable.
    The gate's evidence for 'all' is simply the base floor itself.

    Only a target_regime the fractions can't speak to (unknown label) still
    skips — there the gate genuinely has no basis to evaluate.
    """
    if target_regime == "all":
        trades = bt.get("total_trades", 0)
        if trades >= base_min_trades:
            return _pass(
                "PASS_REGIME",
                f"{trades} trades clears unadjusted floor of {base_min_trades} "
                f"(target=all — no regime adjustment applicable)",
                trades=trades, adjusted_floor=base_min_trades, regime_fraction=1.0,
            )
        return _fail(
            "FAIL_REGIME",
            f"{trades} trades < unadjusted floor of {base_min_trades} "
            f"(target=all — no regime adjustment applicable)",
            trades=trades, adjusted_floor=base_min_trades, regime_fraction=1.0,
        )

    if target_regime not in regime_fractions:
        return _skip("PASS_REGIME_NA", f"target_regime={target_regime} — no adjustment")

    frac = regime_fractions.get(target_regime, 0.0)
    adjusted = max(absolute_min_trades, int(round(base_min_trades * frac)))
    trades = bt.get("total_trades", 0)

    if trades >= adjusted:
        return _pass(
            "PASS_REGIME",
            f"{trades} trades clears regime-adjusted floor of {adjusted} "
            f"(target={target_regime}, was {frac:.0%} of window)",
            trades=trades, adjusted_floor=adjusted, regime_fraction=frac,
        )
    return _fail(
        "FAIL_REGIME",
        f"{trades} trades < regime-adjusted floor of {adjusted} "
        f"(target={target_regime} was {frac:.0%} of window)",
        trades=trades, adjusted_floor=adjusted, regime_fraction=frac,
    )


# ---------------------------------------------------------------------------
# Beat-buy-and-hold
# ---------------------------------------------------------------------------

def compute_btc_buyhold(btc_data_path: Path | str, timerange: str = None) -> dict:
    """Compute BTC HODL return % and max drawdown over the given Freqtrade
    timerange (e.g. '20251101-20260501'). If timerange is None, uses all
    available data.

    Returns {"profit_pct": float, "max_drawdown_pct": float, "days": int}
    or {"error": str} on failure.
    """
    import pandas as pd

    p = Path(btc_data_path)
    if not p.exists():
        return {"error": f"BTC data not found at {p}"}

    try:
        df = pd.read_feather(p) if p.suffix == ".feather" else pd.read_json(p)
    except Exception as e:
        return {"error": f"Failed to read {p}: {e}"}

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    if "close" not in df.columns:
        return {"error": "BTC dataframe has no 'close' column"}

    # Freqtrade's parse_backtest_output returns "all" when no timerange was
    # passed — treat that (and any non-Freqtrade-format string) as "use full
    # available range" rather than crashing on strptime.
    if timerange and timerange != "all" and "date" in df.columns:
        start_s, _, end_s = timerange.partition("-")
        try:
            if start_s:
                start = datetime.strptime(start_s, "%Y%m%d").replace(tzinfo=timezone.utc)
                df = df[df["date"] >= start]
            if end_s:
                end = datetime.strptime(end_s, "%Y%m%d").replace(tzinfo=timezone.utc)
                df = df[df["date"] <= end]
        except ValueError:
            return {"error": f"invalid timerange format: {timerange!r}"}

    if df.empty or len(df) < 2:
        return {"error": "no BTC data in timerange"}

    closes = df["close"].reset_index(drop=True)
    profit_pct = float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)

    # Max drawdown = largest peak-to-trough decline
    running_max = closes.cummax()
    drawdown = (closes - running_max) / running_max * 100
    max_dd = float(abs(drawdown.min()))

    days = 0
    if "date" in df.columns:
        days = (df["date"].iloc[-1] - df["date"].iloc[0]).days

    return {"profit_pct": profit_pct, "max_drawdown_pct": max_dd, "days": days}


def gate_beat_buyhold(
    bt: dict,
    bh: dict,
    profit_floor_ratio: float = 0.7,
    drawdown_advantage_pct: float = 5.0,
) -> dict:
    """Strategy must clear ONE of:
      (a) profit >= profit_floor_ratio * BH profit (default 70% of HODL), OR
      (b) max drawdown is at least drawdown_advantage_pct percentage points
          lower than BH drawdown (lower-risk alternative).

    Rationale: if you can't beat HODL on returns AND you're not meaningfully
    safer than HODL, why deploy a strategy with execution + slippage costs?
    """
    if bh.get("error"):
        return _skip("PASS_BH_NA", f"buyhold unavailable: {bh['error']}")

    s_profit = float(bt.get("profit_total_pct", 0.0))
    s_dd = float(bt.get("max_drawdown_pct", 0.0))
    bh_profit = float(bh["profit_pct"])
    bh_dd = float(bh["max_drawdown_pct"])

    # Degenerate case: strategy and HODL both at 0% (typically a strategy
    # that didn't trade vs a holding period that didn't move). Returning
    # PASS_BH_PROFIT here lets a 0-trade strategy silently clear the gate —
    # we saw exactly this masking the parser bug pre-PR-35. Make it an
    # explicit skip so strict mode rejects it.
    if s_profit == 0.0 and bh_profit == 0.0:
        return _skip(
            "PASS_BH_DEGENERATE",
            "both strategy and HODL profit are 0% — comparison meaningless",
            strategy_profit=s_profit, buyhold_profit=bh_profit,
            strategy_dd=s_dd, buyhold_dd=bh_dd,
        )

    # When BH is negative or flat, anything positive trivially wins on profit.
    # Cap the floor at 0 so we don't reward strategies for being "less bad
    # than a crash" without explicit consideration.
    profit_floor = bh_profit * profit_floor_ratio if bh_profit > 0 else 0.0
    profit_ok = s_profit >= profit_floor
    dd_ok = (bh_dd - s_dd) >= drawdown_advantage_pct

    details = {
        "strategy_profit": s_profit, "strategy_dd": s_dd,
        "buyhold_profit": bh_profit, "buyhold_dd": bh_dd,
        "profit_floor": profit_floor, "drawdown_advantage_required": drawdown_advantage_pct,
    }

    if profit_ok:
        return _pass(
            "PASS_BH_PROFIT",
            f"strategy {s_profit:.2f}% clears 70%-of-HODL floor ({profit_floor:.2f}%)",
            **details,
        )
    if dd_ok:
        return _pass(
            "PASS_BH_SAFER",
            f"strategy DD {s_dd:.2f}% is {bh_dd - s_dd:.2f}pp lower than HODL DD {bh_dd:.2f}%",
            **details,
        )
    return _fail(
        "FAIL_BH",
        f"strategy {s_profit:.2f}% < floor {profit_floor:.2f}% AND "
        f"DD advantage {bh_dd - s_dd:.2f}pp < required {drawdown_advantage_pct}pp",
        **details,
    )


# ---------------------------------------------------------------------------
# Walk-forward robustness
# ---------------------------------------------------------------------------

def split_timerange(end_date: datetime, total_days: int, n_splits: int) -> list[str]:
    """Produce N contiguous Freqtrade timeranges ending at end_date, each
    covering total_days // n_splits days.

    Example: end=2026-05-01, total=180, n=3 →
      ['20251102-20260101', '20260101-20260301', '20260301-20260501']
    """
    if n_splits < 2:
        raise ValueError("walk-forward needs at least 2 splits")
    per = total_days // n_splits
    ranges = []
    for i in range(n_splits):
        end = end_date - timedelta(days=(n_splits - 1 - i) * per)
        start = end - timedelta(days=per)
        ranges.append(f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}")
    return ranges


def run_walk_forward(
    strategy_name: str,
    backtest_fn: Callable[[str, str], dict],
    n_splits: int = 3,
    days_per_split: int = 60,
    end_date: datetime = None,
    retry_crashed: int = 1,
    retry_delay_seconds: float = 5.0,
) -> list[dict]:
    """Run the same strategy across N consecutive sub-windows. Returns a list
    of backtest result dicts (one per window). Pure shell — relies on the
    injected backtest_fn so it's trivially testable.

    backtest_fn signature: (strategy_name, timerange) -> dict

    A crashed window is retried up to `retry_crashed` times (with a short
    pause) before it counts: windows launch back-to-back containers, and a
    transient failure (exchange market load, docker hiccup) would otherwise
    turn into a permanent FAIL_WF_CRASH — observed once during the issue #47
    shakedown, where the crash error itself was discarded. A real crash
    fails the retry too, and now gets its error logged.
    """
    import time as _time

    if end_date is None:
        end_date = datetime.now(timezone.utc)
    ranges = split_timerange(end_date, n_splits * days_per_split, n_splits)
    results = []
    for tr in ranges:
        log.info(f"walk-forward window: {strategy_name} {tr}")
        r = backtest_fn(strategy_name, tr)
        attempt = 0
        while not r.get("success") and attempt < retry_crashed:
            attempt += 1
            log.warning(
                f"walk-forward window {tr} crashed "
                f"({r.get('error', 'unknown')}); retry {attempt}/{retry_crashed}"
            )
            if retry_delay_seconds:
                _time.sleep(retry_delay_seconds)
            r = backtest_fn(strategy_name, tr)
        if not r.get("success"):
            log.warning(
                f"walk-forward window {tr} failed permanently: "
                f"{r.get('error', 'unknown')}\n"
                f"--- last output ---\n{str(r.get('raw_output', ''))[-500:]}"
            )
        r["_window_timerange"] = tr
        results.append(r)
    return results


def gate_walk_forward(
    window_results: list[dict],
    min_passing_windows: int = 2,
    max_sharpe_std: float = 1.5,
) -> dict:
    """Pass if at least `min_passing_windows` sub-windows had sharpe > 0
    AND the cross-window std of sharpe is below max_sharpe_std (catches
    "one lucky month carried the full backtest" pathology).

    Doesn't require any window to be amazing — just that the result is
    consistently non-terrible. The full-period backtest already enforces
    the "good" bar separately.
    """
    if not window_results:
        return _skip("PASS_WF_NA", "no window results provided")

    sharpes = [float(r.get("sharpe", 0.0)) for r in window_results if r.get("success")]
    n_ok = len(sharpes)
    if n_ok < len(window_results):
        return _fail(
            "FAIL_WF_CRASH",
            f"{len(window_results) - n_ok}/{len(window_results)} walk-forward windows crashed",
            window_sharpes=sharpes,
        )

    passing = sum(1 for s in sharpes if s > 0)
    sharpe_std = pstdev(sharpes) if len(sharpes) > 1 else 0.0
    med = median(sharpes) if sharpes else 0.0

    details = {
        "window_sharpes": sharpes,
        "passing_windows": passing,
        "required_passing": min_passing_windows,
        "sharpe_std": sharpe_std,
        "sharpe_median": med,
    }

    if passing < min_passing_windows:
        return _fail(
            "FAIL_WF_INCONSISTENT",
            f"only {passing}/{len(sharpes)} windows had sharpe > 0 "
            f"(need {min_passing_windows}); medians={med:.2f}",
            **details,
        )
    if sharpe_std > max_sharpe_std:
        return _fail(
            "FAIL_WF_UNSTABLE",
            f"sharpe std {sharpe_std:.2f} > {max_sharpe_std} — "
            f"one window likely carried the full backtest",
            **details,
        )
    return _pass(
        "PASS_WF",
        f"{passing}/{len(sharpes)} windows positive, std {sharpe_std:.2f}, "
        f"median {med:.2f}",
        **details,
    )


# ---------------------------------------------------------------------------
# Correlation filter (R7.4)
# ---------------------------------------------------------------------------

def trades_to_daily_returns(trades: list[dict]) -> "pd.Series":
    """Convert a Freqtrade trade list to a daily-indexed Series of returns.

    Each day's value = sum of `profit_ratio` of trades that CLOSED that day.
    Days with no closing trades become 0.0. Index is sorted ascending UTC
    timestamps at midnight.

    Empty list returns an empty Series.
    """
    import pandas as pd

    if not trades:
        return pd.Series(dtype=float)

    close_dates = []
    profits = []
    for t in trades:
        cd = t.get("close_date")
        if cd is None:
            continue
        try:
            close_dates.append(pd.to_datetime(cd, utc=True))
            profits.append(float(t.get("profit_ratio", 0.0)))
        except (ValueError, TypeError):
            continue

    if not close_dates:
        return pd.Series(dtype=float)

    df = pd.DataFrame({"close": close_dates, "profit": profits})
    daily = df.groupby(df["close"].dt.floor("D"))["profit"].sum().sort_index()

    # Reindex to full daily range so days with no trades count as 0 returns
    # (an "inactive" day is meaningful for correlation — two strategies that
    # are both off on the same days are still correlated in a portfolio
    # sense, since neither smooths the other's drawdown).
    full_idx = pd.date_range(daily.index.min(), daily.index.max(),
                              freq="1D", tz="UTC")
    return daily.reindex(full_idx, fill_value=0.0)


def gate_correlation(
    candidate_trades: list[dict],
    active_strategies: list[dict],
    *,
    threshold: float = 0.7,
    min_overlap_days: int = 30,
    load_trades=None,
) -> dict:
    """Reject the candidate if its daily-return series correlates above
    `threshold` with ANY active strategy.

    candidate_trades: the candidate's exported trade list (already loaded).
    active_strategies: list of {name, trades_export_path} for each promoted
        strategy. Entries with empty or missing trades_export_path are
        skipped silently (legacy strategies pre-R2d).
    threshold: max acceptable Pearson correlation (default 0.7 — the
        common diversification cutoff).
    min_overlap_days: skip a pairwise comparison if the two series overlap
        for fewer days than this — correlation on tiny samples is noise.
    load_trades: injectable trade-loader for testability; defaults to
        trade_attribution.load_trades_from_zip.

    Returns a verdict dict. PASS_CORR_NA when there are no active
    strategies to compare against, or the candidate has no trades.
    """
    if not active_strategies:
        return _skip("PASS_CORR_NA", "no active strategies to compare against")

    cand_series = trades_to_daily_returns(candidate_trades)
    if cand_series.empty:
        return _skip("PASS_CORR_NA", "candidate has no trades to correlate")

    if load_trades is None:
        # Lazy import to avoid pulling pandas/zipfile into modules that
        # only need the simpler gates.
        from trade_attribution import load_trades_from_zip as _load
        load_trades = _load

    compared = 0
    checked = []
    for s in active_strategies:
        path = s.get("trades_export_path", "")
        if not path:
            continue
        active_trades = load_trades(path, s["name"])
        if not active_trades:
            continue
        active_series = trades_to_daily_returns(active_trades)
        if active_series.empty:
            continue

        # Align on the intersection of dates. If the overlap is too small,
        # correlation is statistical noise — skip the comparison.
        common_idx = cand_series.index.intersection(active_series.index)
        if len(common_idx) < min_overlap_days:
            continue
        a = cand_series.loc[common_idx]
        b = active_series.loc[common_idx]
        # corr returns NaN if either series has zero variance (constant)
        corr = a.corr(b)
        if corr != corr:  # NaN check
            continue

        compared += 1
        checked.append({"name": s["name"], "corr": round(float(corr), 3),
                         "overlap_days": int(len(common_idx))})
        if corr > threshold:
            return _fail(
                "FAIL_CORRELATION",
                f"corr {corr:.2f} > {threshold} with active strategy "
                f"{s['name']!r} ({len(common_idx)} overlapping days)",
                threshold=threshold, peer=s["name"], correlation=float(corr),
                overlap_days=int(len(common_idx)), all_checked=checked,
            )

    if compared == 0:
        return _skip(
            "PASS_CORR_NA",
            "no comparable active strategies (missing exports or insufficient overlap)",
        )

    max_corr = max(c["corr"] for c in checked)
    return _pass(
        "PASS_CORR",
        f"max correlation with {compared} active strategies = {max_corr:.2f} (threshold {threshold})",
        threshold=threshold, max_correlation=max_corr, checked=checked,
    )


# ---------------------------------------------------------------------------
# Orchestration helper — run all gates and combine
# ---------------------------------------------------------------------------

def run_all_gates(
    bt: dict,
    target_regime: str,
    *,
    btc_data_path: Path | str = None,
    timerange: str = None,
    regime_fractions: dict[str, float] = None,
    walk_forward_results: list[dict] = None,
    base_min_trades: int = 20,
    candidate_trades: list[dict] = None,
    active_strategies: list[dict] = None,
    correlation_threshold: float = 0.7,
) -> dict:
    """Run every gate, return {"all_passed": bool, "verdicts": [verdict, ...]}.

    The orchestrator should only promote if all_passed is True. Each
    individual verdict is preserved so we can log per-gate outcomes.

    Correlation gate runs only when BOTH candidate_trades AND
    active_strategies are provided; otherwise it skips. Loading active
    strategies' trade exports requires file IO so callers that want a
    pure-data run can omit those args.
    """
    verdicts = []

    if regime_fractions is None:
        verdicts.append(_skip("PASS_REGIME_NA", "regime_fractions not provided"))
    else:
        verdicts.append(gate_regime_conditional_floor(
            bt, target_regime, regime_fractions, base_min_trades=base_min_trades
        ))

    if btc_data_path is None:
        verdicts.append(_skip("PASS_BH_NA", "btc_data_path not provided"))
    else:
        bh = compute_btc_buyhold(btc_data_path, timerange)
        verdicts.append(gate_beat_buyhold(bt, bh))

    if walk_forward_results is None:
        verdicts.append(_skip("PASS_WF_NA", "walk-forward not run"))
    else:
        verdicts.append(gate_walk_forward(walk_forward_results))

    if candidate_trades is None or active_strategies is None:
        verdicts.append(_skip("PASS_CORR_NA", "correlation inputs not provided"))
    else:
        verdicts.append(gate_correlation(
            candidate_trades, active_strategies, threshold=correlation_threshold,
        ))

    # A verdict is "blocking" only if .passed is False. Skipped ones pass.
    all_passed = all(v["passed"] for v in verdicts)
    return {"all_passed": all_passed, "verdicts": verdicts}
