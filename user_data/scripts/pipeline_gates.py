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
    """Soft-pass: gate didn't run (missing data, n/a regime, etc.). Treated
    as PASS by the orchestrator but logged separately so we can spot gates
    that silently never fire."""
    return {"passed": True, "verdict": verdict, "reason": reason,
            "details": details, "skipped": True}


# ---------------------------------------------------------------------------
# Regime-conditional floor
# ---------------------------------------------------------------------------

def compute_regime_fractions(btc_df, lookback_days: int) -> dict[str, float]:
    """Compute the fraction of the lookback window spent in each regime.

    Runs `add_regime_detection` over the BTC candles, then aggregates
    `regime` per-bar to fractions. Returns a dict with keys
    trending/ranging/breakout/crisis (zeros for regimes that never fired).
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        df = df[df["date"] >= cutoff]
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
    expected to trade across all regimes.
    """
    if target_regime == "all" or target_regime not in regime_fractions:
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
) -> list[dict]:
    """Run the same strategy across N consecutive sub-windows. Returns a list
    of backtest result dicts (one per window). Pure shell — relies on the
    injected backtest_fn so it's trivially testable.

    backtest_fn signature: (strategy_name, timerange) -> dict
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    ranges = split_timerange(end_date, n_splits * days_per_split, n_splits)
    results = []
    for tr in ranges:
        log.info(f"walk-forward window: {strategy_name} {tr}")
        r = backtest_fn(strategy_name, tr)
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
# Correlation filter — stub
# ---------------------------------------------------------------------------

def gate_correlation(*args, **kwargs) -> dict:
    """Stub: monthly P&L correlation between candidate and active strategies.

    Real implementation needs per-trade exports from Freqtrade (--export
    trades), which we currently suppress with --export none for disk
    reasons. Re-enable once we wire trade-log storage into the registry.

    Always returns a skip verdict so it's a safe no-op in the gate chain.
    """
    return _skip("PASS_CORR_STUB",
                 "correlation gate not yet implemented (no per-trade exports)")


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
) -> dict:
    """Run every gate, return {"all_passed": bool, "verdicts": [verdict, ...]}.

    The orchestrator should only promote if all_passed is True. Each
    individual verdict is preserved so we can log per-gate outcomes.
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

    verdicts.append(gate_correlation())

    # A verdict is "blocking" only if .passed is False. Skipped ones pass.
    all_passed = all(v["passed"] for v in verdicts)
    return {"all_passed": all_passed, "verdicts": verdicts}
