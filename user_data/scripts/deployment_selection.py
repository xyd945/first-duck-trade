"""
Greedy correlation-aware selection of which approved strategies to deploy.

The Phase 2 reconciler asks:
   given the eligible-to-deploy pool, which TOP N should run live?

Policy is the V1 spec in ``docs/deployment-lifecycle.md``:

  1. Sort eligible by Sharpe descending.
  2. Walk the list. Add each candidate to ``selected`` unless its
     daily-return correlation with anything already-selected is at or
     above ``corr_threshold`` (default 0.7).
  3. Stop when ``len(selected) == max_deploy``.

Why this lives in its own module: the policy is the load-bearing piece
the operator needs to reason about. Keeping it out of the orchestrator
job + out of the Docker SDK wrapper means tests are tiny (no docker,
no DB, no scheduler) and we can swap policies later (e.g. Phase 5's
sharpe-weighted variant) without touching the reconciler control flow.

Pair correlation re-uses the same primitives as ``gate_correlation``:
trades → daily returns → align on intersection → Pearson corr. A pair
that can't be compared (one side has no trade export, or the overlap
window is too small for meaningful correlation) is treated as
"uncorrelated for selection purposes" — same conservative posture the
promotion-side gate uses. This means selection can't FALSELY reject a
candidate for missing data; if the pool is dominated by data-less
strategies, the operator will see them all selected and can intervene.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

log = logging.getLogger("deployment_selection")


# ---------------------------------------------------------------------------
# Defaults — also surfaced in the locked V1 spec
# ---------------------------------------------------------------------------

DEFAULT_MAX_DEPLOY = 3
DEFAULT_CORR_THRESHOLD = 0.7
DEFAULT_MIN_OVERLAP_DAYS = 30


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _max_corr_against(
    candidate_trades,
    selected_rows: list[dict],
    *,
    min_overlap_days: int,
    load_trades: Callable,
) -> Optional[tuple[str, float]]:
    """Return (peer_name, correlation) of the most-correlated already-selected
    peer, or None if no pair could be compared.

    Skips pairs where either side has no trade data or the date overlap is
    too small. A NaN correlation (constant series) is treated as skip.
    """
    from pipeline_gates import trades_to_daily_returns

    cand_series = trades_to_daily_returns(candidate_trades)
    if cand_series.empty:
        return None

    best_peer = None
    best_corr = float("-inf")

    for s in selected_rows:
        path = s.get("trades_export_path", "")
        name = s.get("name", "")
        if not path or not name:
            continue
        peer_trades = load_trades(path, name)
        if not peer_trades:
            continue
        peer_series = trades_to_daily_returns(peer_trades)
        if peer_series.empty:
            continue

        common_idx = cand_series.index.intersection(peer_series.index)
        if len(common_idx) < min_overlap_days:
            continue

        corr = cand_series.loc[common_idx].corr(peer_series.loc[common_idx])
        if corr != corr:  # NaN guard
            continue

        corr = float(corr)
        if corr > best_corr:
            best_corr = corr
            best_peer = name

    if best_peer is None:
        return None
    return best_peer, best_corr


def compute_desired_deployments(
    eligible: list[dict],
    *,
    max_deploy: int = DEFAULT_MAX_DEPLOY,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    min_overlap_days: int = DEFAULT_MIN_OVERLAP_DAYS,
    load_trades: Optional[Callable] = None,
) -> dict:
    """Apply greedy correlation-aware selection to the eligible pool.

    Args:
      eligible: rows from ``strategy_registry.get_deployment_eligible()``.
                Each must include ``id``, ``name``, ``sharpe``, and
                ``trades_export_path`` (may be empty string for
                pre-export strategies).
      max_deploy: hard cap on returned slots.
      corr_threshold: candidate skipped if max correlation with any
                already-selected meets or exceeds this.
      min_overlap_days: pairs with fewer overlapping trade-days than
                this are treated as "uncomparable" (not as 0-corr or
                high-corr — just skipped, since correlation on tiny
                samples is statistical noise).
      load_trades: injectable trade-loader (path, name) -> trades, for
                tests. Defaults to trade_attribution.load_trades_from_zip.

    Returns:
      {
        'desired':  [selected rows, in selection order — best Sharpe first]
        'skipped':  [{row: dict, reason: str} for each candidate not selected]
      }

    Notes:
      - A candidate with empty ``trades_export_path`` selected first acts
        as a "no-correlation-data" anchor — subsequent candidates can't be
        rejected for correlating with it (we can't compute the pair). This
        is intentional: the promotion-side correlation gate already
        enforced diversification against the at-promotion-time peer set.
        Selection here is a second pass, conservative on missing data.
      - The function is PURE — no DB, no docker, no IO except the
        injected load_trades. Easy to test, easy to reason about.
    """
    if load_trades is None:
        from trade_attribution import load_trades_from_zip as _load
        load_trades = _load

    sorted_by_sharpe = sorted(
        eligible, key=lambda r: float(r.get("sharpe", 0.0)), reverse=True
    )

    desired: list[dict] = []
    skipped: list[dict] = []

    for cand in sorted_by_sharpe:
        if len(desired) >= max_deploy:
            skipped.append({"row": cand, "reason": "deployment_slots_full"})
            continue

        cand_path = cand.get("trades_export_path", "")
        if not cand_path:
            # No trade data on the candidate — we can't compute pairs against
            # it. Allow it through (conservative on missing data); log it so
            # operator can spot a pool dominated by data-less strategies.
            desired.append(cand)
            log.info(f"  [selection] no trades_export_path for {cand.get('name')!r} "
                     f"— admitting without correlation check")
            continue

        cand_trades = load_trades(cand_path, cand.get("name", ""))
        if not cand_trades:
            desired.append(cand)
            log.info(f"  [selection] zero-trade export for {cand.get('name')!r} "
                     f"— admitting without correlation check")
            continue

        peer_info = _max_corr_against(
            cand_trades, desired,
            min_overlap_days=min_overlap_days,
            load_trades=load_trades,
        )
        if peer_info is None:
            # No comparable peer (either desired is empty, or pairs lacked
            # comparable data). Admit.
            desired.append(cand)
            continue

        peer_name, max_corr = peer_info
        if max_corr >= corr_threshold:
            skipped.append({
                "row": cand,
                "reason": f"corr {max_corr:.3f} >= {corr_threshold} with "
                          f"already-selected {peer_name!r}",
            })
        else:
            desired.append(cand)

    return {"desired": desired, "skipped": skipped}
