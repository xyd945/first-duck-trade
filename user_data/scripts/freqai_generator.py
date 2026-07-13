"""
FreqAI experiment generator — LLM-proposed ML specs (issue #47, Phase 3a).

The LLM proposes DECLARATIVE experiment specs (features from the whitelist,
target horizon, bounded LightGBM params, thresholds) — never code. Every
proposal goes through freqai_spec.validate_freqai_spec and the same
materialize/register path as a hand-written spec, so a hallucinated feature
or out-of-bounds param is a clean rejection with retry feedback, not a risk.

Anti-drift property: the prompt's allowed-lists and bounds are BUILT from
the validator's own constants (freqai_spec bounds, freqai_features keys).
Extending the whitelist automatically teaches the LLM about it; the prompt
can never promise something validation rejects.

Batch-first by design — the operator's actual workflow is manual rounds of
"propose N, evaluate, compare" (the weekly cron hookup is a thin layer on
top, off by default via FREQAI_WEEKLY_COUNT=0):

  propose   generate + validate + register N specs via the LLM
  evaluate  run pending freqai candidates through the real lifecycle
            (full backtest + gates + promote/retire), one at a time
  report    comparison table of every freqai experiment so far
  run-batch propose + evaluate + report in one go

Run inside the orchestrator container:
  docker exec ft-orchestrator python /app/user_data/scripts/freqai_generator.py \\
      run-batch --count 3
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("freqai_generator")

BASE_DIR = Path(__file__).resolve().parent.parent  # user_data/

# Human-readable descriptions for the prompt, keyed by the SAME feature
# keys the validator accepts. A key here without a library entry (or vice
# versa) fails the drift test in test_freqai_generator.py.
FEATURE_DESCRIPTIONS = {
    "rsi": "RSI momentum oscillator (period-expanded)",
    "ema_dist": "distance of close from EMA, normalized (period-expanded)",
    "natr": "normalized ATR volatility (period-expanded)",
    "adx": "trend strength (period-expanded)",
    "bb_width": "Bollinger band width — volatility compression (period-expanded)",
    "roc": "rate of change over the period (period-expanded)",
    "volume_z": "volume z-score vs rolling window (period-expanded)",
    "pct_change": "1-candle close pct change",
    "hl_range": "candle high-low range relative to close",
    "time_cycle": "sin/cos hour-of-week cycle (session/time-of-day effects)",
    "funding": "BTC perp funding rate — positioning (positive = longs pay)",
    "oi_change": "BTC open-interest 24h pct change — leverage building/unwinding",
    "eth_btc": "ETH/BTC 7d change — alt momentum proxy",
    "alt_strength": "ETH/BTC 30d z-score — alt-season regime signal",
    "macro_fgi": "composite Fear & Greed index",
    "macro_vix": "CBOE VIX close — cross-asset risk appetite",
}


def build_system_prompt() -> str:
    """Assemble the system prompt from the validator's own constants."""
    from freqai_spec import (
        ENTRY_THRESHOLD_BOUNDS, FREQAI_PARAM_BOUNDS, HORIZON_BOUNDS,
        INDICATOR_PERIOD_BOUNDS, MIN_FEATURES, MODEL_FAMILIES,
        MODEL_PARAM_BOUNDS, VALID_REGIMES,
    )

    features_block = "\n".join(
        f"  {key:14s} {desc}" for key, desc in FEATURE_DESCRIPTIONS.items()
    )
    model_params_block = "\n".join(
        f"  {name}: {typ.__name__} in [{lo}, {hi}]"
        for name, (typ, lo, hi) in MODEL_PARAM_BOUNDS.items()
    )
    freqai_params_block = "\n".join(
        f"  {name}: {typ.__name__} in [{lo}, {hi}]"
        for name, (typ, lo, hi) in FREQAI_PARAM_BOUNDS.items()
    )

    return f"""You are an ML experiment designer for an automated crypto trading factory.
You propose FreqAI experiment SPECS — structured JSON, never code. A spec
trains a {MODEL_FAMILIES[0]} to predict the forward return over a fixed
horizon from whitelisted features, and trades long-only on thresholded
predictions (spot policy: no shorting).

OUTPUT FORMAT: exactly ONE JSON object. No prose, no markdown fences.

{{
  "spec_type": "freqai",
  "name": "FreqaiPascalCaseName",
  "thesis": "one or two sentences: WHY these features should predict returns at this horizon",
  "target_regime": "trending|ranging|breakout|all",
  "features": ["rsi", "funding", ...],
  "target": {{"type": "future_return", "horizon_candles": 24}},
  "model": {{"family": "{MODEL_FAMILIES[0]}", "params": {{"n_estimators": 400, "learning_rate": 0.05}}}},
  "thresholds": {{"entry": 0.005, "exit": 0.0}},
  "freqai": {{"train_period_days": 60, "backtest_period_days": 7,
              "include_shifted_candles": 2, "indicator_periods_candles": [14, 50],
              "test_size": 0.25}},
  "risk": {{"stoploss": -0.06, "minimal_roi": {{"0": 0.15, "60": 0.08, "120": 0.04, "240": 0.02}}}}
}}

ALLOWED FEATURES (pick >= {MIN_FEATURES}, no duplicates, nothing outside this list):
{features_block}

HARD BOUNDS (validation rejects anything outside):
  target.horizon_candles: int in [{HORIZON_BOUNDS[0]}, {HORIZON_BOUNDS[1]}]  (1h candles: 24 = one day)
  thresholds.entry: number in [{ENTRY_THRESHOLD_BOUNDS[0]}, {ENTRY_THRESHOLD_BOUNDS[1]}] — predicted fwd return to enter long
  thresholds.exit: number in [-0.05, entry) — predicted fwd return to exit
  risk.stoploss: negative number >= -0.5
  risk.minimal_roi: keys are minute-strings ("0", "60", ...), values numbers
  freqai.indicator_periods_candles: 1-4 ints in [{INDICATOR_PERIOD_BOUNDS[0]}, {INDICATOR_PERIOD_BOUNDS[1]}]
model.params (all optional):
{model_params_block}
freqai block (all optional):
{freqai_params_block}

DESIGN GUIDANCE:
- Scale thresholds to the horizon: a 24-candle horizon has larger typical
  moves than a 6-candle one; an entry threshold the market rarely clears
  produces near-zero trades and the candidate is retired for it.
- target_regime is a real hypothesis, not a label: regime-specific
  candidates get a regime-adjusted trade floor; "all" candidates must clear
  the full 20-trade floor over ~6 months.
- Prefer a coherent thesis over kitchen-sink feature lists — every feature
  should have a reason to predict returns at YOUR horizon.
- The candidate must survive: profitable overall, >= 20 trades
  (regime-adjusted), beat 70% of BTC buy-and-hold OR have 5pp lower
  drawdown, walk-forward across 3 consecutive 60-day windows (majority of
  windows positive sharpe, stable across windows), and < 0.7 correlation
  with active strategies. Overfitting one lucky month fails walk-forward.
"""


def _format_freqai_failures(rows: list) -> str:
    """Compact 'do NOT repeat these' block from retired freqai experiments.

    Uses the spec sidecar (features/horizon/thresholds) when it still
    exists so the LLM sees the experiment shape that failed, not just the
    name."""
    if not rows:
        return ""
    from freqai_spec import load_spec_sidecar

    lines = ["PRIOR FAILED ML EXPERIMENTS (do NOT propose near-duplicates):"]
    for r in rows:
        spec = load_spec_sidecar(r.get("filepath", "")) or {}
        shape = ""
        if spec:
            shape = (f" features={spec.get('features')} "
                     f"horizon={spec.get('target', {}).get('horizon_candles')} "
                     f"entry_thr={spec.get('thresholds', {}).get('entry')}")
        lines.append(
            f"- {r['name']} [{r.get('failure_verdict', '')}] "
            f"trades={r.get('total_trades')} profit={r.get('profit_total_pct')}% "
            f"sharpe={r.get('sharpe')}{shape}\n"
            f"  reason: {r.get('failure_reason', '')[:200]}"
        )
    return "\n".join(lines)


def build_freqai_prompt(
    target_regime: str | None = None,
    context: str = "",
    failure_examples: str = "",
    reflector_insights: str = "",
    prior_in_batch: list[dict] | None = None,
) -> str:
    """User-prompt assembly, mirroring the rule generator's ordering:
    instructions -> insights -> failures -> batch-diversity -> context."""
    parts = []
    if target_regime:
        parts.append(f"Propose ONE FreqAI experiment spec targeting "
                     f"regime: {target_regime}.")
    else:
        parts.append("Propose ONE FreqAI experiment spec. Choose the "
                     "target_regime that best fits your thesis.")
    if reflector_insights:
        parts.append(f"RECENT REFLECTOR INSIGHTS:\n{reflector_insights}")
    if failure_examples:
        parts.append(failure_examples)
    if prior_in_batch:
        summaries = "\n".join(
            f"- {s['name']}: features={s.get('features')} "
            f"horizon={s.get('horizon')} regime={s.get('regime')}"
            for s in prior_in_batch
        )
        parts.append(
            "ALREADY PROPOSED IN THIS BATCH (make yours structurally "
            f"different — other features, horizon, or regime):\n{summaries}"
        )
    if context:
        parts.append(f"MARKET CONTEXT:\n{context}")
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


def _unique_name(name: str) -> str:
    """Suffix the class name if the registry already has it."""
    from strategy_registry import get_strategy_by_name

    if get_strategy_by_name(name) is None:
        return name
    for i in range(2, 100):
        candidate = f"{name}_{i}"
        if get_strategy_by_name(candidate) is None:
            return candidate
    raise RuntimeError(f"could not find a free name for {name!r}")


def propose_freqai_specs(
    count: int = 3,
    target_regime: str | None = None,
    context: str = "",
    reflector_insights: str = "",
    model: str | None = None,
    provider: str | None = None,
    max_retries: int = 2,
) -> list[dict]:
    """Generate `count` specs via the LLM; validate + register each.

    Returns one result dict per attempt slot:
      {"success": True, "name", "strategy_id", "spec"} or
      {"success": False, "error", "raw": <last extracted spec or text>}

    A validation failure feeds the validator's error message back to the
    LLM (up to `max_retries` extra turns per spec) — the same retry shape
    the rule generator uses.
    """
    from freqai_spec import (
        FreqaiSpecError, register_freqai_candidate, validate_freqai_spec,
    )
    from llm_client import chat_completion
    from strategy_generator import _extract_spec_json
    from strategy_registry import get_recent_failures, init_db

    init_db()
    system_prompt = build_system_prompt()
    failures = get_recent_failures(k=8, spec_type="freqai")
    failure_block = _format_freqai_failures(failures)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    results: list[dict] = []
    accepted: list[dict] = []  # batch-diversity summaries

    for i in range(count):
        base_prompt = build_freqai_prompt(
            target_regime=target_regime,
            context=context,
            failure_examples=failure_block,
            reflector_insights=reflector_insights,
            prior_in_batch=accepted,
        )
        messages = [{"role": "user", "content": base_prompt}]
        outcome: dict = {"success": False, "error": "no attempts made"}

        for attempt in range(max_retries + 1):
            try:
                reply = chat_completion(
                    messages, system=system_prompt, model=model,
                    max_tokens=4096, provider=provider,
                )
            except Exception as e:
                outcome = {"success": False, "error": f"LLM call failed: {e}"}
                break

            spec = _extract_spec_json(reply)
            if spec is None:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content":
                                 "That was not a single parseable JSON object. "
                                 "Return ONLY the JSON object."})
                outcome = {"success": False,
                           "error": "no parseable JSON in LLM reply",
                           "raw": reply[:500]}
                continue

            # Stamp fields the factory owns — never trust the LLM's.
            spec["spec_type"] = "freqai"
            spec["generation_id"] = f"gen-freqai-{ts}-i{i}v{attempt}"
            if isinstance(spec.get("name"), str):
                spec["name"] = _unique_name(spec["name"])

            try:
                validate_freqai_spec(spec)
                strategy_id = register_freqai_candidate(spec)
            except FreqaiSpecError as e:
                log.info(f"  spec {i} attempt {attempt}: rejected — {e}")
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content":
                                 f"Spec rejected by the validator: {e}\n"
                                 f"Fix that and return the corrected JSON "
                                 f"object only."})
                outcome = {"success": False, "error": str(e), "raw": spec}
                continue

            outcome = {"success": True, "name": spec["name"],
                       "strategy_id": strategy_id, "spec": spec}
            accepted.append({
                "name": spec["name"],
                "features": spec.get("features"),
                "horizon": spec.get("target", {}).get("horizon_candles"),
                "regime": spec.get("target_regime"),
            })
            log.info(f"  spec {i}: registered {spec['name']} "
                     f"(id={strategy_id})")
            break

        results.append(outcome)

    ok = sum(1 for r in results if r["success"])
    log.info(f"FreqAI proposal batch: {ok}/{count} specs registered")
    return results


def evaluate_pending(limit: int | None = None) -> list[str]:
    """Run every pending freqai candidate through the real lifecycle
    (full backtest + gates + promote/retire + model purge), one at a time.
    Returns the names evaluated."""
    from strategy_registry import get_candidates
    import orchestrator

    names = [c["name"] for c in get_candidates()
             if (c.get("spec_type") or "rule") == "freqai"]
    if limit:
        names = names[:limit]
    for name in names:
        log.info(f"=== Evaluating freqai candidate: {name} ===")
        orchestrator.job_backtest_candidates(only_name=name)
    return names


def build_report() -> str:
    """Comparison table of every freqai experiment in the registry."""
    from freqai_spec import load_spec_sidecar
    from strategy_registry import get_strategies_overview

    rows = get_strategies_overview(spec_type="freqai")
    if not rows:
        return "No freqai experiments in the registry yet."

    header = (f"{'name':32s} {'regime':9s} {'hrz':>4s} {'entry':>7s} "
              f"{'nfeat':>5s} {'trades':>6s} {'profit%':>8s} {'sharpe':>7s} "
              f"{'status':9s} verdict")
    lines = [header, "-" * len(header)]
    for r in rows:
        spec = load_spec_sidecar(r.get("filepath", "")) or {}
        horizon = spec.get("target", {}).get("horizon_candles", "?")
        entry = spec.get("thresholds", {}).get("entry", "?")
        nfeat = len(spec.get("features", []) or [])
        lines.append(
            f"{r['name'][:32]:32s} {r['target_regime']:9s} {horizon!s:>4s} "
            f"{entry!s:>7s} {nfeat:>5d} {r['total_trades']:>6d} "
            f"{r['profit_total_pct']:>8.2f} {r['sharpe']:>7.2f} "
            f"{r['status']:9s} {r.get('failure_verdict') or '-'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-proposed FreqAI experiment specs (issue #47)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_propose = sub.add_parser("propose", help="generate + register N specs")
    p_propose.add_argument("--count", type=int, default=3)
    p_propose.add_argument("--regime", default=None)
    p_propose.add_argument("--model", default=None)
    p_propose.add_argument("--provider", default=None)

    p_eval = sub.add_parser("evaluate", help="evaluate pending freqai candidates")
    p_eval.add_argument("--limit", type=int, default=None)

    sub.add_parser("report", help="comparison table of freqai experiments")

    p_batch = sub.add_parser("run-batch", help="propose + evaluate + report")
    p_batch.add_argument("--count", type=int, default=3)
    p_batch.add_argument("--regime", default=None)
    p_batch.add_argument("--model", default=None)
    p_batch.add_argument("--provider", default=None)

    args = parser.parse_args(argv)

    if args.cmd in ("propose", "run-batch"):
        if not (os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")):
            print("No LLM API key set (DEEPSEEK_API_KEY or ANTHROPIC_API_KEY).",
                  file=sys.stderr)
            return 2
        results = propose_freqai_specs(
            count=args.count, target_regime=args.regime,
            model=args.model, provider=args.provider,
        )
        for r in results:
            if r["success"]:
                print(f"registered: {r['name']} (id={r['strategy_id']})")
            else:
                print(f"failed: {r['error']}")
        if args.cmd == "propose":
            return 0 if any(r["success"] for r in results) else 1

    if args.cmd in ("evaluate", "run-batch"):
        names = evaluate_pending(limit=getattr(args, "limit", None))
        print(f"evaluated: {', '.join(names) if names else '(none pending)'}")

    if args.cmd in ("report", "run-batch"):
        print(build_report())

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main(sys.argv[1:]))
