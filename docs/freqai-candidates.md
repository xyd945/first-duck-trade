# FreqAI candidates (issue #47)

Status: Phase 1 + Phase 2 landed — baseline strategy, spec→renderer path, orchestrator
lifecycle integration with mandatory walk-forward. Phase 3 (live deployment of ML
strategies) is deliberately NOT implemented; see "What's excluded and why".

## What this is

A second candidate type for the strategy factory. Rule-based candidates encode a
human-readable trading thesis as entry/exit conditions; FreqAI candidates train a
model (LightGBM, currently) that predicts the forward return over a fixed horizon
from engineered features, and trade on thresholded predictions.

Both types run the SAME lifecycle: register → full backtest → R7 gates →
promote/retire, with results, attribution, and failure memory in the same registry.

## How it differs from the rule-based path

| | Rule-based | FreqAI |
|---|---|---|
| Spec | `strategy_spec.py` JSON (entry/exit condition DSL) | `freqai_spec.py` JSON (features + target + model + thresholds) |
| Rendered file | Full strategy code (conditions inlined) | **Declarations only** — feature keys, thresholds, risk numbers |
| Executable logic | In the rendered file (validated) | In `strategies/base_freqai.py` + `indicators/freqai_features.py` (hand-written, reviewed once) |
| Backtest image | `freqtradeorg/freqtrade:stable` | `freqtradeorg/freqtrade:stable_freqai` (ships LightGBM) |
| Backtest service | `freqtrade-backtest` | `freqtrade-freqai` (4 CPU / 6G limits) |
| Config | shared `config.json` | per-candidate `configs/freqai/<Name>.json` (unique model identifier) |
| Runtime | seconds–minutes | tens of minutes (retrains every `backtest_period_days` per pair) |
| Walk-forward | opt-in (`R7_WALK_FORWARD`) | **mandatory** — skipped/failed WF blocks promotion even in non-strict mode |
| Hyperopt rescue | yes | no (nothing for hyperopt to search — the tunable surface is the spec) |
| Deployment | via reconciler | **excluded** (Phase 3) |

## The safety model

The spec is the trust boundary, exactly like the rule-based factory:

- **Feature whitelist.** A spec picks feature keys (`rsi`, `funding`, `macro_vix`, …)
  from `indicators/freqai_features.py`. Every key maps to hand-written computation.
  Unknown keys are rejected at validation time.
- **Declarations-only rendering.** The rendered candidate contains zero executable
  statements — `validate_freqai_strategy_file` REJECTS any method definition in the
  subclass, so a spec (or a template bug) cannot override feature engineering,
  targets, or entry logic with unreviewed code.
- **Bounded numerics.** Model family whitelist (`LightGBMRegressor`), bounded
  LightGBM params, horizon 4–72 candles, train window 30–180 days, thresholds with
  entry strictly above exit, negative stoploss.
- **Look-ahead containment.** Features never see the future: external/macro series
  are pre-shifted by their source modules, fills are ffill-only (no bfill). The one
  legitimate `shift(-N)` — the training target — lives in the trusted feature
  library, NOT in candidate files, so the standard look-ahead validator applies to
  rendered candidates unchanged.

## Files

```
user_data/strategies/base_freqai.py           # all executable ML strategy logic
user_data/strategies/candidates/base_freqai.py  # copy — freqtrade's resolver only
                                                # sys.path's the candidate's own dir;
                                                # refreshed on every materialization
user_data/strategies/FreqaiBaselineLGBM.py    # hand-written baseline (smoke tests)
user_data/indicators/freqai_features.py      # feature/target whitelist library
user_data/scripts/freqai_spec.py             # validate / render / register / purge
user_data/configs/config-freqai-base.json    # committed base config (dry-run, no secrets)
user_data/configs/freqai/<Name>.json         # rendered per-candidate configs
user_data/freqai_specs/baseline_lgbm.json    # committed reference spec
user_data/strategies/candidates/<Name>.py    # rendered candidates
user_data/strategies/candidates/<Name>.freqai.json  # spec sidecar (model family etc.)
```

## Running one manually

```bash
# validate + register a spec as a candidate
docker exec ft-orchestrator python /app/user_data/scripts/freqai_spec.py \
    validate /app/user_data/freqai_specs/baseline_lgbm.json
docker exec ft-orchestrator python /app/user_data/scripts/freqai_spec.py \
    register /app/user_data/freqai_specs/baseline_lgbm.json

# evaluate JUST that candidate through the real lifecycle
docker exec ft-orchestrator python -c "
import sys; sys.path.insert(0, '/app/user_data/scripts')
import orchestrator; orchestrator.job_backtest_candidates(only_name='FreqaiLgbmBaseline')"

# or a raw backtest of the hand-written baseline
docker compose --profile backtest run --rm freqtrade-freqai backtesting \
    --config /freqtrade/user_data/configs/config-freqai-base.json \
    --strategy FreqaiBaselineLGBM --freqaimodel LightGBMRegressor \
    --timerange 20260501-20260701 --timeframe 1h
```

The scheduled Sunday `job_backtest_candidates` picks up registered FreqAI candidates
automatically — no separate job.

## Promotion and retirement

FreqAI candidates face every existing gate (regime-conditional floor, beat-buy-hold,
walk-forward, correlation) plus one extra rule: promotion requires a REAL
walk-forward pass (`is_strict_pass`), regardless of `STRICT_PROMOTION_GATES`. The
orchestrator force-runs walk-forward for baseline-passing freqai candidates even
when `R7_WALK_FORWARD` is off; candidates that already failed baseline skip the
windows (they can't promote anyway, and each window costs model training) but
still record an explicit `SKIP_WF` verdict that blocks promotion.

ML-specific failure verdicts in the registry / failure memory:

| Verdict | Meaning |
|---|---|
| `FAIL_WF_INCONSISTENT` | too few positive-sharpe walk-forward windows (unstable model) |
| `FAIL_WF_UNSTABLE` | one window carried the full backtest (overfit signature) |
| `FAIL_WF_CRASH` | training/backtest crashed in a window |
| `FAIL_ML_NO_WALKFORWARD` | walk-forward produced no real evidence (skip) — blocked by the mandatory-WF rule |

Model artifacts (`user_data/models/<Name>/`) are purged after evaluation, win or
lose — a full evaluation leaves O(100MB) per candidate otherwise.

## LLM-proposed experiments (Phase 3a)

`freqai_generator.py` lets the LLM propose experiment specs — JSON only, never
code. Every proposal passes through the same `validate_freqai_spec` +
materialize/register path as a hand-written spec; a hallucinated feature or
out-of-bounds param is a clean rejection whose error message is fed back to the
LLM for a retry (up to 2 extra turns per spec).

Anti-drift property: the system prompt's feature list and bounds are BUILT from
the validator's constants (`freqai_spec` bounds, `freqai_features` keys) — a
test asserts the sync, so extending the whitelist automatically teaches the
prompt and the two can never disagree.

Batch-first workflow (the operator's real loop — manual rounds of propose /
evaluate / compare):

```bash
docker exec ft-orchestrator python /app/user_data/scripts/freqai_generator.py \
    run-batch --count 3          # propose + evaluate + report in one go
# or individually:
#   propose --count N [--regime trending]
#   evaluate [--limit N]         # full lifecycle per pending freqai candidate
#   report                       # comparison table of every experiment so far
```

The prompt carries the ML failure memory (retired freqai experiments with
their feature sets, horizons, thresholds, and verdicts — sourced from the spec
sidecars) plus in-batch diversity pressure (each proposal sees the batch's
prior specs and is told to differ structurally). Factory-owned fields
(`spec_type`, `generation_id`) are stamped server-side — the LLM's values are
ignored. Names are auto-suffixed on registry collision.

Weekly automation exists but is OFF by default: set `FREQAI_WEEKLY_COUNT=N`
to make the Saturday generation job add N LLM-proposed FreqAI specs alongside
the rule-based batch. Leave it 0 until a few manual batches look sane.

## What's excluded and why (Phase 3b)

- **Deployment**: `get_deployment_eligible` filters `spec_type='freqai'`. The
  reconciler's `ft-deployed-*` containers run the non-ML image, and live FreqAI
  needs retraining/model-freshness/low-confidence ops that don't exist yet. A
  promoted FreqAI strategy means "research-validated", not "deployable".
- **Hyperopt rescue**: excluded — rescue for an ML candidate is a new spec, not a
  parameter sweep.

## Positive-control research runs

`FREQAI_BACKTEST_END=YYYYMMDD` (env, unset in production) anchors the FreqAI
evaluation window to a historical end date: the 180-day full backtest, its
mandatory walk-forward windows, and the regime-fraction window all end there,
so every gate judges the same market the backtest saw. Purpose: discriminate
"the factory can't find edge" from "this window has no long edge" by pointing
the same search at a rising window (e.g. Apr–Oct 2025, BTC +62%). Run it via:

```bash
docker exec -e FREQAI_BACKTEST_END=20251020 ft-orchestrator \
    python /app/user_data/scripts/freqai_generator.py run-batch --count 3
```

Caveats: it's a research knob — failure-memory entries from control windows
mix with production ones (the report's per-experiment rows stay honest), and
a candidate promoted from a historical window is research-validated for THAT
window only (deployment exclusion applies to freqai regardless).

## Data provisioning

FreqAI reads the same OKX futures feathers the rule-based backtests use
(`user_data/data/okx/futures/`), refreshed weekly by the orchestrator's
`job_fetch_ohlcv` (Saturday 19:30 UTC). Two constraints that job imposes:

- **Depth**: it downloads 400 days — the 180-day backtest window plus up to
  180 days of pre-window training data (the spec validator's `train_period_days`
  ceiling) plus slack. A FreqAI backtest needs training history BEFORE its
  timerange start; if you widen the backtest window or raise the train-period
  bound, resize the fetch depth with it.
- **Pairs**: the job unions the pair lists of `config.json` and
  `config-freqai-base.json`, so a pair added only to the FreqAI config still
  gets downloaded. (The lists are identical today; the union exists so drift
  is harmless rather than a silent missing-data failure.)

## Operational cost

Measured on the live host (baseline spec, 3 pairs, 1h timeframe): a 6-month FreqAI
backtest — ~26 training windows × 3 pairs of LightGBM on ~1.4k-row windows — runs in
roughly half a minute to a few minutes; the full lifecycle (6-month backtest +
3 × 60d walk-forward + gates) completed in ~70 s during the shakedown. Two reasons
it stays cheap: LightGBM is fast on data this small, and FreqAI caches per-window
predictions under the candidate's identifier, so walk-forward windows that overlap
the full backtest reuse them instead of retraining. Bigger feature sets, more pairs,
or heavier model families will grow this quickly — `FREQAI_BACKTEST_TIMEOUT`
(seconds, default 5400) caps each run, and walk-forward windows retry once on
transient crashes (see `run_walk_forward`).
