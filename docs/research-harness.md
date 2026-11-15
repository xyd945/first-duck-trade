# Frozen research harness

This is the fix-2 research boundary with fix-4 net accounting, not a trading system or a G1 verdict. H1's strategy and baselines are separate work. No real data, API keys, strategy performance or account access is included in these fixtures. See `docs/cost-risk-accounting.md` for the accounting and paper-risk contracts.

## Pause and environment

Python and Compose default these flags to false:

- `ENABLE_LLM_GENERATION`: rule/FreqAI generation, scheduled reflection and LLM regime override.
- `ENABLE_STRATEGY_CRITIC`: the direct critic entry point returns `SKIP` without an LLM call.
- `ENABLE_HYPEROPT_RESCUE`: scheduled and manual hyperopt search.

Fix 4 additionally hard-disables managed REST starts, forced liquidation and dynamic deployment starts through `paper_risk.EXECUTION_ENABLED = False`. Environment flags cannot turn this on. Do not start the full Compose stack, orchestrator or trading containers; manually launching Freqtrade bypasses these Python guards. Enabling a factory flag is not approval to restart execution. The old promotion/status and causality defects are separate fixes.

Research runs require:

- `RESEARCH_CONFIG`: an explicit JSON policy. Relative snapshot paths resolve beside this file.
- `TRIAL_LEDGER_PATH`: an absolute path to the shared SQLite ledger. Locally it defaults to `user_data/data/trial_ledger.sqlite`. Both repositories can target the same durable database; do not create a fresh ledger to evade evaluation consumption or N counting.
- `RESEARCH_GIT_SHA`: the full 40-character commit SHA only when `.git` is unavailable, such as inside a container. Otherwise the runner reads `git rev-parse HEAD`. Source hashes also capture the actual files, including uncommitted changes.

For local use, set a host ledger path rather than copying the container's `/app/...` default from `.env.example`. Preserve and back up the ledger; do not delete or reset it to reopen evaluation.

## Policy and snapshots

Copy `user_data/configs/research-h1.example.json` to an ignored local policy file. The example's medium ladder and 45% target are an example configuration, not the selected winner. The G0 grid is run by the later H1 workflow.

`warmup` and `splits.dev`, `splits.selection`, `splits.eval` contain **inclusive UTC dates**, formatted `YYYYMMDD`. Windows must be adjacent and non-overlapping. The default policy is:

| Window | Start | End |
|---|---|---|
| Warm-up | 20180111 | 20190113 |
| Development | 20190114 | 20211231 |
| Selection | 20220101 | 20240630 |
| Evaluation | 20240701 | 20261218 |

S1 ends at selection's last day. S2 ends at evaluation's last day. Both start at the warm-up start, while later-listed pairs naturally have shorter histories. The pair set is BTC-USDT, ETH-USDT, SOL-USDT and XRP-USDT, OKX spot `1Dutc`.

The freezer accepts local Freqtrade files named `BTC_USDT-1d.feather`, `ETH_USDT-1d.feather`, `SOL_USDT-1d.feather`, `XRP_USDT-1d.feather`. Obtain these from confirmed public OKX candles; the freezer does not download data or certify its provider. If a `confirm` field exists, every included bar must be confirmed. Dates must be unique, increasing and at midnight UTC; prices/volume must be finite and valid. The source may contain later bars, but only the requested prefix is frozen.

Canonical representation:

- One UTF-8 CSV per pair, LF line endings and a final newline.
- Header exactly `ts,open,high,low,close,volume`.
- `ts` is integer Unix milliseconds at the UTC candle open.
- Numeric fields use plain decimal notation without trailing fractional zeroes.
- Each pair hash is SHA-256 of those exact CSV bytes.
- `data_snapshot_hash` is SHA-256 of the concatenated lowercase hexadecimal pair hashes, in lexicographic pair-name order, with no delimiter or trailing newline.

The manifest names the venue, bar interval, explicit bounds and each pair file/hash. The pinned composite hash is **not** a hash of Feather bytes or of the manifest. Every run verifies the actual CSV bytes and their canonical form before staging temporary engine-format Feather files. Snapshot paths cannot escape the snapshot directory.

Using an existing isolated Python environment:

```bash
export RESEARCH_CONFIG="$PWD/user_data/configs/research-h1.json"
export TRIAL_LEDGER_PATH="$PWD/user_data/data/trial_ledger.sqlite"
.test-venv/bin/python user_data/scripts/research.py freeze S1 \
  --source user_data/data/okx --destination user_data/data/research/S1
```

The destination must be new: no silent replacement. Pin the printed `data_snapshot_hash` in `snapshots.S1.hash`, and keep the manifest path in `snapshots.S1.manifest`. The example leaves both hashes empty deliberately; no real snapshot is claimed.

After selection is recorded and **2026-12-19 00:10 UTC** has passed:

```bash
.test-venv/bin/python user_data/scripts/research.py freeze S2 \
  --source user_data/data/okx --destination user_data/data/research/S2
```

Pin the S2 hash in the same manner. The freezer refuses a cutoff until the final bar has closed plus ten minutes. S2's canonical CSV prefix through 2024-06-30 must equal S1 byte for byte. Evaluation rechecks that prefix. Failed freeze directories are not reused; inspect them rather than treating an incomplete output as a valid snapshot.

## Configuration identity

H1 `params` contains `ladder`, `sigma_target`, `vol_window`, `rebalance_band` and `cost_rate`. `config_hash` is SHA-256 of the exact strategy source bytes immediately followed by UTF-8 JSON of the first four fields, with sorted keys and separators `(',', ':')`. `cost_rate` is recorded but excluded from this statistical hash. Dates, snapshot hashes and execution fingerprints are not silently folded into G0 N.

H1 engine configuration must use OKX spot, daily bars, the four static pairs, `dry_run: true`, numeric `dry_run_wallet`, and `fee` equal to `cost_rate`. The policy must declare `costs.fee_rate`, `spread_rate`, `slippage_rate` and `stress_multiplier`; their combined one-side rate must equal `cost_rate`. The full quoted spread contributes half per side. The example uses taker fees plus five basis points slippage, doubled. The frozen engine config includes `research_costs`. Exchange credentials and config includes are rejected. H1 rejects a hidden strategy parameter sidecar. The staged engine config exposes the registered values at `self.config['research_params']`; the later hand-written H1 strategy must consume those values. Tests do not claim that an arbitrary strategy obeys them.

A separate registration fingerprint binds the effective engine config, source/helper/sidecar hashes, split policy, S1 hash and declared run kind. All needed strategies/helpers must exist before registration. S2 is intentionally not required then because it does not exist yet. Register base/gate/Bybit-cost sensitivity variants before locking selection; their differing execution fingerprints do not inflate G0's strategy/parameter `config_hash`.

## Running and selecting

The following are workflow commands, not permission to run H1 before its accounting, strategy checks and G0 approval are ready:

```bash
.test-venv/bin/python user_data/scripts/backtest_runner.py H1Strategy \
  --split dev --timeframe 1d --config configs/research-engine.json
.test-venv/bin/python user_data/scripts/backtest_runner.py H1Strategy \
  --register-eval --timeframe 1d --config configs/research-engine.json
.test-venv/bin/python user_data/scripts/backtest_runner.py H1Strategy \
  --split selection --timeframe 1d --config configs/research-engine.json
.test-venv/bin/python user_data/scripts/research.py select SELECTION_TRIAL_ID
```

Use `--verdict baseline` or `--verdict sensitivity` for those preregistrations and runs. Ordinary successful runs are `ok`; identical successful repeats become `rerun`. Every attempted backtest/hyperopt batch/evaluation gets a ledger row, including validation failures and engine crashes. A crash before finalisation remains `error` with `notes.state = running`, not a non-standard verdict.

`select` copies a completed ordinary selection trial into the single `selected` row for the hypothesis. It cannot choose a failed/baseline/sensitivity trial, overwrite the choice or backfill it at/after forward start (2026-12-19 00:00 UTC under the example policy). It does not compute the H1 winner or G1: the later workflow must apply the preregistered net-Sharpe/tie-breaking rule before invoking it.

Legacy `--mini` means the configured development window; `--full` means selection. Legacy `--evaluate` still means development plus selection checks, **not** untouched evaluation. There is no moving six-month or unrestricted default. An optional `--timerange` uses Freqtrade's exclusive end and must remain inside the requested dev/selection window.

## Untouched evaluation

```bash
.test-venv/bin/python user_data/scripts/backtest_runner.py H1Strategy \
  --evaluation REGISTRATION_HASH --timeframe 1d --config configs/research-engine.json
```

Evaluation requires a matching `selected` ledger row and the closed S2 cutoff. Registration variants cannot be added after selection. SQLite consumes a registration atomically **before** reading S2, so concurrent callers cannot both open it. A later hash error, engine timeout or failed analysis does not restore the consumed registration. Re-registering identical inputs is idempotent and never resets consumption.

All preregistered grid configurations can be evaluated for the G1 comparison matrix; the selection row names the winner, not the only eligible evaluation column. After any evaluation opens, the whole hypothesis cannot return to dev/selection search. The evaluation entry point is manual, never scheduled, and `strategy_registry.record_backtest` rejects evaluation results. The generator/reflector use that separate registry, not the shared trial ledger.

## Continuous-path artefacts and accounting handoff

Each run executes from the configured **development start** through the requested window's exclusive end. The warm-up prefix remains available to indicators. Selection does not start a fresh wallet at 2022-01-01, and evaluation does not start one at 2024-07-01. Each longer run replays the same frozen prefix rather than carrying a mutable live account between processes.

The run directory under `user_data/research_runs/TRIAL_ID/` contains frozen code, engine config, descriptor, allowed data prefix and Freqtrade's exported archive. Only that directory is mounted into the one-off backtest service; its data submount is read-only, cache is disabled and Compose uses `--no-deps`. No ledger, sibling results, live data directory or S2 data is mounted into a dev/selection run.

`split.json` retains trades intersecting the scoring window, including positions opened before it, plus the last pre-window wallet state. `split.feather` retains the window's wallet records without reinitialising balances. The original full-path archive remains available for G1's full-path matrix. Missing wallet exports, shortened engine dates and incomplete ending coverage fail the run rather than falling back to optimistic console aggregates. Engine startup requirements must fit the supplied warm-up; the H1 card must check that explicitly.

The returned `timerange` is the scoring window; `path_timerange` is the engine interval. Whole-path engine metrics are kept under `path_metrics`, never presented as selection/evaluation net metrics. Actual filled-order quantities and frozen daily closes produce `net-equity.feather`; pre-bar wallet inventory is cross-checked, not mistaken for closing NAV. A validated terminal synthetic `force_exit` sale is omitted, leaving that inventory marked at the final close with no fictitious exit charge. Ambiguous exports fail closed.

Ledger `net_return`, `net_sharpe` and `max_drawdown` now contain independently computed split results. `n_trades_or_rebalances` counts actual fills in the split, excluding the synthetic terminal sale. Notes retain `opening_equity`, `ending_equity`, `cost_total`, `cost_model`, `equity_path`, `net_path_metrics`, `path_metrics` and the split artefact paths. A flat or single-return sample has NULL Sharpe rather than an invented statistic. Old promotion checks are not a substitute for the G0/G1 criteria.

## Shared ledger and search budget

The SQLite `trial_ledger` table has exactly:

```text
trial_id, created_utc, hypothesis, repo, git_sha, config_hash, data_snapshot_hash,
split, window_start, window_end, params_json, n_trades_or_rebalances,
net_return, net_sharpe, max_drawdown, verdict, notes
```

Ledger window ends are inclusive. The allowed verdicts are `ok`, `selected`, `g1_pass`, `g1_fail`, `missed`, `error`, `baseline`, `sensitivity`, `rerun`, `invalid`. G1/forward scoring is later work; this card does not manufacture those outcomes. `params_json` retains parameters, execution descriptor, run kind and any hyperopt batch's requested epochs. Engine epoch completion information stays in notes, distinct from requested epochs.

```bash
.test-venv/bin/python user_data/scripts/research.py budget
```

`N` is the distinct `config_hash` count on selection or eval, excluding `baseline` and `sensitivity` rows. Errors/invalid attempts count; identical hashes do not add N. Dev-only and forward rows do not add N. Row totals and hyperopt batch/epoch totals are separate diagnostics, not claims of independent discoveries. The report does not pad N to nine before nine configurations have actually been tried.

## Verification and limits

Tests use synthetic OHLCV/engine exports and a real local SQLite ledger. They exercise hash rejection, prefix identity, pause calls, split staging, crossing-position preservation, concurrency, crash consumption, failure accounting, G0 N, execution drift and CLI routing. They do not download bars, run Freqtrade, start schedulers/containers, call an LLM or place orders.

This is an application integrity boundary, not an operating-system sandbox against a malicious owner. A user able to edit code, rewrite the ledger or replace pinned hashes can bypass it. The retained backtest image has network access for public exchange market metadata; this change does not claim a network-isolated sandbox. Image/dependency pinning and broader replay guarantees remain fix 5.
