# H1 paper start-readiness

PR #56 supplies the isolated paper execution path. Its recovery fix adds audited operator commands and preserves completed decisions when later acquisition fails. Neither change authorises launch or live trading. Hermes reviews and merges; G0 sign-off, the selected configuration and Yudi's notional-mapping approval remain launch prerequisites. No public-data paper service, legacy service or Compose stack was started during this work.

## Verified gates

- Pinned engine H1 suite: **182 passed**, no failures or skips; 15 upstream pandas warnings.
- Orchestrator suite: **722 passed, 26 skipped**, no failures; 7 existing warnings. Engine-dependent skips run in the separate engine suite. The counts overlap and must not be added as unique tests.
- Archived development replay: **120/120 daily pair targets matched**, both raw and risk-adjusted agreement **100%**, over **2–31 December 2021**. It produced **32 real Freqtrade dry-run order records**.
- Subprocess tests run the real service loop, pinned engine, SQLite journals and OS signals with exchange I/O replaced by synthetic fixtures and sockets denied. They cover a first-tick ticker failure followed by readiness within five minutes, lock contention, stop/status, foreign-PID refusal, failed starts, partial entry and restart without duplicate orders, malformed/stale risk state, stopped-only reconciliation, and audited code reidentification.
- Weekly CLI subprocess tests cover complete evidence, missing decisions/NAV, late decisions, durable error receipts, corrupt data, changed strategy identity, placeholders, cost-check failure and idempotency. A decision finished at 00:01 retains **4/4** adjusted matches after a 03:00 latch or reconciliation; a 10:00 code change vetoes the next day's decision instead. Unfinished decisions and legacy batches without completion timestamps remain vetoed. An interrupted review retains an error row and its denominator.
- Check GitHub separately at the final PR head. Local verification is not a claim of green CI.

The archived comparison is correctness evidence, not forward performance or a G2 pass. `h1-paper-replay/comparison.json` contains derived weights only; raw candles and runtime databases stay private. The adjacent manifest records source hashes and input snapshot provenance.

## Dedicated pinned checkout

Hermes must create a durable, detached worktree from the **merge commit of this recovery fix PR**, tagged `h1-paper-v1`. Set `FIX_MERGE_SHA` to that verified commit before running these commands from the normal repository checkout:

```sh
git fetch origin
git tag -a h1-paper-v1 "$FIX_MERGE_SHA" -m "Pinned H1 paper service v1"
git worktree add --detach ~/GitHub/first-duck-trade-h1 h1-paper-v1
cd ~/GitHub/first-duck-trade-h1
git rev-parse HEAD
git status --short
```

If the tag already exists, verify its commit instead of recreating or moving it. Verify `HEAD` equals `FIX_MERGE_SHA` and tracked files are clean. Do not pull, switch branches, edit tracked code or update dependencies in this worktree during the forward run. Pulls and development in other worktrees do not change this checkout. No tag or durable deployment worktree was created by the tests.

Install the pinned engine into this worktree's own `.venv-h1`. Provision the approved selection ledger using SQLite's backup facility at `user_data/data/trial_ledger.sqlite`, or consistently use the existing ledger's absolute path with `--ledger`. Untracked data and environments are not shared automatically between worktrees. Keep the same state directory and ledger throughout the forward record; never initialise a replacement directory to bypass a latch.

## Commands for 19 December 2026

Run from repository root. Install `docker/requirements-h1-engine.txt` into a separate Python 3.12 environment named `.venv-h1`. The service requires **Freqtrade 2026.10 / ccxt 4.5.90**, keyless `okx`, spot USDT pairs, a **10,000 USDT** wallet and **400-candle** preload.

After approvals and selection, from the pinned worktree root, launch at **2026-12-19 00:00:00 UTC**:

```sh
.venv-h1/bin/python user_data/scripts/h1_service.py start
.venv-h1/bin/python user_data/scripts/h1_service.py status
.venv-h1/bin/python user_data/scripts/h1_service.py stop
```

The exact initial window is **2026-12-19 00:00:00–00:05:00 UTC**. `start` and direct `run` refuse before 00:00:00. Acquisition and the decision must complete by 00:05:00; after that instant the decision is missed, not backfilled. Starting the process before the deadline is not sufficient. A matching `selected` row in `user_data/data/trial_ledger.sqlite` is required. `--placeholder` explicitly permits M45 diagnostics only; those runs are never G2 evidence. Never mix placeholder and selected state directories.

`start` waits for a fresh **ready** heartbeat from its own child, not merely the startup heartbeat. It waits at most 300 seconds, including through a non-latched halted heartbeat. A latched kill, exited child or readiness timeout causes failure; cleanup terminates and reaps only that child. The child retries brief lock contention for one second. `status` distinguishes a held process lock from readiness and exposes persisted `risk` even when stopped. `stop` verifies the PID's command and directory, sends SIGTERM and waits up to 30 seconds for lock release. It reports failure rather than claiming an unconfirmed stop. No automatic reboot launcher is installed; restart is manual.

State lives under `user_data/data/h1-paper/`:

- `service.log`, `heartbeat.json`, `service.lock` — operations.
- `identity.json` — selected configuration, rules, parameters and source commit.
- `trades.sqlite` — Freqtrade simulated orders/trades.
- `journal.sqlite` — reconciled risk, batches, first dated tick errors, reference opens, daily NAV and append-only latch/reconcile/reidentify receipts.
- `data/` — locally captured completed candles.

Identity still includes `git_sha`; an unacknowledged code change refuses reuse of the directory. Do not delete or edit journals or identity files to clear a latch, accept code or erase missed decisions.

The single `research_h1.designated_portfolio_notional` remains **250000, pending Yudi's OK**. The service validates registered rules separately so this mapping remains a one-value change. It is an explicit paper reserve mapping, not actual capital or approval.

### Latch and reconciliation

Inspect top-level `risk.kill_switch_active` and `risk.reason` in `status`, not only the final `stopped` heartbeat. A failed fetch grants nothing and records an error without manufacturing an empty risk snapshot. Failed ticker/book acquisition does not refresh the last risk timestamp. A gap exceeding **600 seconds** therefore still latches on the next evaluation, as do stale/invalid evaluated inputs and wallet/SQL reconciliation mismatches. Healthy data alone never clears a latch.

After investigating the outage or discrepancy, stop the service and confirm `running=false`. Preserve a stopped-state backup before recovery. Use the same pinned checkout, configuration, ledger and directory:

```sh
.venv-h1/bin/python user_data/scripts/h1_service.py stop
.venv-h1/bin/python user_data/scripts/h1_service.py status
.venv-h1/bin/python user_data/scripts/h1_service.py reconcile --reason "Investigated outage; wallet and SQL reconciled"
.venv-h1/bin/python user_data/scripts/h1_service.py start
.venv-h1/bin/python user_data/scripts/h1_service.py status
```

Run `start` immediately after successful `reconcile`, **within 10 minutes**; do not pause between them to inspect the receipt. The next risk evaluation must also fall within 600 seconds of the reconciled timestamp, or it latches again. Any stop longer than 10 minutes latches on the next start unless fresh stopped-service reconciliation precedes it. Preserve the command's audit output and inspect it after startup; if the window expires, investigate and reconcile again before restarting.

Replace the example reason with the actual incident explanation. `reconcile` holds the exclusive service lock, requires existing state and uses fresh keyless public marks, the dry-run wallet and SQL fills/inventory, plus the single configured designated-notional value. It never sends orders. Its transaction appends the previous/new state, reason and UTC time before updating current risk/snapshot. It retains the equity high-water mark and same-day loss anchor/halt; an ongoing 5% drawdown or invalid evidence refuses reset. A remaining daily halt still denies new exposure. Restarting during that same-day halt times out after 300 seconds and terminates the child; reconcile and start again within **00:00–00:05 UTC** on the next day, or that day's decision is missed. The malformed first-tick state is recoverable without inventing trades or deleting receipts.

Latch and reconciliation receipts veto only decisions not proven finished before the receipt was observed; later incidents do not erase completed matches. Their full audit receipts remain in the weekly notes. Terminal missed/blocked decisions are not replayed after recovery. A successful recovery command is not permission to ignore another risk denial.

### Deliberate code upgrades

Normal operation never updates the pinned checkout. If Hermes approves a code upgrade, preserve a stopped-state backup and verify the reviewed replacement tag/commit. The required order is **stop, switch, reidentify, reconcile, start** in the same detached worktree and state directory:

```sh
.venv-h1/bin/python user_data/scripts/h1_service.py stop
git switch --detach "$APPROVED_REPLACEMENT_TAG"
.venv-h1/bin/python user_data/scripts/h1_service.py reidentify --reason "Approved service fix; reviewed replacement commit"
.venv-h1/bin/python user_data/scripts/h1_service.py reconcile --reason "Approved upgrade; wallet and SQL reconciled"
.venv-h1/bin/python user_data/scripts/h1_service.py start
.venv-h1/bin/python user_data/scripts/h1_service.py status
```

`reidentify` requires a stopped service and only accepts a changed source commit; selection, strategy hash, parameters and rules must still match exactly. It appends old/new identity, reason and UTC time to the existing journal before atomically replacing `identity.json`. An interrupted replacement can leave a conservative extra audit receipt, never an unaudited accepted identity; inspect state and retry explicitly. The first decision under the new code remains a G2 divergence: the receipt day's decision if not yet finished, otherwise the next day's. This command does not reset risk: reconciliation is a separate step, followed immediately by `start` **within 10 minutes**. Existing weekly rows remain immutable; a changed identity cannot rewrite a review already recorded for that week.

## Execution and operational interpretation

- Targets and triggers use confirmed closes. Triggered sells settle before all triggered buys receive one common cash/risk factor. No leverage, financing from pending sales, or sale of band-skipped positions is introduced.
- At-most-once dispatch attempts are persisted before submission. A crash between receipt and simulated order may leave an unfilled target; it must not generate a duplicate on restart.
- Active batches expire at the next daily tick even when that new decision is late. Terminal missed/blocked receipts are not converted into active batches.
- Acquisition can retry within the initial five-minute decision window. Until all prior-day candles arrive, the service waits rather than using an older close as yesterday's NAV. If acquisition is still incomplete or late, the whole decision is marked missed, not backfilled. This conservative operational divergence remains in G2's denominator.
- Yesterday's exact closing NAV is captured before today's decision when complete candles and reconciled inventory are available, including when today's decision is late. Missing NAV is explicit; old prices are not substituted.
- The decision deadline is rechecked after reference acquisition. Future-dated timestamps, stale/missing marks, changed confirmed candles, and unreconciled wallet/SQL inputs fail closed.
- The first tick failure per UTC day remains a separate error receipt even if later ticks recover. Execution receipts keep attempted orders and targets. A failed journal write aborts processing rather than granting permission.

## Weekly forward writer

Run after the service has captured the final completed day's candle and NAV. Example for a Monday review on **28 December 2026**:

```sh
.venv-h1/bin/python user_data/scripts/h1_review.py --through 2026-12-27
```

`--directory` selects a non-default state directory. The writer uses the shared `TRIAL_LEDGER_PATH` environment variable, or defaults to `user_data/data/trial_ledger.sqlite`. If using another ledger, also pass that same path via the service's `--ledger` option. Do not create another ledger to evade the immutable weekly receipt.

Each row has `hypothesis=H1`, `split=forward`, and a cumulative window starting **2026-12-19**. The review key is the ISO UTC week containing the day after `--through`; use the completed Sunday cutoff for Monday reviews. Repeating the same week/cutoff/identity returns the original row. A different cutoff or identity for an already-recorded review week is rejected, not silently substituted. Interrupted rows remain `error`; the writer does not backfill them.

- `ok` means a completed ordinary review, **not a G2 pass**. A cost or agreement check can fail in an `ok` row.
- `missed` preserves missing evidence or unfinished/late daily decisions.
- `error` preserves missing/unfinished decisions with dated errors, or scoring failure. A later tick error never rewrites a finished decision; all recorded in-window errors remain in `notes.operational_errors`.
- `invalid` excludes placeholders and changed selected strategy identity.

Exit 0 means `ok`; exit 1 means a written non-`ok` row; argument/preflight errors exit 2. The JSON output contains the actual ledger row. Inspect its JSON `notes` for checks and evidence coverage.

Every cumulative day/pair stays in the target comparison denominator. Missing, error, blocked, expired or unfinished decisions cannot count as adjusted matches. Raw target agreement is separate from risk-adjusted agreement. Both use a **0.5 percentage-point tolerance**, with the existing **95%** match threshold. When a rerun cannot be computed, `comparison_available=false` accompanies a conservative zero-match denominator; it is not a measured mismatch rate.

`notes.operational_divergences` retains every in-window latch, reconciliation and code-change receipt, whether or not it vetoes a decision. The adapter records `finished_at` once, after final settlement and reconciliation; restarts never move it. A latch or reconciliation vetoes its day's decision unless the batch is finished and `finished_at` is strictly earlier than the receipt's `observed` time. Missing timestamps and equal timestamps are conservatively vetoed. A later `reidentify` leaves that completed day unchanged and vetoes the next day's decision; otherwise it vetoes the receipt day's decision. Persistent latches still leave subsequent blocked batches divergent. Operational fetch errors alone do not impose an additional veto on finished decisions.

Net metrics independently replay real SQL fills against captured closing marks and must reconcile to daily NAV. The cost check weights signed fill-versus-daily-open displacement plus the configured **0.15% aggregate simulated fee** by reference notional, and compares it with **0.225% per side**. This retains the accepted base-cost convention, including its embedded slippage allowance; it is not a separately measured exchange fee. No fills means the cost check is NULL, not passed. Full G2 duration, Monday-count and Sharpe gates remain outside this writer.

## Reproduction

```sh
PYTHONPATH=user_data/scripts .venv-h1/bin/python -m pytest tests/test_h1*.py -q
.suite-venv/bin/python -m pytest tests/ -q
.venv-h1/bin/python user_data/scripts/h1_replay.py --data /path/to/h1-dev-bounded --output /new/path/replay
```

The replay defaults to 2–31 December 2021 and rejects data beyond the development cutoff. It uses archived opens and explicitly flat offline books. These are replay assumptions, not historical bid/ask evidence or measured public-data slippage. No selection/evaluation window or public-data forward trading ran. Live credentials, exchange orders, restart-on-reboot and production network availability were not tested.
