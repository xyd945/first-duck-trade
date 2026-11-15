# H1 paper start-readiness

PR #56 supplies the isolated paper execution path. It does not authorise launch or live trading. Hermes reviews and merges; G0 sign-off, the selected configuration and Yudi's notional-mapping approval remain launch prerequisites. No public-data paper service, legacy service or Compose stack was started during this work.

## Verified gates

- Pinned engine H1 suite: **144 passed**, no failures or skips; 15 upstream pandas warnings.
- Orchestrator suite: **721 passed, 14 skipped**, no failures; 7 existing warnings. Engine-dependent skips run in the separate engine suite. The counts overlap and must not be added as unique tests.
- Archived development replay: **120/120 daily pair targets matched**, both raw and risk-adjusted agreement **100%**, over **2–31 December 2021**. It produced **32 real Freqtrade dry-run order records**.
- Subprocess tests run the real service loop, pinned engine, SQLite journals and OS signals with exchange I/O replaced by synthetic fixtures and sockets denied. They cover ready acknowledgement, duplicate starts, stop/status, foreign-PID refusal, constructor failure, stalled startup, failed acquisition, partial entry and restart without duplicate orders.
- Weekly CLI subprocess tests cover complete evidence, missing decisions/NAV, late decisions, durable error receipts, corrupt data, changed strategy identity, placeholders, cost-check failure and idempotency. An interrupted review retains an error row and its denominator.
- GitHub currently reports no checks for this branch. Local verification is not a claim of green CI.

The archived comparison is correctness evidence, not forward performance or a G2 pass. `h1-paper-replay/comparison.json` contains derived weights only; raw candles and runtime databases stay private. The adjacent manifest records source hashes and input snapshot provenance.

## Commands for 19 December 2026

Run from repository root. Install `docker/requirements-h1-engine.txt` into a separate Python 3.12 environment named `.venv-h1`. The service requires **Freqtrade 2026.10 / ccxt 4.5.90**, keyless `okx`, spot USDT pairs, a **10,000 USDT** wallet and **400-candle** preload.

After approvals and selection, at the **2026-12-19 00:00 UTC** decision:

```sh
.venv-h1/bin/python user_data/scripts/h1_service.py start
.venv-h1/bin/python user_data/scripts/h1_service.py status
.venv-h1/bin/python user_data/scripts/h1_service.py stop
```

`start` and direct `run` refuse dates before 19 December. A matching `selected` row in `user_data/data/trial_ledger.sqlite` is required. `--placeholder` explicitly permits M45 diagnostics only; those runs are never G2 evidence. Never mix placeholder and selected state directories.

`start` waits for a fresh **ready** heartbeat from its own child, not merely the startup heartbeat. It waits at most 300 seconds. A halted, exited or timed-out child causes failure; timeout cleanup terminates and reaps only that child. `status` distinguishes a held process lock from readiness. `stop` verifies the PID's command and directory, sends SIGTERM and waits up to 30 seconds for lock release. It reports failure rather than claiming an unconfirmed stop. No automatic reboot launcher is installed; restart is manual.

State lives under `user_data/data/h1-paper/`:

- `service.log`, `heartbeat.json`, `service.lock` — operations.
- `identity.json` — selected configuration, rules, parameters and source commit.
- `trades.sqlite` — Freqtrade simulated orders/trades.
- `journal.sqlite` — reconciled risk, batches, first dated tick errors, reference opens and daily NAV.
- `data/` — locally captured completed candles.

Changed identity refuses reuse of the directory. Do not delete or edit journals to clear a risk latch or erase missed decisions. Stop, preserve evidence and reconcile explicitly. A monitoring gap over 600 seconds can latch the existing risk gate. Recovery does not automatically reset it.

The single `research_h1.designated_portfolio_notional` remains **250000, pending Yudi's OK**. The service validates registered rules separately so this mapping remains a one-value change. It is an explicit paper reserve mapping, not actual capital or approval.

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
- `error` preserves dated operational failures or scoring failure.
- `invalid` excludes placeholders and changed selected strategy identity.

Exit 0 means `ok`; exit 1 means a written non-`ok` row; argument/preflight errors exit 2. The JSON output contains the actual ledger row. Inspect its JSON `notes` for checks and evidence coverage.

Every cumulative day/pair stays in the target comparison denominator. Missing, error, blocked, expired or unfinished decisions cannot count as adjusted matches. Raw target agreement is separate from risk-adjusted agreement. Both use a **0.5 percentage-point tolerance**, with the existing **95%** match threshold. When a rerun cannot be computed, `comparison_available=false` accompanies a conservative zero-match denominator; it is not a measured mismatch rate.

Net metrics independently replay real SQL fills against captured closing marks and must reconcile to daily NAV. The cost check weights signed fill-versus-daily-open displacement plus the configured **0.15% aggregate simulated fee** by reference notional, and compares it with **0.225% per side**. This retains the accepted base-cost convention, including its embedded slippage allowance; it is not a separately measured exchange fee. No fills means the cost check is NULL, not passed. Full G2 duration, Monday-count and Sharpe gates remain outside this writer.

## Reproduction

```sh
PYTHONPATH=user_data/scripts .venv-h1/bin/python -m pytest tests/test_h1*.py -q
.suite-venv/bin/python -m pytest tests/ -q
.venv-h1/bin/python user_data/scripts/h1_replay.py --data /path/to/h1-dev-bounded --output /new/path/replay
```

The replay defaults to 2–31 December 2021 and rejects data beyond the development cutoff. It uses archived opens and explicitly flat offline books. These are replay assumptions, not historical bid/ask evidence or measured public-data slippage. No selection/evaluation window or public-data forward trading ran. Live credentials, exchange orders, restart-on-reboot and production network availability were not tested.
