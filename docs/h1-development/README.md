# H1 development results — 15 November 2026

**Development evidence only. No configuration was selected. These returns are not G1 evidence or a forecast.**

The scored window is **2019-01-14 through 2021-12-31**, inclusive: 1,083 daily decisions. The only earlier candles are the registered indicator warm-up. No selection, evaluation or forward-window backtest ran. No bot, dry-run, scheduler, trading container, API key or exchange order was used.

## Evidence

- [Results table](results-table.md) and [machine-readable results](results.csv).
- [All 24 market trial ledger rows](ledger.csv), including the full recorded parameters, configuration/data hashes and engine-version notes. Only machine-local path prefixes were made repository-relative for publication; numeric fields, identifiers and hashes are unchanged.
- [Verification](verification.json): 24 successful market attempts, 24 finished dev rows, nine `ok`, three `baseline`, twelve `sensitivity`, no sealed-window rows and statistical N = 0. The shared ledger also contains four earlier synthetic diagnostics, all invalid; they are not market results or counted trials.
- [Frozen dev-prefix manifest](snapshot-manifest.json) and [download coverage/request receipts](coverage.json). This is a development prefix, **not the full S1**. Raw exchange candles are not committed to the public repository; redistribution terms have not been reviewed.
- [Execution evidence](execution-evidence.zip): each run's decisions, actual fill exports, daily net equity in CSV and Feather, and code fingerprints. The original Freqtrade archives and frozen candles remain in the internal task evidence.

All 24 runs used source commit `588dcf02abc7db6039e71da59c9950afad3e6002`, Freqtrade **2026.10** (`4cf0351e08d26e96a6429ef48103b726bb230cf8`) and **ccxt 4.5.90**. Each ledger row records the installed engine and ccxt versions under `notes.params`. Strategy and execution source remain unchanged in the subsequent evidence commit.

The snapshot hash is `ffb658c29d7a0799997a57efbdb6770bdf03f286a6ed1a5de0b8abb4618287d3`.

| Pair | First candle | Last candle | Rows | Gaps |
|---|---|---|---:|---:|
| BTC-USDT | 2018-01-11 | 2021-12-31 | 1,451 | 0 |
| ETH-USDT | 2018-01-11 | 2021-12-31 | 1,451 | 0 |
| SOL-USDT | 2020-10-01 | 2021-12-31 | 457 | 0 |
| XRP-USDT | 2020-01-01 | 2021-12-31 | 731 | 0 |

Every downloaded row has OKX `confirm = 1` and a 00:00 UTC timestamp. The download uses the same recommended OKX `history-candles` source directly, rather than the bulk Freqtrade wrapper. Inspection of the pinned wrapper showed that its last request can overshoot the requested end. Each direct response is bounded by `after <= 2022-01-01T00:00:00Z`, checked before storage, and checked for exact gap-free coverage. No selection-window candle was fetched by this task. The inherited venue report describes broader read-only history checks performed separately by the venue-check sibling; those are not inputs to these backtests.

## Metric definitions and checks

- **Net return:** final daily closing NAV / 10,000 USDT initial cash − 1. Positions are marked, not synthetically liquidated.
- **Net Sharpe:** mean daily NAV return / sample standard deviation × √365; risk-free rate zero.
- **Max drawdown:** positive peak-to-trough NAV loss, including initial capital in the peak series.
- **Turnover:** total absolute buy-plus-sell fill notional / mean daily closing NAV, over the complete window; **not annualised** and not divided by two.
- **Costs:** total USDT debited on actual fills. Gate charges 0.30% per side; base charges 0.15%. Fee and slippage are combined into the Freqtrade fee parameter. Base cases are complete reruns, not cost subtraction from unchanged gate fills, so fill counts may differ.
- **Fills:** actual entries, reductions and increases; not completed round trips. Synthetic terminal closes are excluded.

The export check independently recomputed the ledger metrics and total costs from actual orders. It compared closing NAV against adapter cash plus marked holdings on **every day of every run**; the largest difference was `1.7462298274040222e-10` USDT. It also checked nonnegative cash, all 1,083 decisions, every staged file's end date, pair eligibility, all nine distinct H1 configuration hashes, cost-only hash invariance, and the absence of selection/evaluation/forward rows. Freqtrade's wallet inventory is reconciled by `research.slice_result` before a run can succeed.

Target-versus-filled differences are recorded in `decision-batches.json`, with counts in `results.csv`. Those counts include band-skipped targets, fee-funded underfills and exchange precision/minimum-order effects; they are **not** a G2 tracking pass rate.

No rule, grid parameter or cost was chosen based on these returns. DSR, PBO, SPA, selection and G1 pass/fail were not run. Survivor selection and the development window's market regime make these historical returns optimistic. Paper risk caps are deliberately absent from historical research; they apply only to the paper phase.

## Inactive dry-run config

Path: `user_data/configs/h1_dryrun.json`.

It uses keyless `okx`, the four USDT pairs, `1d`, 10,000 USDT paper cash, unlimited dynamically sized stakes, four positions, 400 startup candles and the 0.15% base cost. REST is enabled; websockets, Telegram and the API server are disabled. The initial bot state is stopped. The single `designated_portfolio_notional: 250000` value remains **pending Yudi's OK**.

**This is a validated inactive template, not a start-ready trading service.** M45 is an explicit placeholder, not a selected configuration. The inherited strategy deliberately refuses non-historical startup. G0 sign-off alone does not remove that guard: the paper decision/execution adapter, reconciled risk-state input, selected configuration and activation review are still required before Hermes's planned **19 December 2026** start. This PR does not add a service, remove the guard or initialise invented capital. Use the existing engine requirements pin; any later container service must pin `freqtradeorg/freqtrade:2026.10`, never `stable`.

`freqtrade show-config` completed without validation errors. A separate test applies the strict `RunMode.DRY_RUN` schema without instantiating a bot. This matters because `show-config` may print a validation error yet return zero.

## Verification

Separate environments are required because engine and orchestrator dependencies have incompatible cffi constraints.

- Orchestrator suite: **701 passed, 0 failed, 5 skipped**, seven existing warnings. Skips are the four Freqtrade-dependent H1 modules plus the strict engine-schema test.
- Engine H1/config suite: **75 passed, 0 failed, 0 skipped**, fifteen upstream pandas deprecation warnings. It covers all the skipped tests above. One pure-config test runs in both environments; do not add suite counts as unique tests.
- The new notes check failed before its fix, then passed. The cost-sensitivity test failed before the runner supported it, then passed.
- The bounded downloader and export verifier executed all of their assertions on the actual downloaded data and market-run exports.
- No independent review approval or remote CI success is implied by these local checks.

## Reproduction

Run from the repository root with an isolated engine environment installed from `docker/requirements-h1-engine.txt`. The task used `../.engine-venv`; adjust only that executable path for a new checkout. Output directories must not already exist. Do not run Python with `-O`, because the evidence scripts use assertions as fail-closed self-checks.

```sh
../.engine-venv/bin/python docs/h1-development/download.py user_data/data/h1-dev-bounded
../.engine-venv/bin/python docs/h1-development/run.py
../.engine-venv/bin/python docs/h1-development/export.py
../.engine-venv/bin/freqtrade show-config -c user_data/configs/h1_dryrun.json --userdir user_data
```

The runner always appends to the shared trial ledger, including errors; it refuses to overwrite its local run directory. The exporter regenerates only the derived files in this evidence directory. Original raw candles, engine archives and existing ledger history are not overwritten. The source/API may revise history later; a new download must match the saved snapshot hash before it is treated as the same dataset.

For individual cost runs, `user_data/scripts/h1_backtest.py` accepts `--sensitivity`; the policy cost components, `params.cost_rate` and engine `fee` must agree. The development helper does not expose selection or evaluation arguments.

See [strategy interpretations and G0 ambiguities](../h1-crypto-trend.md#interpretations-of-g0-for-the-pr-description) and [the cited venue report](../h1-venue-check.md). Per Hermes's steering, the Bybit check is **not runnable on registered pairs**; this task did not substitute an OKX-data Bybit-fee experiment despite the venue report's earlier suggestion.
