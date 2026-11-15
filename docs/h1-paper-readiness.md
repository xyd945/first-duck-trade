# H1 paper readiness — draft, not launch-approved

The dry-run adapter, isolated service and offline comparison are implemented. This draft must not be merged or started before the remaining checks below pass. No live-data service was started. No legacy service or Compose stack is involved.

## Verified locally

- Pinned engine H1 suite: 106 passed, 15 upstream pandas deprecation warnings.
- Orchestrator suite: 715 passed, 9 skipped, 7 dependency/deprecation warnings.
- Real Freqtrade simulated-order tests cover wallet/SQL reconciliation, sell-first funding, common buy scaling, expiry, restart risk vetoes and stale/missing marks.
- Forward writer tests cover idempotent missing-evidence rows and real SQL-filled-order accounting with a cumulative backtest comparison. These are synthetic tests, not forward performance evidence.

## Service commands

Run from repository root using an isolated Python 3.12 environment installed from `docker/requirements-h1-engine.txt`, named `.venv-h1` below. The runtime requires Freqtrade 2026.10 and ccxt 4.5.90.

```sh
.venv-h1/bin/python user_data/scripts/h1_service.py status
.venv-h1/bin/python user_data/scripts/h1_service.py stop
```

After readiness review, selection and the 19 December 2026 00:00 UTC decision, the intended start command is:

```sh
.venv-h1/bin/python user_data/scripts/h1_service.py start
```

State lives in `user_data/data/h1-paper/`: `service.log`, `heartbeat.json`, `service.lock`, `identity.json`, `trades.sqlite` (Freqtrade simulated orders/trades), `journal.sqlite` (risk, batches, reference opens and NAV), and `data/` (captured confirmed candles). Start and direct run both reject dates before 19 December. Start requires a matching `selected` row in `user_data/data/trial_ledger.sqlite`. Explicit `--placeholder` permits M45 testing only, never G2 evidence. No reboot launcher is installed; restart is manual, using the same command and directory. Changed configuration/code identity refuses reuse of state.

The single `research_h1.designated_portfolio_notional` value remains 250000, pending Yudi's approval. The paper wallet remains 10000 USDT.

## Offline development comparison

```sh
.venv-h1/bin/python user_data/scripts/h1_replay.py --data /path/to/h1-dev-bounded --output /new/path/replay
```

Defaults are 2–31 December 2021. Replay uses actual Freqtrade dry-run orders with flat books at archived opens. These are explicit replay assumptions, not historical spread measurements. Missing/blocked/expired decisions stay in the denominator. Raw and risk-adjusted target agreement are reported separately, with a 0.5 percentage-point tolerance and a 95% threshold. This is not measured live slippage or a G2 pass.

## Remaining before ready-for-review

1. Expose `h1_review.write_forward` through a documented weekly CLI and test it end to end. It is currently a tested Python API only. Check placeholder/missed/error verdicts, repeat-week behaviour, cost checks and cumulative dates in the shared ledger.
2. Exercise process start/stop/status in an isolated offline harness, including readiness and failure paths. Do not start public-data paper trading. Unit coverage does not prove deployed operation.
3. Review the complete adapter/service/reporting diff for operational edge cases and fail-closed behaviour. Re-run both suites and the archived replay after fixes.
4. Verify public CI at the final head and remove draft status only when all acceptance criteria pass. Hermes reviews and merges.

Keep raw development data and runtime databases private and untracked. Publish only derived replay evidence.
