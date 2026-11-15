# H1 crypto trend: strategies, execution adapter and tests

The registered specification is G0 H1 (Quant desk, "G0 H1 crypto trend"). This code implements it for historical development runs only. It does not start a bot, a dry-run or a scheduler, and it has no selection or evaluation entry point.

## Files

| Path | Purpose |
|---|---|
| `user_data/strategies/candidates/H1Trend.py` | `H1Trend` (registered rule) and the three baselines `H1SimpleTrend` (20/120 SMA), `H1EqualWeight` and `H1BuyHold` (BTC). One source file, so all four share the G0 §8 `config_hash` source. |
| `user_data/configs/research-h1.example.json` | Policy: splits, snapshots, costs, the `h1` block (grid, eligibility dates, startup, SMA lengths, Monday weekday, paper mapping, engine pin) and the default `params`. |
| `user_data/scripts/h1_backtest.py` | `H1Backtesting`: sell-first historical adapter (G0 §3 item 6), engine/ccxt version assertion and execution-contract checks. `run_development(...)` and the CLI run the development window only and always write a trial-ledger row. |
| `user_data/scripts/research.py` | Existing fix-2/4 orchestrator. H1 additions: rules passed into the engine config, H1 script hashes, dev-only snapshot end, and same-open sell-first ordering when replaying the export. |
| `docker/requirements-h1-engine.txt` | Engine pin for H1 runs and tests. |
| `tests/test_h1*.py` | Signal, prefix-invariance, sizing/risk, real-engine and ledger tests (see below). |

`user_data/strategies/candidates/` is git-ignored by default; `H1Trend.py` is tracked explicitly.

## Engine pin

Every H1 backtest and the H1 dry-run use Freqtrade `2026.10` (commit `4cf0351`) with `ccxt==4.5.90`, the version in that tag's `requirements.txt` (Hermes decision, 2026-11-15). `H1Backtesting` refuses to start on any other version. The legacy compose services keep their `stable` tags. `stable` may move to 2026.11 before 19 December, and a mid-test engine change would break the same-engine assumption in G1 and G2.

## Running the tests

The H1 tests need Freqtrade, so they skip in the orchestrator environment. Use a separate virtual environment for them:

```sh
python3.12 -m venv .engine-venv
.engine-venv/bin/python -m pip install -r docker/requirements-h1-engine.txt
PYTHONPATH=user_data/scripts .engine-venv/bin/python -m pytest tests/test_h1.py tests/test_h1_execution.py tests/test_h1_engine.py tests/test_h1_runner.py -q
```

The engine tests use offline OKX market metadata and synthetic prices; they make no network calls and need no keys.

## Development runs and inactive paper config

The completed development evidence is in [h1-development/README.md](h1-development/README.md): 24 market ledger rows, gate/base results, data hashes and reconciliation receipts. No selection or evaluation window ran.

The keyless config is `user_data/configs/h1_dryrun.json`. It passes `show-config` and the strict dry-run schema but remains **not start-ready**: M45 is a placeholder and paper activation is deliberately refused. G0 approval, the actual selection, a paper execution adapter and reconciled risk inputs are still required. No service was added or started. Bybit is not runnable on the registered pairs; see [the venue report](h1-venue-check.md).

### Individual development run

```sh
PYTHONPATH=user_data/scripts .engine-venv/bin/python user_data/scripts/h1_backtest.py \
  --policy <policy.json> --engine-config <engine-config.json> --strategy H1Trend --config-id M45
```

Baselines take `--strategy H1SimpleTrend|H1EqualWeight|H1BuyHold` and no `--config-id`. Cost-only reruns use `--sensitivity`, with matching policy and engine costs, and record a sensitivity verdict without a new configuration hash. Each finished row records engine/ccxt versions in `notes.params`, including failures. Data comes from exchange `okx`, keyless, USDT pairs, frozen through the fix-2 snapshot. Each attempt, including crashes, writes one `trial_ledger` row with `split = dev`; dev rows are not counted in N. Batch records, including target-versus-filled divergence, are written to `decision-batches.json` in the run directory.

## Interpretations of G0 (for the PR description)

1. **Eligibility and `N_t`.** A coin needs both its registered eligibility date and 366 confirmed bars in the loaded data at the decision. `N_t` is the count of coins eligible that day, computed only from bars available then.
2. **Lookbacks.** `L` counts daily bars. A missing or unconfirmed reference close is replaced by the latest confirmed close at most 3 days earlier. Otherwise the previous `s` is kept and the row is flagged as an anomaly in the batch record.
3. **Volatility.** `σ̂` is the sample standard deviation (`ddof = 1`) of the last 60 daily log returns of confirmed closes, times √365. A coin with no finite `σ̂` gets no target that day.
4. **Triggers.** A change in `s` from the previous decision trades that coin to its new `w` without the 2.5% band, as G0 states the band only for the Monday resize. Monday means the 00:00 UTC Monday decision after the Sunday bar. The band compares `|w − current weight|` with 2.5% of pre-trade marked equity.
5. **Self-financing execution (G0 §3 item 6).** Triggered sells go first at the same open. Sale proceeds pay their fees before funding buys, and all triggered buys are scaled by one common factor. Fee-funded shortfalls remain unfilled and appear in the batch record's `unfilled` map. Orders below the exchange minimum get a zero stake and stay visible there too.
6. **Final day.** An entry at the last registered open is executable. Only Freqtrade's terminal liquidation after the path end is synthetic, matching the fix-4 accounting contract.
7. **Startup.** Historical runs load 365 startup candles plus the preceding signal bar (G0 §9 engine note), so 2019-01-14 is the first executable day.
8. **Cost.** Fee and slippage are charged together as Freqtrade's `fee` (0.30% gate, 0.15% base), taken from the policy `costs` block through `accounting.cost_rate`. There is no separate fill-price slippage model.
9. **BTC buy-and-hold.** One entry at the 2019-01-14 path start on the continuous path, subject to BTC eligibility and a confirmed bar. Window results are slices of this path, so a selection or evaluation slice does not charge a new entry. G0 §5 says "from the first day of each window, charged one entry", while §6 forbids restarting positions at window boundaries; the difference is one 0.30% entry per window.
10. **Simple trend baseline.** The 20- and 120-day SMAs use confirmed closes with the same 3-day reference fill. A missing latest bar means no trade that day. It holds a full sleeve `1/N_t` with no volatility scaling, trades daily on a signal change and resizes on Monday with the 2.5% band.
11. **Equal-weight baseline.** `1/N_t` in every eligible coin, resized only on Monday with the 2.5% band. A change in `N_t` takes effect at the next Monday.
12. **Ledger.** `H1Trend` rows have verdict `ok`; an identical repeat becomes `rerun`. Baseline rows have verdict `baseline`, including repeats. Baselines share `H1Trend.py` and the default params, so their `config_hash` equals the matching H1 configuration's; `verdict` and `params_json.config.strategy` tell them apart. Neither affects N.
13. **Development snapshot.** A dev run accepts a snapshot ending at the development end (2021-12-31) as well as at the S1 end, so building does not need selection-window bars on disk.
14. **Paper limits.** Fix-4 `paper_risk.evaluate(...)` gates buys in paper modes only. The 250,000 `designated_portfolio_notional` is recorded in the policy as "pending Yudi's OK" but not read by code: fix 4 requires the risk state to be initialised with explicit reconciled values, with no default capital. `bot_start` refuses every run mode except a backtest under the logged historical adapter, so paper activation is disabled. In paper mode a batch waits while any sell is pending and expires at the next daily decision. Historical research runs are not capped by the paper-phase limits.
15. **Bybit cost check.** Not runnable on registered pairs: Bybit EU lists USDC and EUR pairs only, with history from 2025-06-11. No substitute market is used.
16. **Export replay.** Freqtrade's minified export groups orders by trade, which loses same-open ordering. When replaying an H1 export, `research.slice_result` orders fills on the same date sells first, matching the adapter.
