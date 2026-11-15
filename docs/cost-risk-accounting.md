# Cost and paper-risk accounting

This is fix 4, built on merged fix 2. Execution remains disabled. No market result, account balance or approved portfolio allocation is implied by the synthetic tests.

## Backtest accounting

`accounting.equity_curve(initial_capital, fills, marks, cost_rate=...)` replays actual spot quantities against a fixed initial cash balance. There are no deposits, leverage, shorts or funding costs. Daily equity is cash plus all open inventory marked at the frozen UTC candle close. Idle cash and unrealised losses count.

`net-equity.feather` records `date`, `cash`, `gross_exposure`, `costs`, `equity`, `net_return` and `fixed_capital_return`. Daily `net_return` is `equity / previous_equity - 1`; `fixed_capital_return` is the same cash P&L divided by initial capital. Neither averages closed-trade percentages.

The explicit one-side rate is:

```text
(fee_rate + spread_rate / 2 + slippage_rate) * stress_multiplier
```

`spread_rate` is the full bid/ask spread. Costs apply once to each actual buy or sell reference notional. The agreed OKX model supports 0.0008 maker or 0.001 taker, with taker used by the example. Maker fees are an assumption, not a promise that a limit order receives maker treatment. The example's five-basis-point slippage allowance includes spread, so its separate spread field is zero. Do not add a second spread charge for the same allowance. Stress 2 doubles the entire cost. Changing fee/components/stress changes the registration fingerprint, not the statistical configuration hash.

The frozen engine `fee` equals this aggregate rate so sizing and available funds face the cost stress too. The independent replay starts from initial cash and raw fill prices; it does not subtract costs again from an already net wallet/engine profit. This is a notional-cost stress model, not an order-book simulation or evidence about market impact.

For each scored split:

- Opening equity is the preceding daily marked equity, or initial capital at path start. Positions and cash are never reset at a split boundary.
- `net_return` is ending/opening equity minus one, as a fraction.
- `net_sharpe` is mean daily NAV return divided by sample standard deviation, annualised by `sqrt(365)`, with zero risk-free rate. Insufficient or zero-variance samples return NULL.
- `max_drawdown` is the positive peak-to-trough loss fraction, including opening equity in the peak series.
- `cost_total` includes only costs incurred inside the split. `net_path_metrics` separately scores the entire replayed path.
- Ledger `n_trades_or_rebalances` counts actual fills in the split, including resizes. It is not a count of completed round trips or weekly rebalance batches.

`research.slice_result` retains the original archive, `split.json` and `split.feather`. It reconstructs the net curve from the full-path order export and frozen closes. Freqtrade's pre-bar wallet captures are checked for inventory consistency and continuous coverage; their cash is not used as closing NAV because open-entry fees are deferred in that export. Unknown currencies, missing positions, duplicate records, partial/ambiguous order exports, missing held-asset marks, short inventory or insufficient cash fail the trial. An unheld, not-yet-listed pair may have missing early marks.

Only a final `force_exit` order at the validated path end is synthetic. It is excluded from fills and fees; earlier partial sales remain real. The remaining position is marked at the final close, not liquidated at the synthetic export price. Full-path engine aggregates remain labelled `path_metrics` and never populate the net split columns.

The export contract was inspected in Freqtrade tag `2026.10`: `persistence/trade_model.py`, `wallets.py`, `optimize/backtesting.py`, `optimize/optimize_reports/bt_storage.py` and `rpc/api_server/api_trading.py`. Tests run synthetic exports through the real parser, accounting and SQLite ledger. They do not claim a real-engine compatibility run; H1 must check that before using performance results. For H1, `tests/test_h1_runner.py` now runs the real pinned 2026.10 engine and its export through this parser, accounting and ledger on synthetic data (see `docs/h1-crypto-trend.md`).

## Paper account input and limits

`paper_risk.evaluate(snapshot, state, expected_instances, now, proposals=[...])` is a deterministic **batch risk decision**, not an execution permission or an order reservation. It never sends an order. Use one coherent snapshot and all proposed increases together; independent concurrent batches cannot reserve the same cash safely. Reductions are not proposals for new exposure and pending sells never release buying capacity before filling.

The snapshot must contain every expected instance exactly once. Shared account cash appears once, while independent paper wallets have distinct account IDs. Unallocated designated-portfolio cash may be represented by an account with empty instances/positions/pending lists; never invent reserve capital to dilute risk. The draft 10,000/250,000 mapping is not automatically accepted.

Input shape (illustrative only; timestamps below are intentionally stale):

```json
{
  "asof": "2020-01-02T12:00:00+00:00",
  "quote_currency": "USDT",
  "prices": {
    "BTC/USDT": {"price": 100, "asof": "2020-01-02T12:00:00+00:00"}
  },
  "accounts": [{
    "id": "paper-shared",
    "mode": "spot",
    "cash": 9900,
    "instances": ["sweep", "momentum"],
    "positions": [{"id": "one", "instance": "sweep", "pair": "BTC/USDT", "quantity": 1}],
    "pending": []
  }]
}
```

`cash` means total quote cash, including cash reserved for pending buys, after settled fees. `quantity` means total held units, including units reserved for pending sales. Pending records require `id`, `instance`, `pair`, `side` (`buy` or `sell`) and positive `notional`. Proposals require `instance`, `pair` and positive `notional`. Position ownership, account mode, unique IDs, nonnegative finite values and spot USDT pairs are validated. Prices are required for held assets, pending orders and proposals.

The persisted state requires `capital`, `peak_equity`, `day_start_equity`, `last_equity`, UTC `day`, timezone-aware `asof`, and boolean `kill_switch_active`/`daily_halt`. Initialisation and manual reset require explicit reconciled designated-portfolio values. There is no default capital and no automatic reset after an outage.

- A new/increased position, summed across instances and pending buys, must not exceed **1% of marked designated-portfolio equity**.
- All positions plus pending buys and the proposed batch must not exceed **5% gross exposure**. Each account must also have enough cash.
- A **0.5% daily loss** halts new exposure for the rest of the UTC day, even after recovery.
- A **5% drawdown from the persistent equity high-water mark** latches the kill switch. A positive lifetime P&L does not exempt a drawdown.
- UTC rollover uses the last marked equity as daily opening equity and preserves the high-water mark. Thresholds include equality.
- Snapshot/price timestamps must be timezone-aware, not future-dated and at most 300 seconds old. Previous risk state may be up to 600 seconds old to tolerate jitter around the five-minute monitoring job. Longer gaps require reconciliation.
- Missing/stale/invalid inputs latch a kill rather than granting permission. Valid snapshots continue to update equity while a kill remains latched; recovery never clears it automatically.

## Orchestrator enforcement and boundary

`job_check_risk` reads `user_data/data/paper_account.json` and `risk_state.json`. Expected instances include legacy `sweep`/`momentum` and every container discovered by the deployment manager. Dynamic local REST credentials, if configured by an operator, use `FT_DEPLOYED_API_USERNAME` and `FT_DEPLOYED_API_PASSWORD`; no values are supplied by this change. An unavailable Docker inventory or missing file is a denial, never an empty healthy account.

Denied checks request REST `stopentry` on every known instance, preserving position management rather than forcing liquidation. Calls are retried on later checks even when a kill is already latched. One failing request does not skip other instances. `stop_failures` is persisted and latches the kill. State-write errors still attempt all halts and return a denial. Failed REST calls are **not** proof that an already-running external process stopped.

Managed REST starts, forced exits and dynamic container starts remain hard-disabled by `EXECUTION_ENABLED = False`; neither `RECONCILER_ACTING` nor an allowlist overrides it. With execution disabled, checks request entry halts even if the pure risk calculation passes. Legacy Docker unit tests explicitly override the constant only with mocked Docker/REST edges.

The snapshot contract is not a live exchange collector. No snapshot producer, H1 strategy hook, capital approval, order dispatcher or scheduler is started here. The existing H1 work must supply complete reconciled snapshots and use the batch gate in its sizing path before paper activation. Manually starting Freqtrade or editing code bypasses these application guards; this is not a broker-side or operating-system interlock. Do not run the full Compose stack.

## Verification

Run in an isolated environment with the repository's orchestrator requirements:

```bash
.test-venv/bin/python -m pytest tests/test_accounting.py tests/test_research.py tests/test_paper_risk.py tests/test_risk_wiring.py -q
.test-venv/bin/python -m pytest tests/ -q
```

Tests include hand-computed capital weights, maker/taker costs and 2× stress, cross-split holdings, synthetic-close exclusion, wallet reconciliation, shared-account equity, exact daily/drawdown boundaries, pending exposure, malformed/stale inputs, all-instance entry halts and REST/filesystem failures. They use no keys, orders, market-data downloads, Docker engine, running scheduler or G1 evaluation data.
