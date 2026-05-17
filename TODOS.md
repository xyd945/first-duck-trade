# TODOs

## Phase 1 — DONE
- [x] Fix look-ahead bias in LiquiditySweepStrategy.py
- [x] Fix look-ahead bias in fear_and_greed.py
- [x] Standardize indicator API
- [x] Vectorize fetch_extra_data.py
- [x] Build regime detector
- [x] Write MomentumTrendStrategy
- [x] Write pytest tests (46 passing)
- [x] Multi-instance Docker setup
- [x] Orchestrator with APScheduler

## Phase 2 — DONE
- [x] BaseGeneratedStrategy template
- [x] Validation pipeline (security + look-ahead + structure)
- [x] Strategy generator (Claude API)
- [x] Backtest runner (2-stage, sandboxed)
- [x] Strategy registry (SQLite)
- [x] End-to-end verified: first profitable LLM-generated strategy

## Phase 3 — DONE
- [x] Factory loop wired into orchestrator (weekly: generate → backtest → register → promote)
- [x] LLM regime classifier (daily Claude call, overrides indicator regime when confident)
- [x] Reflector agent (weekly trade review, saves insights to markdown)
- [x] Ban VWAP in generator prompt (crashes Freqtrade backtests)
- [x] Fix fear_and_greed ValueError on duplicate index labels

## Phase 4 — DONE (shipped; edge not found)
- [x] Paper trading for 2+ weeks on dry-run
  - **Result (Apr 2026):** small profit, few trades, every LLM-generated strategy failed backtest.
    System is fully automated end-to-end but not effective enough. Loop is *open* — the
    generator has no memory of prior failures and never reads reflector output.
- [ ] Tune risk parameters based on paper trading results *(defer — fix signal quality first)*
- [ ] Set up OKX sub-accounts for live trading *(defer — don't go live without an edge)*
- [ ] Monitoring alerts (Telegram: strategy degradation, kill switch, LLM failures)
- [ ] Walk-forward validation in backtest runner *(moved to Phase 5, Round 7)*

## Phase 5 — Strategy Quality (current)

Motivation: paper trading proved the plumbing works but not the alpha. Each round below is
one branch / PR / feedback cycle. Ship → measure → next round.

### Round 1 — Failure memory + reflector → generator loop  *(in progress)*
- [ ] Add `failure_reason` + `failure_verdict` columns to `strategies` table (migration)
- [ ] Persist reason + verdict in `retire_strategy()` (today it's only logged)
- [ ] `registry.get_recent_failures(k, regime)` — returns failed candidates w/ reason + code excerpt
- [ ] `registry.load_recent_reflections(n)` — reads last N reflector markdowns
- [ ] Extend generator prompt with two new sections:
  - "LESSONS FROM RECENT REFLECTIONS"
  - "RECENT FAILURES TO AVOID" (negative examples: thesis + failure reason)
- [ ] Wire into `job_generate_strategies()` + `job_backtest_candidates()`
- [ ] Dry-prompt flag on generator CLI for inspection without burning tokens
- [ ] Unit tests + one end-to-end dry run

### Round 2 — External data signals
- [ ] Funding rates (OKX/Binance perps) as an indicator
- [ ] Open interest as an indicator
- [ ] BTC dominance + ETH/BTC cross-asset features
- [ ] Per-trade attribution log (which rule fired on entry)

### Round 3 — Structured strategy spec → codegen
- [ ] LLM emits JSON spec (entry rules, exit rules, filters, param ranges)
- [ ] Python template renders Freqtrade class from spec
- [ ] LLM stops picking magic numbers; focuses on ideas

### Round 4 — Hyperopt layer
- [ ] Run Freqtrade hyperopt on passing candidates (LLM shape, optimizer params)
- [ ] Gate: must beat un-optimized version significantly

### Round 5 — Critic pass
- [ ] Second LLM call reviews generated code for look-ahead, over-fit smells, regime assumptions
- [ ] Reject before backtest to save time/tokens

### Round 6 — Multi-turn backtest-in-the-loop
- [ ] Generate → quick held-out backtest → feed metrics + traces back to LLM → refine
- [ ] Cap turns to bound cost

### Round 7 — Pipeline gates & evaluation
- [ ] Walk-forward validation (non-overlapping windows)
- [ ] Statistical gates: min 50 trades, Sharpe > 1.0, profit factor > 1.3
- [ ] **Must beat buy-and-hold** on the same period
- [ ] Correlation filter: reject candidates correlated > 0.7 with deployed set
- [ ] Regime-conditional scoring (evaluate ranging strategies only on ranging windows)

## Ongoing / cross-cutting
- [ ] Telegram alerts (degradation, kill switch, LLM failures) — any round
- [ ] OKX sub-accounts (only after a round actually produces edge)
