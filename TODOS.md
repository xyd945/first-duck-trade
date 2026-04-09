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

## Phase 4 — TODO
- [ ] Paper trading for 2+ weeks on dry-run
- [ ] Tune risk parameters based on paper trading results
- [ ] Set up OKX sub-accounts for live trading
- [ ] Monitoring alerts (Telegram: strategy degradation, kill switch, LLM failures)
- [ ] Walk-forward validation in backtest runner
