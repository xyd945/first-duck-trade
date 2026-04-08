# TODOs

## Phase 1
- [x] Fix look-ahead bias in LiquiditySweepStrategy.py: change `shift(1)` to `shift(pivot_len + 1)` for pivot values
- [x] Fix look-ahead bias in fear_and_greed.py: shift VIX and GOLD data by 1 day before merging
- [x] Standardize indicator API: make fear_and_greed return DataFrame instead of Series
- [x] Vectorize fetch_extra_data.py: replace iterrows() with vectorized pandas ops
- [x] Build regime detector (indicator-based, threshold classification)
- [x] Write MomentumTrendStrategy (2nd strategy for trending/breakout regimes)
- [x] Write pytest tests for indicators and regime detector (24 tests, all passing)
- [ ] Set up OKX sub-accounts for multi-instance Docker setup (separate API keys per strategy instance)
- [ ] Add .venv/ to .gitignore
- [ ] Set up multi-instance docker-compose.yml (separate containers per strategy + orchestrator)
- [ ] Build risk manager (max drawdown kill switch, position limits)
- [ ] Backtest both strategies with fixed look-ahead bias and validate results
- [ ] Build orchestrator with APScheduler in Docker

## Phase 2
- [ ] Add `rolling(center=True)` detection to validation pipeline (catches look-ahead bias that shift(-N) check misses)
- [ ] Include structured logging from Phase 2 start (LLM calls, backtest results, registry changes)
- [ ] Build LLM regime classifier (daily, reads macro data + news)
- [ ] Build strategy generator with constrained templates and validation pipeline
- [ ] Build automated backtest runner for candidate evaluation
- [ ] Build strategy registry (SQLite) with promotion/retirement logic
