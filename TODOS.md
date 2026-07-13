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

## Phase 5 — Strategy Quality — DONE

All 7 rounds shipped plus the two follow-on consumption layers + correlation gate + LLM
abstraction. Test suite grew from 46 → 251.

- [x] **Round 1** — Failure memory + reflector → generator loop (PR #11)
- [x] **Round 2a** — Macro context: FGI / VIX / Gold / DXY / SPX (PR #12)
- [x] **Round 2b** — BTC perp signals: funding + OI (PR #13)
- [x] **Round 2c** — Alt-strength proxy via ETH/BTC ratio (PR #19)
- [x] **Round 2d** — Per-trade macro-bucket attribution (PR #20)
- [x] **Round 3** — JSON spec → codegen (PR #16)
- [x] **Round 4** — Hyperopt rescue layer, v1 primitive + v2 orchestrator wiring (PRs #13, #14)
- [x] **Round 5** — Critic pass (PR #15)
- [x] **Round 6** — Multi-turn iterative refinement (PR #17)
- [x] **Round 7** — Pipeline gates: regime / buyhold / walk-forward (PR #18)
- [x] **R7.4** — Correlation gate (PR #23, deferred from R7 until R2d enabled trade exports)
- [x] **Reflector consumption** — reflector reads attribution patterns (PR #21)
- [x] **Generator consumption** — generator reads cross-strategy attribution aggregates (PR #22)
- [x] **LLM provider abstraction** — DeepSeek default, Anthropic auto-fallback (PR #24)

## Phase 6 — Archetype Diversity *(next)*

Motivation: LLM keeps converging on ~6 textbook archetypes (Donchian breakouts, Keltner
channels, EMA crosses, BB-RSI bounces, volume breakouts, MACD divergence). The factory is
producing diversity-in-name-only. Move from suggestion-based steering (prompt asks) to
structural enforcement (spec validator requires).

### Active: 10 archetypes × coherent regime cells (22 cells total)

- [ ] Add `archetype` field to spec validator with strict enum:
  `momentum_continuation`, `mean_reversion`, `breakout_volume`, `vol_squeeze`,
  `vol_compression_mean_reversion`, `funding_contrarian`, `oi_cascade_followthrough`,
  `alt_strength_divergence`, `macro_led_risk_on`, `liquidity_sweep_followthrough`
- [ ] Per-archetype prompt blurbs (thesis + typical indicators + threshold conventions)
- [ ] `generate_batch` iterates the coherence matrix (archetype × regime) instead of
  cycling regimes — one strategy per coherent cell, ~22/cycle
- [ ] Failure memory + attribution tagged by archetype
- [ ] Bump `MAX_CANDIDATES` 30 → 60
- [ ] Reschedule weekly cycle: Saturday 20:00 UTC (10-hour window) instead of Sunday 02:00
- [ ] After 3-4 weeks of data: retire dead archetypes, split successful ones into sub-variants

### Deferred — need new infrastructure first

- [ ] **`multi_tf_confirmation` archetype** — 1h entry only when 4h/1d trend agrees.
  Requires spec renderer support for Freqtrade's `informative_pairs()` +
  `merge_informative_pair()` mechanism (~half day of work in `strategy_spec.py`).
  Massively reduces false signals.
- [ ] **`session_pattern` archetype** — time-of-day / day-of-week entries (Asian/EU/US session
  effects, weekend liquidity gaps). Requires adding `hour_of_day` and `day_of_week`
  columns to the dataframe via `add_external_data` or a new indicator helper, plus
  prompt guidance on typical session windows (~3-4 hours of work).

## FreqAI candidate type (issue #47) — Phases 1+2 DONE

- [x] FreqAI backtest service (`freqtrade-freqai`, stable_freqai image)
- [x] Hand-written baseline (`FreqaiBaselineLGBM`) — trains + backtests in Docker
- [x] Declarative spec → renderer path (`freqai_spec.py`): feature whitelist,
      bounded LightGBM params, declarations-only rendered candidates
- [x] Registry `spec_type` column; freqai excluded from hyperopt rescue + deployment
- [x] Orchestrator lifecycle integration with MANDATORY walk-forward for freqai
- [x] Live shakedown: baseline ran the full lifecycle and was correctly retired
      (`FAIL_UNPROFITABLE`, gates + failure memory + model purge all exercised)
- [ ] **Phase 3 — LLM-proposed freqai specs**: wire generator/critic to emit
      `freqai_spec` JSONs (the safe renderer path now exists); start with 1–2
      specs/week alongside rule candidates
- [ ] **Phase 3 — live deployment of ML strategies**: reconciler support for the
      freqai image, model retraining/freshness ops, low-confidence retirement
      (`stale model`, `low confidence` verdicts)

## Future enhancements

- [ ] **Adaptive `macro_min_confidence`** — spec renderer auto-tunes the threshold per
  strategy based on attribution lift magnitudes of the chosen macro_confidence buckets.
  Designed but deferred until enough live attribution data accumulates to validate the
  tuning rule.
- [ ] **Pool diversification dashboard** — pairwise correlation heatmap of active
  strategies, rolling top-lift buckets across the pool, week-over-week alpha decay.
  Monitor service exists but doesn't surface this yet.
- [ ] **Real BTC.D from market cap** — currently ETH/BTC proxy (~-0.85 correlation with
  mcap-based BTC.D). Upgrade requires CoinGecko Pro tier (~$129/mo) or building our own
  daily snapshot accumulator.
- [ ] **Auto-deploy promoted strategies to live containers** — currently promotion is to a
  status in the registry, not to a running freqtrade instance. Auto-deployment needs
  templated `docker-compose` service generation + per-instance config + safety review gates.

## Ongoing / cross-cutting
- [ ] Telegram alerts (degradation, kill switch, LLM failures) — any round
- [ ] OKX sub-accounts (only after a round actually produces edge)
