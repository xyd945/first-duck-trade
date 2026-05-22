# First Duck Trade

A self-improving algorithmic crypto trading system: an LLM (DeepSeek by default, Claude as fallback) automatically generates trading strategies, runs them through a multi-layer defensive filter, deploys the survivors to paper-trade on OKX, and feeds the results back into the next generation cycle. Built on [Freqtrade](https://github.com/freqtrade/freqtrade).

The point is **not** to ask an LLM to write one strategy and run it. The point is to build a pipeline that produces, evaluates, and learns from many strategies on a weekly cadence, so that what gets deployed is what survived several rounds of evidence-based filtering — not what a single LLM call happened to produce.

## Quick start

You need three things: Docker, an OKX account (demo trading is fine), and an LLM API key (DeepSeek by default, or Anthropic).

```bash
# 1. Clone
git clone https://github.com/xyd945/first-duck-trade.git
cd first-duck-trade

# 2. Configure
cp .env.example .env
# Edit .env — fill in:
#   DEEPSEEK_API_KEY   (https://platform.deepseek.com)
#   ANTHROPIC_API_KEY  (optional; serves as auto-fallback if DeepSeek errors)
#   TELEGRAM_TOKEN / TELEGRAM_CHAT_ID  (optional; for alerts)
mkdir -p user_data/configs
# Create user_data/configs/config-sweep.json and config-momentum.json
# (Freqtrade configs with your OKX demo API keys; see "Per-instance configs" below)

# 3. Download historical data (one-time, ~3-5 min)
docker compose run --rm freqtrade-sweep download-data \
  --config /freqtrade/user_data/configs/config-sweep.json \
  --pairs BTC/USDT ETH/USDT SOL/USDT XRP/USDT \
  --timeframe 1h 4h 1d \
  --days 365

# 4. Bring everything up
docker compose up -d

# 5. Verify (all should be "Up")
docker compose ps
docker compose logs --tail 30 orchestrator | grep "Jobs registered"
```

After this, the orchestrator runs autonomously. Nothing for you to do until the first weekly cycle fires on Sunday 02:00 UTC.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                ORCHESTRATOR                   │
                    │        APScheduler — runs the loop            │
                    └──────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
  DATA INGESTION              STRATEGY FACTORY                  LIVE TRADING
                                                                      │
  Macro     (Yahoo daily)     ┌─ generate ── DeepSeek/Claude      ┌──┴──┐
   ↳ VIX, GOLD, DXY, SPX      ├─ critic   ── second LLM pass      │     │
                              ├─ iterate  ── mini-backtest loop   ▼     ▼
  Perp     (Binance fut)      └─ register ── candidate            ft-sweep  ft-momentum
   ↳ funding rate                                                 │     │
   ↳ open interest                  │                             │     │
                                    ▼                             ▼     ▼
  Spot     (Binance daily)    ┌─ backtest ── full 6-month        OKX spot trading
   ↳ ETH, BTC                 ├─ attribute ── per-trade macro     (demo by default)
   ↳ → ETH/BTC ratio          ├─ gate     ── regime/buyhold/      
                              │              walk-forward/        
                                              correlation         
                              └─ promote ── if all gates pass     
                                                │                  
                                                ▼                  
                                       ┌─────────────────┐         
                                       │ REGISTRY (SQL)  │         
                                       │ candidate →     │         
                                       │ active →        │         
                                       │ retired         │         
                                       └────────┬────────┘         
                                                │                   
                              ┌─────────────────┴─────────────────┐
                              │  FEEDBACK INTO NEXT GENERATION    │
                              │  (closes the loop)                │
                              │                                   │
                              │  - failure memory (don't repeat)  │
                              │  - reflection (weekly LLM review) │
                              │  - attribution patterns           │
                              │  - hyperopt rescue                │
                              └───────────────────────────────────┘
```

## The strategy factory loop

The core insight: **a single LLM-generated strategy almost never works**. What works is a pipeline where the LLM is one stage in a multi-stage filter, and where each cycle's failures become the next cycle's training signal.

There are nine moving parts. Each was added because the previous version of the pipeline had a specific, observable failure mode.

### 1. Failure memory

After a strategy fails (validation, backtest, or post-deploy), its name, thesis, entry logic, and failure verdict are written to the SQLite registry. The next generation cycle injects up to 8 recent failures per regime into the prompt as a "do NOT repeat these" block. Without this, the LLM generates the same losing strategy ideas every week.

### 2a. Macro context (FGI, VIX, Gold, DXY, SPX)

Daily Yahoo Finance fetcher writes external-asset closes to disk. The `add_external_data()` helper injects them into every strategy's dataframe as columns, with a 1-day shift to prevent look-ahead bias. Generated strategies can gate entries on `dataframe['vix'] < 25` or compose a Fear & Greed composite (`dataframe['fgi']`).

### 2b. Crypto positioning (funding rate, open interest)

Binance Futures public API gives 333 days of 8h-funding-rate history and 500 days of daily open-interest data in one call each. Generated strategies see `btc_funding_rate`, `btc_oi`, `btc_oi_pct_change_24h`. Funding > 0.0005 means market is long-loaded (exhaustion risk); negative OI change is forced de-leveraging.

### 2c. Alt-strength regime (ETH/BTC proxy)

Binance spot daily ETH/USDT ÷ BTC/USDT, plus a 30-day rolling z-score. Real BTC dominance from market cap requires a paid CoinGecko tier; this proxy correlates with BTC.D at ~-0.85 because ETH is the dominant alt. The LLM sees `eth_btc_ratio`, `eth_btc_change_7d`, `alt_strength_zscore_30d`.

### 2d. Per-trade macro attribution

After each backtest, every closed trade is bucketed by the macro context that was in effect at entry time (fgi, vix, funding, OI change, alt-strength). Bucket win-rates are aggregated and stored as JSON next to the backtest record. This is the raw signal the reflector and the next generation cycle consume.

### 3. JSON spec → codegen

Free-form Python from an LLM is brittle (forgotten imports, wrong column names, invalid threshold semantics). Instead, the LLM emits a structured JSON spec (`{indicators, params, entry: {core, macro_confidence, macro_min_confidence}, exit, risk}`) and a renderer writes Python that's correct by construction. Entry conditions split into `core` (all must be true) and `macro_confidence` (fraction must clear threshold) — solves the "5 ANDs of macro filters means entries never fire" problem.

### 4. Hyperopt rescue

Marginal failures (`FAIL_TOO_FEW` or `FAIL_UNPROFITABLE` with > 0 trades) go through Freqtrade's hyperopt on Sunday 04:00 UTC. The LLM declares parameter ranges via `IntParameter`/`DecimalParameter`; hyperopt searches them. Rescued strategies (post-hyperopt re-backtest meets promote criteria) flip from retired → active.

### 5. Critic pass

After generation, a second LLM call reviews the rendered code for over-constrained logic (6+ AND conditions), missing NaN guards, threshold sanity errors, regime/logic mismatch, and uncaught look-ahead. Returns PASS / WARN / REJECT. REJECT triggers a regeneration with the critic's feedback in the retry prompt. WARN proceeds. Errors are non-blocking (synthetic PASS) — the critic adds signal, it doesn't gate.

### 6. Iterative refinement

After generation + critic, a 90-day mini-backtest runs. If it fails acceptance (< 5 trades OR profit ≤ 0 AND sharpe ≤ 0), the diagnostic ("ZERO TRADES — loosen filters or lower macro_min_confidence" / "UNPROFITABLE — reconsider thesis") is fed back into a second generation turn. Up to 2 turns total. Best-scoring attempt is kept if neither passes. Live trial showed a strategy rescued from 0 trades to 25 trades, Sharpe 1.72.

### 7. Pipeline gates (defensive layer)

Between the full backtest and the promote decision, four gates evaluate the result:

- **Regime-conditional floor**: lowers the min-trades threshold by the fraction of the lookback window that was in the strategy's target regime. A breakout strategy in a 0.3% breakout window shouldn't need 20 trades.
- **Beat-buy-and-hold**: strategy must clear 70% of BTC HODL profit OR be materially safer (5+pp lower drawdown). When BH is negative, the floor caps at 0.
- **Walk-forward** (opt-in, env-gated): N consecutive sub-windows; requires majority of windows positive AND sharpe std below threshold. Catches "one lucky month carried the full backtest".
- **Correlation**: rejects candidates with Pearson > 0.7 on daily returns against any already-active strategy. No point running two strategies that lose at the same time.

Each gate returns a verdict. Retirements are tagged with the first failing gate's code (`FAIL_BH`, `FAIL_REGIME`, `FAIL_CORRELATION`, etc.) so failure memory captures specific reasons.

### Reflector + generator feedback loops

- **Reflector** (weekly, Sunday 03:00 UTC): an LLM reads the week's instance performance, regime state, registry stats, and **the per-strategy attribution data**, then writes a markdown reflection naming cross-strategy patterns (e.g. "fgi_fear appeared in top-positive for 4/6 recent ranging strategies").
- **Generator** (weekly, Sunday 02:00 UTC): reads recent reflections AND aggregates attribution patterns per regime (with pool-wide fallback when a regime has < 3 attributed strategies). The prompt gets a `HISTORICAL ATTRIBUTION PATTERNS` block listing the consistent winners and losers.

This is what closes the loop. Earlier iterations had reflection text but the generator never saw the underlying evidence — now both layers see the same data, at different granularities.

## What runs when

| Job | Schedule | What it does |
|---|---|---|
| `fetch_macro` | daily 00:05 UTC | Yahoo macro + Binance perp + ETH/BTC spot |
| `classify_regime` | daily 00:10 UTC | indicator-based regime (ADX + EMA + vol) |
| `llm_regime_override` | daily 00:12 UTC | LLM looks at macro data, may override regime |
| `apply_regime` | daily 00:15 UTC | start/stop freqtrade instances based on regime |
| `generate_strategies` | Sunday 02:00 UTC | 5 strategies × iterative × all the feedback loops |
| `backtest_candidates` | Sunday 02:30 UTC | full backtest + attribution + gates + promote/retire |
| `reflector` | Sunday 03:00 UTC | weekly LLM review with attribution evidence |
| `hyperopt_candidates` | Sunday 04:00 UTC | rescue marginal failures via parameter search |
| `check_risk` | every 5 min | drawdown monitoring + kill switch |
| `health_check` | every 2 min | instance liveness |

## Configuration & tweaking

### LLM provider (`.env`)

```
LLM_PROVIDER=deepseek          # or "anthropic"
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...   # serves as automatic fallback if primary fails
```

The wrapper in `user_data/scripts/llm_client.py` dispatches to either Anthropic SDK or the openai SDK (with custom `base_url`). To add a new OpenAI-compatible provider (OpenRouter, Together, Groq), add one entry to `PROVIDER_DEFAULTS`:

```python
"openrouter": {
    "model": "openai/gpt-4o",
    "api_key_env": "OPENROUTER_API_KEY",
    "base_url": "https://openrouter.ai/api/v1",
    "kind": "openai_compat",
},
```

No new client code needed.

### Trading instances (`user_data/scripts/orchestrator.py`)

The `INSTANCES` dict at the top of the file controls regime routing:

```python
INSTANCES = {
    "sweep":    {"url": "http://ft-sweep:8080",    "strategy": "LiquiditySweepStrategy",  "regimes": ["ranging"]},
    "momentum": {"url": "http://ft-momentum:8080", "strategy": "MomentumTrendStrategy",   "regimes": ["trending", "breakout"]},
}
```

To deploy an LLM-promoted strategy: copy its `.py` from `user_data/strategies/candidates/` to `user_data/strategies/`, add a new instance to `docker-compose.yml`, register it here, restart orchestrator.

### Risk limits (orchestrator)

```python
RISK_LIMITS = {
    "max_drawdown_daily_pct": 3.0,
    "max_drawdown_total_pct": 10.0,     # kill switch
    "crisis_regime_action": "stop_all",
}
```

These are **hardcoded by design** — the LLM cannot change them. If you want different limits, edit them here.

### Generation cadence + count

In `orchestrator.job_generate_strategies`:

```python
generate_batch(count=5, regimes=["trending", "ranging", "breakout", "all", "trending"],
               iterative=True, max_turns=2, ...)
```

To generate more strategies per week, bump `count` and add regimes. To skip the iterative refinement (faster but lower quality), set `iterative=False`. To allow more refinement turns, raise `max_turns` (each turn = 1 generation + 1 mini-backtest).

### Gate thresholds (`user_data/scripts/pipeline_gates.py`)

- `gate_regime_conditional_floor(base_min_trades=20, absolute_min_trades=5)` — adjust the trade-count floor.
- `gate_beat_buyhold(profit_floor_ratio=0.7, drawdown_advantage_pct=5.0)` — tighten or loosen the HODL bar.
- `gate_walk_forward(min_passing_windows=2, max_sharpe_std=1.5)` — change OOS robustness requirements.
- `gate_correlation(threshold=0.7, min_overlap_days=30)` — how correlated is "too correlated".

Walk-forward is opt-in (expensive — N× backtest). Enable per run with `R7_WALK_FORWARD=1` in the environment.

### Per-instance Freqtrade configs

Each freqtrade instance needs its own config with API keys. The repo gitignores `user_data/configs/` to keep secrets out of git. Minimal config shape:

```json
{
  "max_open_trades": 3,
  "stake_currency": "USDT",
  "stake_amount": 100,
  "timeframe": "1h",
  "dry_run": false,
  "exchange": {
    "name": "okx",
    "key": "YOUR_OKX_KEY",
    "secret": "YOUR_OKX_SECRET",
    "password": "YOUR_OKX_PASSPHRASE",
    "ccxt_config": {"hostname": "www.okx.com"},
    "pair_whitelist": ["BTC/USDT"]
  },
  "api_server": {
    "enabled": true, "listen_ip_address": "0.0.0.0", "listen_port": 8080,
    "username": "freqtrader", "password": "CHANGE_ME"
  }
}
```

For OKX demo trading, point `ccxt_config.hostname` at `www.okx.com` and use a demo API key (created from the OKX demo trading section).

## Project structure

```
first-duck-trade/
├── docker-compose.yml              # 4 services: sweep, momentum, orchestrator, monitor
├── docker/
│   ├── Dockerfile.orchestrator     # Python + Docker CLI (for backtest_runner)
│   └── requirements-orchestrator.txt
├── .env.example                    # template — copy to .env
├── user_data/
│   ├── config.json                 # shared backtest config
│   ├── configs/                    # per-instance live configs (gitignored)
│   ├── strategies/
│   │   ├── LiquiditySweepStrategy.py
│   │   ├── MomentumTrendStrategy.py
│   │   ├── base_generated.py       # base class for LLM strategies
│   │   └── candidates/             # LLM-generated strategies (gitignored)
│   ├── indicators/
│   │   ├── regime_detector.py      # ADX + EMA + vol regime classification
│   │   ├── fear_and_greed.py       # composite FGI + external loader
│   │   ├── perp_metrics.py         # funding rate + OI joins
│   │   ├── alt_strength.py         # ETH/BTC ratio + z-score
│   │   ├── external_data.py        # umbrella add_external_data()
│   │   ├── chaikin_money_flow.py
│   │   └── whale_liquidity.py
│   ├── scripts/
│   │   ├── orchestrator.py         # APScheduler — runs all jobs
│   │   ├── llm_client.py           # provider-agnostic chat wrapper (DeepSeek/Claude/...)
│   │   ├── strategy_generator.py   # LLM → JSON spec → Python
│   │   ├── strategy_spec.py        # spec validator + Python renderer
│   │   ├── strategy_critic.py      # second-LLM code review
│   │   ├── validation_pipeline.py  # AST checks (no exec, no I/O, no look-ahead)
│   │   ├── backtest_runner.py     # docker-in-docker backtest + hyperopt
│   │   ├── pipeline_gates.py       # regime/buyhold/walk-forward/correlation gates
│   │   ├── trade_attribution.py    # per-trade macro bucket attribution + aggregation
│   │   ├── strategy_registry.py    # SQLite lifecycle (candidate/active/retired)
│   │   ├── fetch_extra_data.py     # Yahoo macro fetcher
│   │   ├── fetch_perp_data.py      # Binance perp futures fetcher
│   │   ├── fetch_eth_btc.py        # Binance spot ETH/BTC fetcher
│   │   ├── monitor.py              # dashboard HTTP server (port 8888)
│   │   └── notifier.py             # Telegram alerts
│   ├── data/                       # OHLCV, regime state, registry DB (gitignored)
│   └── backtest_results/           # per-call trade exports (gitignored)
└── tests/                          # 251 tests, pytest
```

## Testing

```bash
# Run the full suite (mocked LLM calls — doesn't need API keys)
docker exec ft-orchestrator python -m pytest /app/tests -q

# Specific area
docker exec ft-orchestrator python -m pytest /app/tests/test_pipeline_gates.py -v
```

251 tests cover: validation pipeline, regime detection, indicator math, LLM client (both providers + fallback), strategy spec + critic + iterative generator, pipeline gates (all four), trade attribution (math + reflector/generator consumption), registry lifecycle, hyperopt rescue.

## Tech stack

| Component | Tech |
|---|---|
| Trading engine | [Freqtrade](https://github.com/freqtrade/freqtrade) (Docker) |
| Default LLM | DeepSeek V4 Pro (OpenAI-compatible API) |
| Fallback LLM | Anthropic Claude (auto-retry on primary failure) |
| Scheduling | APScheduler |
| Orchestration | Docker Compose (4 services) |
| Registry | SQLite |
| TA | pandas_ta, numpy, pandas |
| Macro data | yfinance (VIX/GOLD/DXY/SPX), Binance public API (perp + spot) |
| Testing | pytest |

## Status

- **Phase 4: Paper trading** on OKX demo (BTC/USDT spot, can be expanded)
- All 7 rounds of the factory loop are live (R1 failure memory through R7 gates)
- Reflector and generator both consume attribution data
- DeepSeek V4 Pro is the default LLM with automatic Anthropic fallback
- Cron runs weekly on Sunday — first full DeepSeek-powered cycle fires automatically

## What's intentionally NOT done yet

- **Adaptive `macro_min_confidence`** — letting the spec renderer auto-tune the macro confidence threshold based on per-bucket attribution lift. Designed but deferred until enough live attribution data accumulates to validate the tuning rule.
- **Pool diversification dashboard** — pairwise correlation heatmap, rolling top-lift buckets, week-over-week alpha decay tracking. The data is in the registry; the monitor doesn't surface it yet.
- **Real BTC dominance** — currently using ETH/BTC ratio as a free proxy (~-0.85 correlation with mcap-based BTC.D). Upgrade to real BTC.D requires a paid CoinGecko tier.

## License

First Duck Trade is licensed under the **GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`).

That means you can use it for personal or commercial purposes, but if you distribute modified versions or run modified versions as a network service, you must make the corresponding source code available under the same license. Copyright notices and attribution must be preserved.

If you use this project in your own work, please cite or credit **xyd945 / First Duck Trade**. See [CITATION.cff](CITATION.cff).
