# First Duck Trade

A crypto strategy research harness built on [Freqtrade](https://github.com/freqtrade/freqtrade). The legacy LLM factory remains in the repository, but generation, critic calls and hyperopt rescue are **off by default**. H1 uses a hand-written daily spot strategy, not generated rules.

## Research-first quick start

Do not start the full Compose stack, the orchestrator or trading services for research setup. No exchange keys, demo orders or live orders are required. Managed REST/container starts and forced exits are hard-disabled during this phase; manually launching a trading service bypasses those Python guards.

1. Create an isolated Python 3.12 environment and install `docker/requirements-orchestrator.txt` there.
2. Run the tests without starting a scheduler:

   ```bash
   .test-venv/bin/python -m pytest tests/ -q
   ```

3. Copy `user_data/configs/research-h1.example.json` to an ignored local policy file. Set `RESEARCH_CONFIG` to that file. Its dates are inclusive UTC days:

   | Window | Start | End |
   |---|---|---|
   | Warm-up | 2018-01-11 | 2019-01-13 |
   | Development | 2019-01-14 | 2021-12-31 |
   | Selection | 2022-01-01 | 2024-06-30 |
   | Untouched evaluation | 2024-07-01 | 2026-12-18 |

4. Freeze confirmed OKX `1Dutc` bars for BTC, ETH, SOL and XRP against USDT. Pin the returned composite hash in the policy. Empty example hashes deliberately prevent runs.
5. Use the explicit development/selection CLI; preregister every configuration and baseline/sensitivity variant before recording the selection row. S2 and evaluation remain unavailable until the evaluation cutoff has closed plus ten minutes.

See [the research harness contract and commands](docs/research-harness.md) for snapshot creation, the one-use evaluation lock, split artefacts, exact hash formats and the shared ledger.

**Accounting boundary:** fixed-capital marked equity, explicit fee/spread/slippage stress and split `net_return`, `net_sharpe` and `max_drawdown` now feed the ledger. Whole-path engine metrics remain labelled separately; synthetic terminal exits do not create fills or fees. The paper gate enforces all-instance equity limits and fails closed on missing/stale inputs. See `docs/cost-risk-accounting.md` for input contracts and limitations. Do not use legacy promotion gates as H1 G1 evidence.

**H1 paper service:** the isolated start/stop/status commands, weekly forward ledger writer and offline readiness evidence are in [H1 paper readiness](docs/h1-paper-readiness.md). Launch still requires G0 approval, selection and the 19 December 2026 start date. Never start the legacy Compose stack for H1.

## Legacy architecture and factory (paused)

The sections below describe retained factory components, not the active H1 workflow or verified profitability. Their historical short-window gates are not an untouched outer evaluation.

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
  Spot     (Binance daily)    ┌─ backtest ── selection split     OKX spot trading
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

Disabled unless `ENABLE_HYPEROPT_RESCUE=true`. The retained promotion bypass is not repaired here; do not enable it for H1. The runner now restricts search to development data and records each requested epoch batch, including failures.

Marginal failures (`FAIL_TOO_FEW` or `FAIL_UNPROFITABLE` with > 0 trades) go through Freqtrade's hyperopt on Sunday 04:00 UTC. The LLM declares parameter ranges via `IntParameter`/`DecimalParameter`; hyperopt searches them. Rescued strategies (post-hyperopt re-backtest meets promote criteria) flip from retired → active.

### 5. Critic pass

Disabled unless `ENABLE_STRATEGY_CRITIC=true`; the direct entry point returns `SKIP` without calling an LLM.

After generation, a second LLM call reviews the rendered code for over-constrained logic (6+ AND conditions), missing NaN guards, threshold sanity errors, regime/logic mismatch, and uncaught look-ahead. Returns PASS / WARN / REJECT. REJECT triggers a regeneration with the critic's feedback in the retry prompt. WARN proceeds. Errors are non-blocking (synthetic PASS) — the critic adds signal, it doesn't gate.

### 6. Iterative refinement

Historically, this loop used a moving 90-day mini-backtest. It now uses only the configured development window. Generation requires `ENABLE_LLM_GENERATION=true`; evaluation is never a generation input. Engine aggregates are not substituted for the pending split-level net metrics, so the old acceptance path must not be used as an H1 gate.

### 7. Pipeline gates (defensive layer)

Between the full backtest and the promote decision, four gates evaluate the result:

- **Regime-conditional floor**: lowers the min-trades threshold by the fraction of the lookback window that was in the strategy's target regime. A breakout strategy in a 0.3% breakout window shouldn't need 20 trades.
- **Beat-buy-and-hold**: strategy must clear 70% of BTC HODL profit OR be materially safer (5+pp lower drawdown). When BH is negative, the floor caps at 0.
- **Walk-forward** (opt-in, env-gated): N consecutive sub-windows; requires majority of windows positive AND sharpe std below threshold. Catches "one lucky month carried the full backtest".
- **Correlation**: rejects candidates with Pearson > 0.7 on daily returns against any already-active strategy. No point running two strategies that lose at the same time.

Each gate returns a verdict. Retirements are tagged with the first failing gate's code (`FAIL_BH`, `FAIL_REGIME`, `FAIL_CORRELATION`, etc.) so failure memory captures specific reasons.

### FreqAI (ML) candidates

A second candidate type (issue #47): instead of rule-based entry/exit conditions, a
LightGBM regressor predicts the forward return over a fixed horizon from whitelisted
features (technicals + the same macro/positioning columns above), and trades on
thresholded predictions. Same declarative-spec safety model (features, model params,
and thresholds are validated against whitelists and bounds — rendered candidates
contain zero executable code), same registry lifecycle, same gates — plus a stricter
rule: walk-forward is mandatory and a skipped walk-forward blocks promotion
regardless of `STRICT_PROMOTION_GATES`. FreqAI candidates are excluded from hyperopt
rescue and (for now) from live deployment.

The LLM proposes ML experiments too (`freqai_generator.py`): batch-first
propose/evaluate/report CLI whose prompt is built from the validator's own
whitelists and bounds, with ML-specific failure memory and in-batch diversity
pressure. Weekly automation is env-gated off by default (`FREQAI_WEEKLY_COUNT=0`).
See `docs/freqai-candidates.md`.

### Reflector + generator feedback loops

- **Reflector** (weekly, Sunday 03:00 UTC): an LLM reads the week's instance performance, regime state, registry stats, and **the per-strategy attribution data**, then writes a markdown reflection naming cross-strategy patterns (e.g. "fgi_fear appeared in top-positive for 4/6 recent ranging strategies").
- **Generator** (weekly, Sunday 02:00 UTC): reads recent reflections AND aggregates attribution patterns per regime (with pool-wide fallback when a regime has < 3 attributed strategies). The prompt gets a `HISTORICAL ATTRIBUTION PATTERNS` block listing the consistent winners and losers.

This is what closes the loop. Earlier iterations had reflection text but the generator never saw the underlying evidence — now both layers see the same data, at different granularities.

## What runs when

Nothing in this table should be started for the research setup. Generation, reflector and LLM-regime jobs are not registered without `ENABLE_LLM_GENERATION=true`; hyperopt is not registered without `ENABLE_HYPEROPT_RESCUE=true`. Direct job calls also respect those flags. Remaining legacy jobs are not disabled by this pause.

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

The retained walk-forward gate is a selection-data stability check, not untouched evaluation. Requests outside the configured split are rejected and logged. Compose's strict legacy gates are not H1 G1, and the new evaluation API never promotes or writes results into the feedback registry.

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
└── tests/                          # pytest, including research-integrity fixtures
```

## Testing

```bash
# Run the full suite in the isolated environment; no scheduler or keys
.test-venv/bin/python -m pytest tests/ -q

# Specific area
.test-venv/bin/python -m pytest tests/test_research.py tests/test_research_snapshots.py tests/test_research_pause.py -v
```

The suite covers the retained factory plus pause flags, canonical snapshots, continuous-path slicing, selection/evaluation locks, trial accounting and CLI routing. Research tests use explicitly synthetic prices and mock only the engine process boundary; they are not a market backtest or engine-integration certification. Exact counts are recorded on the PR.

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

- Research-only setup: generation, critic and rescue are paused by default.
- Frozen S1/S2 inputs and split policies replace unrestricted historical backtests.
- H1 strategy, net cost/risk accounting and G1 scoring are separate work; no edge is claimed.
- No schedulers, trading services, credentials or orders are required to run the tests.

## What's intentionally NOT done yet

- **Adaptive `macro_min_confidence`** — letting the spec renderer auto-tune the macro confidence threshold based on per-bucket attribution lift. Designed but deferred until enough live attribution data accumulates to validate the tuning rule.
- **Pool diversification dashboard** — pairwise correlation heatmap, rolling top-lift buckets, week-over-week alpha decay tracking. The data is in the registry; the monitor doesn't surface it yet.
- **Real BTC dominance** — currently using ETH/BTC ratio as a free proxy (~-0.85 correlation with mcap-based BTC.D). Upgrade to real BTC.D requires a paid CoinGecko tier.

## License

First Duck Trade is licensed under the **GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`).

That means you can use it for personal or commercial purposes, but if you distribute modified versions or run modified versions as a network service, you must make the corresponding source code available under the same license. Copyright notices and attribution must be preserved.

If you use this project in your own work, please cite or credit **xyd945 / First Duck Trade**. See [CITATION.cff](CITATION.cff).
