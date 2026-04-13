# First Duck Trade

An algorithmic crypto trading system with an LLM-powered **Strategy Factory** — Claude automatically generates, validates, backtests, and deploys trading strategies. Built on [Freqtrade](https://github.com/freqtrade/freqtrade).

## Quick Start

Unless you plan to rewrite the core engine of Freqtrade (which is rare), do not fork the repository. Forking creates a maintenance nightmare when Freqtrade releases updates.

- Repo: Create a new, clean repository for your project (in our case First Duck Trade)
- Structure: You will strictly separate your logic from Freqtrade's core using the user_data folder pattern.
- Dependency: Treat Freqtrade as an external dependency (via Docker), not your own code.

```bash
# Create your clean repo
mkdir first-duck-trade
cd first-duck-trade

# Download the official docker file (don't clone the whole repo)
curl https://raw.githubusercontent.com/freqtrade/freqtrade/stable/docker-compose.yml -o docker-compose.yml

# Initialize the user_data folder
docker compose run --rm freqtrade create-userdir --userdir user_data

# Create configuration - Requires answering interactive questions
docker compose run --rm freqtrade new-config --config user_data/config.json
```

### 1. Download Historical Data

```bash
docker compose run --rm freqtrade download-data \
  --config /freqtrade/user_data/config.json \
  --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT \
  --timeframe 5m 15m 1h 4h \
  --days 365
```

### 2. Run Backtest

```bash
docker compose run --rm freqtrade backtesting \
  --config /freqtrade/user_data/config.json \
  --strategy MyFirstStrategy \
  --timeframe 1h
```

### 3. Run Hyperopt (Optimization)

```bash
docker compose run --rm freqtrade hyperopt \
  --config /freqtrade/user_data/config.json \
  --strategy MyFirstStrategy \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy sell \
  --epochs 100
```

### 4. Start Live/Dry Run

```bash
docker compose up -d
```

### 5. View Logs

```bash
docker compose logs -f
```

### 6. Stop Bot

```bash
docker compose down
```

## Architecture

```
                         ┌─────────────────────────────────┐
                         │        ORCHESTRATOR              │
                         │     (APScheduler, Python)        │
                         │                                  │
                         │  Daily:                          │
                         │    00:05  Fetch macro data       │
                         │    00:10  Classify regime        │
                         │    00:12  LLM regime override    │
                         │    00:15  Apply regime routing   │
                         │                                  │
                         │  Weekly (Sunday):                │
                         │    02:00  Generate strategies    │
                         │    02:30  Backtest candidates    │
                         │    03:00  Weekly reflection      │
                         │                                  │
                         │  Continuous:                     │
                         │    Every 2m  Health check        │
                         │    Every 5m  Risk check          │
                         └──────────┬──────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
          ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐
          │  ft-sweep   │ │ ft-momentum │ │  ft-backtest    │
          │  Port 8081  │ │  Port 8082  │ │  (on-demand)    │
          │             │ │             │ │                 │
          │  Ranging    │ │  Trending   │ │  Sandboxed      │
          │  strategy   │ │  strategy   │ │  CPU/mem limits │
          └─────────────┘ └─────────────┘ └─────────────────┘
                    │               │
                    └───────┬───────┘
                            ▼
                    ┌───────────────┐
                    │   OKX Demo    │
                    │   Trading     │
                    │  (simulated)  │
                    └───────────────┘
```

### How It Works

The system has four main layers that work together:

**1. Market Regime Detection**

Every day the orchestrator classifies the market into one of four regimes: `trending`, `ranging`, `breakout`, or `crisis`. It uses two signals:

- **Indicator-based**: ADX for trend strength, EMA alignment, volatility percentile, Fear & Greed index
- **LLM override**: Claude analyzes macro data (VIX, Gold, DXY, SPX) and confirms or overrides the indicator regime

The detected regime determines which trading instance is active:

| Regime | Active Instance | Strategy |
|--------|----------------|----------|
| `ranging` | ft-sweep | LiquiditySweepStrategy (mean-reversion) |
| `trending` / `breakout` | ft-momentum | MomentumTrendStrategy (trend-following) |
| `crisis` | none | All trading stopped |

**2. Strategy Factory**

Every Sunday, the system automatically generates new trading strategies:

```
  Claude API              Validation Pipeline          Backtest Runner
 ┌──────────┐            ┌──────────────────┐         ┌──────────────┐
 │ Generate │───────────▶│ 1. Security      │────────▶│ Stage 1:     │
 │ Python   │  .py file  │ 2. Syntax        │ passed  │ 30-day mini  │
 │ strategy │            │ 3. Look-ahead    │         │              │
 │ code     │            │ 4. Structure     │         │ Stage 2:     │
 └──────────┘            │ 5. Spot-only     │         │ 6-month full │
                         └──────────────────┘         └──────┬───────┘
                                                             │
                                                             ▼
                                                     ┌──────────────┐
                                                     │  Strategy    │
                                                     │  Registry    │
                                                     │  (SQLite)    │
                                                     │              │
                                                     │  candidate   │
                                                     │  → active    │
                                                     │  → retired   │
                                                     └──────────────┘
```

- **Generation**: Claude writes a complete Freqtrade strategy in Python, targeting a specific market regime
- **Validation**: 5-stage pipeline rejects unsafe code (no `exec/eval`, no file I/O, no network, no look-ahead bias, no shorting)
- **Backtesting**: 2-stage evaluation — quick 30-day filter, then full 6-month backtest via sandboxed Docker container
- **Registry**: SQLite database tracks strategy lifecycle (candidate → active → retired), max 10 active / 30 candidates

**3. Risk Management**

- **Kill switch**: All trading stops if total drawdown reaches -10%
- **Daily limit**: -3% max daily drawdown
- **Crisis mode**: Detected regime = `crisis` stops all instances
- Risk limits are hardcoded and cannot be changed by the LLM

**4. Monitoring**

- **Dashboard**: HTTP server on port 8888 with real-time system status
- **Telegram** (optional): Alerts for regime changes, kill switch triggers, LLM failures
- **Weekly reflections**: Claude reviews the past week's trades and writes analysis to markdown

## Project Structure

```
first-duck-trade/
├── docker-compose.yml                  # 5 services: sweep, momentum, backtest, orchestrator, monitor
├── docker/
│   ├── Dockerfile.orchestrator         # Python + Docker CLI (for backtest runner)
│   └── requirements-orchestrator.txt
├── user_data/
│   ├── configs/                        # Per-instance Freqtrade configs (gitignored, contains API keys)
│   │   ├── config-sweep.json           #   OKX demo trading config for sweep instance
│   │   └── config-momentum.json        #   OKX demo trading config for momentum instance
│   ├── config.json                     # Backtest config (dry_run, no API keys)
│   ├── strategies/
│   │   ├── LiquiditySweepStrategy.py   # Ranging market: liquidity sweep detection
│   │   ├── MomentumTrendStrategy.py    # Trending market: EMA crossover + ADX
│   │   ├── base_generated.py           # Base class for LLM-generated strategies
│   │   └── candidates/                 # LLM-generated strategies land here
│   ├── indicators/
│   │   ├── regime_detector.py          # Market regime classification
│   │   ├── fear_and_greed.py           # Fear & Greed index
│   │   ├── chaikin_money_flow.py       # CMF indicator
│   │   └── whale_liquidity.py          # Whale tracking
│   ├── scripts/
│   │   ├── orchestrator.py             # Main scheduler — regime, risk, strategy routing
│   │   ├── strategy_generator.py       # Claude API → Python strategy code
│   │   ├── validation_pipeline.py      # 5-stage security + quality checks
│   │   ├── backtest_runner.py          # Docker-in-Docker backtest execution
│   │   ├── strategy_registry.py        # SQLite strategy lifecycle management
│   │   ├── fetch_extra_data.py         # VIX, Gold, DXY, SPX from Yahoo Finance
│   │   ├── monitor.py                  # Dashboard HTTP server
│   │   └── notifier.py                 # Telegram alerts
│   ├── data/                           # OHLCV data, regime/risk state, registry DB
│   └── backtest_results/
├── tests/                              # pytest suite (46 tests)
├── TODOS.md                            # Phase-based development roadmap
└── .env                                # API keys (gitignored)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Trading engine | [Freqtrade](https://github.com/freqtrade/freqtrade) (Docker) |
| Strategy generation | Claude API (Anthropic SDK) |
| Job scheduling | APScheduler |
| Orchestration | Docker Compose (5 containers) |
| Strategy registry | SQLite |
| Technical analysis | pandas_ta, numpy, pandas |
| Macro data | yfinance |
| Testing | pytest |

## Notes

- Currently in **Phase 4: Paper Trading** on OKX demo
- Exchange: **OKX** (spot, USDT pairs: BTC, ETH, SOL, XRP)
- Strategy generation runs weekly on **Sunday 02:00 UTC**
- See `TODOS.md` for development roadmap
