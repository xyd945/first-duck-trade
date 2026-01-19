# First Duck Trade 🥇🦆

A Freqtrade-based algo trading bot built on [freqtrade](https://github.com/freqtrade/freqtrade).

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

## Project Structure

```
first-duck-trade/
├── docker-compose.yml       # Freqtrade service
├── user_data/
│   ├── config.json          # API keys & settings
│   ├── strategies/          # Trading strategies
│   ├── data/                # Historical data (downloaded)
│   └── backtest_results/    # Backtest outputs
└── .gitignore
```

## Notes

- Config is set to **dry_run: true** (paper trading)
- Exchange: **OKX Futures** (USDT-margined)
- Make sure to update `config.json` with your API keys before live trading
