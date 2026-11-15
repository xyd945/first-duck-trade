"""Real Freqtrade engine on explicitly synthetic daily prices, offline metadata."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("freqtrade")
from freqtrade.configuration import TimeRange
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.exchange import Okx
from tests.test_h1 import config as rule_config


ROOT = Path(__file__).resolve().parents[1]


def engine_config(strategy="H1EqualWeight", pairs=None):
    pairs = pairs or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    return {**rule_config(), "strategy": strategy,
        "dry_run_wallet": 10000, "stake_currency": "USDT", "stake_amount": "unlimited",
        "max_open_trades": 4, "tradable_balance_ratio": 1.0, "timeframe": "1d",
        "trading_mode": TradingMode.SPOT, "candle_type_def": CandleType.SPOT,
        "strategy_path": str(ROOT / "user_data/strategies/candidates"),
        "user_data_dir": ROOT / "user_data", "datadir": ROOT / "user_data/data",
        "dataformat_ohlcv": "feather", "dataformat_trades": "feather", "export": "trades",
        "timerange": "20190113-20190205", "exchange": {"name": "okx",
            "pair_whitelist": pairs, "pair_blacklist": []}, "pairlists": [{"method": "StaticPairList"}],
        "entry_pricing": {"price_side": "other", "use_order_book": False},
        "exit_pricing": {"price_side": "other", "use_order_book": False},
        "order_types": {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False},
        "order_time_in_force": {"entry": "GTC", "exit": "GTC"},
        "unfilledtimeout": {"entry": 10, "exit": 10, "unit": "minutes"}}


def offline_exchange(cfg):
    exchange = Okx(cfg, validate=False, load_leverage_tiers=False)
    markets = {p: {"id": p.replace("/", "-"), "symbol": p, "base": p.split("/")[0],
        "quote": "USDT", "spot": True, "swap": False, "future": False, "type": "spot",
        "active": True, "precision": {"amount": 1e-8, "price": 1e-8},
        "limits": {"amount": {"min": 1e-8, "max": None}, "cost": {"min": 1e-8, "max": None},
                   "leverage": {"min": 1, "max": 1}}, "contract": False, "contractSize": 1.}
        for p in cfg["exchange"]["pair_whitelist"]}
    exchange._markets = markets
    exchange._api.set_markets(markets)
    exchange._api_async.set_markets(markets)
    return exchange


def prices(pairs):
    result = {}
    dates = pd.date_range("2018-01-11", "2019-02-04", tz="UTC")
    for p in pairs:
        price = [100. if d < pd.Timestamp("2019-01-20", tz="UTC") else
                 (80. if p.startswith("BTC") else 120.) for d in dates]
        result[p] = pd.DataFrame({"date": dates, "open": price, "high": price,
                                 "low": price, "close": price, "volume": 1000.})
    return result


def run_synthetic(cfg):
    import research
    from h1_backtest import H1Backtesting
    descriptor = {"hypothesis": "H1", "strategy": cfg["strategy"],
                  "params": cfg["research_params"], "data_snapshot_hash": "synthetic-only"}
    trial = research.start_trial(descriptor, "dev", "backtest", "20190114-20190205", research.git_sha(ROOT))
    cfg["h1_trial_id"] = trial
    exchange = offline_exchange(cfg)
    try:
        engine = H1Backtesting(cfg, exchange=exchange)
        engine.backtest_one_strategy(engine.strategylist[0], prices(cfg["exchange"]["pair_whitelist"]),
                                     TimeRange.parse_timerange(cfg["timerange"]))
        research.finish_trial(trial, {"success": True, "params": {"synthetic": True}}, "invalid")
        return engine.all_bt_content[cfg["strategy"]], deepcopy(engine.batches)
    except Exception as exc:
        research.finish_trial(trial, {"success": False, "error": str(exc)})
        raise
    finally:
        exchange.close()


def test_real_engine_sells_first_independently_of_pair_order():
    cfg = engine_config()
    forward, batches = run_synthetic(deepcopy(cfg))
    cfg["exchange"]["pair_whitelist"].reverse()
    reverse, reverse_batches = run_synthetic(cfg)
    assert batches == reverse_batches
    monday = next(b for b in batches if b["time"].startswith("2019-01-21"))
    assert monday["execution_order"][:2] == ["ETH/USDT", "BTC/USDT"]
    assert monday["cash"] >= -1e-7
    # The sale funds only (1-fee)/(1+fee) of the buy; costs leave an explicit gap.
    assert abs(monday["unfilled"]["ETH/USDT"]) < 1e-7
    expected = monday["requests"]["BTC/USDT"] * (1 - .997 / 1.003)
    assert monday["unfilled"]["BTC/USDT"] == pytest.approx(expected, abs=1e-7)
    assert forward["final_balance"] == pytest.approx(reverse["final_balance"])


@pytest.mark.parametrize("strategy", ["H1Trend", "H1SimpleTrend", "H1EqualWeight", "H1BuyHold"])
def test_real_engine_first_execution_and_only_actual_costs(strategy):
    result, batches = run_synthetic(engine_config(strategy))
    assert batches[0]["time"].startswith("2019-01-14")
    assert all(b["cash"] >= -1e-7 for b in batches)
    trades = result["results"]
    if strategy == "H1BuyHold":
        assert set(trades.pair) == {"BTC/USDT"}
        assert len(trades) == 1
        first = batches[0]
        assert first["cash"] == pytest.approx(0, abs=1e-5)
        assert first["holdings"]["BTC/USDT"] == pytest.approx(100 / 1.003)


def test_registered_last_day_entries_are_not_suppressed():
    cfg = engine_config("H1BuyHold")
    cfg["research_h1"]["path_start"] = "2019-02-04"
    _, batches = run_synthetic(cfg)
    assert len(batches) == 1
    assert batches[0]["holdings"]["BTC/USDT"] > 0


@pytest.mark.parametrize("override", [
    {"timeframe": "1h"}, {"dry_run": False}, {"trading_mode": "futures"},
    {"fee": .1}, {"max_open_trades": 1}, {"minimal_roi": {"0": 0.1}},
    {"stoploss": -.1}, {"trailing_stop": True}, {"position_adjustment_enable": False},
    {"use_exit_signal": True}, {"position_stacking": True}, {"startup_candle_count": 400},
])
def test_engine_rejects_execution_rule_drift_before_connecting(override):
    from h1_backtest import H1Backtesting
    cfg = engine_config()
    cfg.update(override)
    with pytest.raises(ValueError, match="contract"):
        H1Backtesting(cfg)


def test_ccxt_version_pin_is_enforced(monkeypatch):
    import ccxt
    from h1_backtest import H1Backtesting
    monkeypatch.setattr(ccxt, "__version__", "unexpected")
    with pytest.raises(ValueError, match="version"):
        H1Backtesting(engine_config())
