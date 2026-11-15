"""Offline development replay and weekly target-weight comparison; no network."""
import math
import json
import sys
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def offline_exchange(cfg):
    from freqtrade.exchange import Okx
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


def replay(config, data, start, end, output):
    from freqtrade.configuration import TimeRange
    from freqtrade.enums import RunMode, TradingMode, CandleType
    from freqtrade.freqtradebot import FreqtradeBot
    from freqtrade.persistence import Trade, enable_database_use
    from freqtrade.resolvers import ExchangeResolver, StrategyResolver
    from h1_backtest import H1Backtesting
    from h1_paper import PaperAdapter, decision_rows
    import research
    import hashlib
    import socket

    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if not pd.Timestamp("2019-01-14", tz="UTC") <= start <= end <= pd.Timestamp("2021-12-31", tz="UTC"):
        raise ValueError("Execution replay is development-only")
    if any(df.date.max() > end for df in data.values()):
        raise ValueError("Replay input escapes the development cutoff")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = deepcopy(config)
    cfg.update(strategy="H1Trend", runmode=RunMode.BACKTEST, trading_mode=TradingMode.SPOT,
               candle_type_def=CandleType.SPOT, user_data_dir=output, datadir=output,
               strategy_path=str(Path(__file__).resolve().parents[1] / "strategies/candidates"),
               export="trades", startup_candle_count=cfg["research_h1"]["historical_startup"],
               db_url=f"sqlite:///{output / 'trades.sqlite'}", initial_state="stopped")
    cfg["timerange"] = f"20190113-{end + pd.Timedelta(days=1):%Y%m%d}"
    descriptor = {"hypothesis": "H1", "strategy": "H1Trend", "params": cfg["research_params"],
                  "data_snapshot_hash": research.composite_hash({p.replace("/", "-"): {
                      "sha256": hashlib.sha256(research.canonical_csv(df)).hexdigest()} for p, df in data.items()})}
    trial = research.start_trial(descriptor, "dev", "paper_replay", f"{start:%Y%m%d}-{end + pd.Timedelta(days=1):%Y%m%d}", research.git_sha(Path(__file__).resolve().parents[2]))
    cfg["h1_trial_id"] = trial
    exchange = offline_exchange(cfg)
    try:
        engine = H1Backtesting(deepcopy(cfg), exchange=exchange)
        engine.backtest_one_strategy(engine.strategylist[0], deepcopy(data), TimeRange.parse_timerange(cfg["timerange"]))
        expected = [b for b in engine.batches if start <= pd.Timestamp(b["time"]) <= end]
    finally:
        exchange.close()
    cfg.update(runmode=RunMode.DRY_RUN, startup_candle_count=cfg["research_h1"]["paper_startup"])
    enable_database_use()
    exchange = offline_exchange(cfg)
    class Clock(datetime):
        current = start.to_pydatetime()
        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz) if tz else cls.current.replace(tzinfo=None)
    prices = {}
    def no_network(*a, **kw):
        raise RuntimeError("Offline replay attempted network I/O")
    load = StrategyResolver.load_strategy
    def strategy(conf):
        instance = load(conf)
        # The offline harness alone bypasses the production guard, with sockets denied.
        instance.bot_start = lambda **kw: None
        return instance
    try:
        with ExitStack() as stack:
            for name, module in list(sys.modules.items()):
                if name.startswith("freqtrade") and getattr(module, "datetime", None) is datetime:
                    stack.enter_context(patch.object(module, "datetime", Clock))
            import freqtrade.util.datetime_helpers as dates
            stack.enter_context(patch.object(dates, "time", lambda: Clock.current.timestamp()))
            stack.enter_context(patch.object(socket.socket, "connect", no_network))
            stack.enter_context(patch.object(ExchangeResolver, "load_exchange", lambda *a, **kw: exchange))
            stack.enter_context(patch.object(StrategyResolver, "load_strategy", strategy))
            stack.enter_context(patch.object(exchange, "validate_config", lambda *a: None))
            stack.enter_context(patch.object(exchange, "get_tickers", lambda *a, **kw: {
                p: {"symbol": p, "last": 100., "bid": 100., "ask": 100.} for p in data}))
            stack.enter_context(patch.object(exchange, "get_rate", lambda pair, **kw: prices[pair]))
            stack.enter_context(patch.object(exchange, "fetch_l2_order_book", lambda pair, *a, **kw:
                {"bids": [[prices[pair], 1e12]], "asks": [[prices[pair], 1e12]]}))
            bot = FreqtradeBot(cfg)
            adapter = PaperAdapter(bot, output / "journal.sqlite")
            actual = []
            for day in pd.date_range(start, end):
                Clock.current = day.to_pydatetime()
                prices = {p: float(df.loc[df.date == day, "open"].iloc[0]) for p, df in data.items()}
                marks = {p: {"price": v, "asof": Clock.current.isoformat()} for p, v in prices.items()}
                actual.append(deepcopy(adapter.step(decision_rows(bot.strategy, data, day), marks, Clock.current)))
                # Exercise the same freshness contract without inventing intraday market prices.
                for minute in range(5, 1440, 5):
                    Clock.current = (day + pd.Timedelta(minutes=minute)).to_pydatetime()
                    marks = {p: {"price": v, "asof": Clock.current.isoformat()} for p, v in prices.items()}
                    adapter.observe(marks, Clock.current)
            count = sum(len(t.orders) for t in Trade.get_trades_proxy())
            report = compare(actual, expected, list(data))
            report.update(days=len(actual), paper_order_count=count, paper_runmode="dry_run",
                          trial_id=trial, start=start.isoformat(), end=end.isoformat(),
                          execution_prices="Historical opens with flat offline books; not historical bid/ask evidence")
            for name, value in (("paper-batches", actual), ("backtest-batches", expected), ("comparison", report)):
                (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
            research.finish_trial(trial, {"success": True, "total_trades": count, "params": report}, "rerun")
            return report
    except Exception as exc:
        research.finish_trial(trial, {"success": False, "error": str(exc)})
        raise
    finally:
        Trade.session.remove()
        exchange.close()



def compare(actual, expected, pairs):
    observed = {row["time"]: row for row in actual}
    if len(observed) != len(actual) or len({r["time"] for r in expected}) != len(expected):
        raise ValueError("Duplicate daily batches")
    details = []
    for reference in expected:
        row = observed.get(reference["time"], {})
        blocked = row.get("status") in ("blocked", "missed", "expired") or row.get("risk_scale", 1.) < 1 - 1e-8
        for pair in pairs:
            want = reference["target_weights"].get(pair)
            got = row.get("target_weights", {}).get(pair)
            target_match = (isinstance(want, (int, float)) and isinstance(got, (int, float))
                            and math.isfinite(want) and math.isfinite(got) and abs(want - got) <= .005 + 1e-12)
            details.append(dict(time=reference["time"], pair=pair, expected=want, actual=got,
                                blocked=blocked, target_match=target_match, match=target_match and not blocked))
    matched = sum(row["match"] for row in details)
    percent = 100 * matched / len(details) if details else 0.
    raw = 100 * sum(row["target_match"] for row in details) / len(details) if details else 0.
    return dict(matched=matched, total=len(details), target_match_percent=raw,
                match_percent=percent, **{"pass": percent >= 95}, details=details)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="user_data/configs/h1_dryrun.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", default="2021-12-02")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    data = {p: pd.read_feather(Path(args.data) / f"{p.replace('/', '_')}-1d.feather")
            for p in cfg["exchange"]["pair_whitelist"]}
    result = replay(cfg, data, args.start, args.end, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

