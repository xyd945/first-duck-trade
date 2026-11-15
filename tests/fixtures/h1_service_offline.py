"""Offline subprocess boundary: real service, engine, database and signals."""
import sys
import time
import socket
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "user_data/scripts"))
import h1_service
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.resolvers import ExchangeResolver
from h1_replay import offline_exchange
import freqtrade.util.datetime_helpers as dates
import pandas as pd
import numpy as np

mode, script, *arguments = sys.argv[1:]
sys.argv = [script, *arguments]
started = time.monotonic()


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        value = datetime(2026, 12, 19, 0, 1, tzinfo=timezone.utc) + timedelta(seconds=time.monotonic() - started)
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)


def no_network(*args, **kwargs):
    raise RuntimeError("Offline service attempted network access")


socket.socket.connect = no_network
for name, module in list(sys.modules.items()):
    if (name.startswith("freqtrade") or name == "h1_service") and getattr(module, "datetime", None) is datetime:
        module.datetime = Clock
dates.time = lambda: Clock.now(timezone.utc).timestamp()


def load_exchange(config, **kwargs):
    if mode == "exit":
        raise RuntimeError("offline constructor failure")
    if mode == "hang":
        time.sleep(60)
    time.sleep(.3)
    exchange = offline_exchange(config)
    exchange.validate_config = lambda *a, **kw: None
    exchange.get_tickers = lambda *a, **kw: {
        p: {"symbol": p, "last": 100., "bid": 100., "ask": 100.}
        for p in config["exchange"]["pair_whitelist"]}
    exchange.get_rate = lambda *a, **kw: 100.
    exchange.fetch_l2_order_book = lambda *a, **kw: {"bids": [[100., 1e9]], "asks": [[100., 1e9]]}
    exchange._api.fetch_ohlcv = lambda *a, **kw: [[int(pd.Timestamp("2026-12-19", tz="UTC").timestamp() * 1000), 100., 100., 100., 100., 1.]]
    return exchange


original = FreqtradeBot.__init__


def initialise(self, config):
    original(self, config)
    day = pd.Timestamp("2026-12-19", tz="UTC")
    index = pd.date_range(end=day - pd.Timedelta(days=1), periods=400)
    close = 100. + np.sin(np.arange(400))
    if mode in ("entry-error", "resume-entry"):
        close[-2:] = [90., 110.]
    frame = pd.DataFrame(dict(date=index, open=close, high=close, low=close, close=close, volume=1000.))
    self.dataprovider.refresh = lambda *a, **kw: None
    self.dataprovider.ohlcv = lambda *a, **kw: frame
    if mode == "tick-error":
        self.exchange.get_tickers = no_network
    if mode == "entry-error":
        enter = self.execute_entry
        attempts = []
        def fail_second(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 2:
                raise RuntimeError("offline second-entry failure")
            return enter(*args, **kwargs)
        self.execute_entry = fail_second


with patch.object(ExchangeResolver, "load_exchange", load_exchange), patch.object(FreqtradeBot, "__init__", initialise):
    raise SystemExit(h1_service.main())
