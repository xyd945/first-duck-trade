"""Isolated H1 paper service; never starts legacy services or live trading."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sqlite3
import fcntl
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PAPER_TOKEN = object()
FORWARD_START = "2026-12-19T00:00:00+00:00"
STARTUP_TIMEOUT = 300
STOP_TIMEOUT = 30


def strategy_hash(params):
    fields = {k: params[k] for k in ("ladder", "sigma_target", "vol_window", "rebalance_band")}
    source = (ROOT / "user_data/strategies/candidates/H1Trend.py").read_bytes()
    return hashlib.sha256(source + json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_config(config, directory, ledger, *, placeholder=False):
    cfg = deepcopy(config)
    fixed = {"dry_run": True, "dry_run_wallet": 10000, "trading_mode": "spot", "strategy": "H1Trend",
             "timeframe": "1d", "startup_candle_count": 400, "fee": .0015, "minimal_roi": {},
             "stoploss": -1., "trailing_stop": False, "position_adjustment_enable": True,
             "use_exit_signal": False, "stake_currency": "USDT", "stake_amount": "unlimited",
             "max_open_trades": 4, "tradable_balance_ratio": 1., "force_entry_enable": False,
             "pairlists": [{"method": "StaticPairList"}]}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("H1 paper execution contract cannot be overridden")
    if cfg.get("add_config_files") or cfg.get("freqai") or cfg.get("external_message_consumer"):
        raise ValueError("No external configuration or consumers")
    for key in ("api_server", "telegram"):
        if cfg.get(key, {}).get("enabled") is not False:
            raise ValueError("H1 control APIs must stay disabled")
    exchange = cfg["exchange"]
    if (set(exchange) != {"name", "key", "secret", "password", "enable_ws", "pair_whitelist", "pair_blacklist"}
            or exchange["name"] != "okx" or any(exchange.get(k) for k in ("key", "secret", "password", "enable_ws"))
            or sorted(exchange["pair_whitelist"]) != ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]):
        raise ValueError("Only keyless OKX public spot data is allowed")
    if any(cfg["order_types"].get(k) != "market" for k in ("entry", "exit", "stoploss")) or cfg["order_types"]["stoploss_on_exchange"]:
        raise ValueError("H1 requires simulated market orders")
    rules = json.loads((ROOT / "user_data/configs/research-h1.example.json").read_text())["h1"]
    variable = {"designated_portfolio_notional", "paper_mapping_status"}
    if {k: v for k, v in cfg["research_h1"].items() if k not in variable} != {k: v for k, v in rules.items() if k not in variable}:
        raise ValueError("H1 registered rules cannot be overridden")
    from accounting import number
    if number(cfg["research_h1"]["designated_portfolio_notional"], "designated notional", positive=True) < 10000:
        raise ValueError("Designated paper notional cannot be below the wallet")
    if placeholder:
        cfg["research_params"] = {"ladder": [20, 60, 180], "sigma_target": .45,
                                  "vol_window": 60, "rebalance_band": .025, "cost_rate": .0015}
        selection = {"placeholder": True, "config_id": "M45", "config_hash": strategy_hash(cfg["research_params"])}
    else:
        try:
            with sqlite3.connect(f"file:{Path(ledger).resolve()}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM trial_ledger WHERE hypothesis='H1' AND verdict='selected'").fetchone()
        except sqlite3.Error:
            row = None
        if not row:
            raise ValueError("A selected H1 ledger row is required; --placeholder is explicit test mode")
        descriptor = json.loads(row["params_json"])["config"]
        params = descriptor["params"]
        if (row["split"] != "selection" or descriptor["strategy"] != "H1Trend"
                or strategy_hash(params) != row["config_hash"]):
            raise ValueError("Selected strategy source or parameters changed")
        cfg["research_params"] = {**params, "cost_rate": .0015}
        selection = {"placeholder": False, "trial_id": row["trial_id"], "config_hash": row["config_hash"]}
    params = cfg["research_params"]
    if (params["ladder"] not in ([10, 30, 90], [20, 60, 180], [40, 120, 365])
            or params["sigma_target"] not in (.3, .45, .6) or params["vol_window"] != 60 or params["rebalance_band"] != .025):
        raise ValueError("Selected parameters escape the registered grid")
    directory = Path(directory).resolve()
    cfg.update(h1_selection=selection, user_data_dir=directory, datadir=directory / "data",
               db_url=f"sqlite:///{directory / 'trades.sqlite'}",
               strategy_path=str(ROOT / "user_data/strategies/candidates"), initial_state="stopped")
    return cfg


def tick(adapter):
    import pandas as pd
    import paper_risk
    import research
    from accounting import number
    from h1_paper import decision_rows
    bot = adapter.bot
    acquired = datetime.now(timezone.utc)
    day = pd.Timestamp(acquired).floor("D")
    if acquired < datetime.fromisoformat(FORWARD_START):
        raise ValueError("Paper service cannot start before 2026-12-19 00:00 UTC")
    adapter.expire(day)
    tickers = bot.exchange.get_tickers(bot.active_pair_whitelist, cached=False)
    fetch_book = bot.exchange.fetch_l2_order_book
    books = {p: fetch_book(p, 20) for p in bot.active_pair_whitelist}
    prices = {}
    for pair in bot.active_pair_whitelist:
        ticker = tickers[pair]
        stamp = acquired
        if ticker.get("timestamp"):
            stamp = datetime.fromtimestamp(ticker["timestamp"] / 1000, timezone.utc)
        prices[pair] = {"price": number(ticker["last"], "mark", positive=True), "asof": stamp.isoformat(),
                        "buy_price": max(number(v[0], "ask", positive=True) for v in books[pair]["asks"]),
                        "sell_price": min(number(v[0], "bid", positive=True) for v in books[pair]["bids"])}
    existing = adapter.get("batch", day.isoformat())
    previous = (day - pd.Timedelta(days=1)).isoformat()
    terminal = existing and existing["status"] in ("finished", "blocked", "missed", "error")
    if terminal and (day == pd.Timestamp(FORWARD_START) or adapter.get("nav", previous)):
        adapter.observe(prices, datetime.now(timezone.utc))
        return existing
    bot.dataprovider.refresh(bot.pairlists.create_pair_list(bot.active_pair_whitelist), [])
    directory = Path(bot.config["user_data_dir"]) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    data = {}
    for pair in bot.active_pair_whitelist:
        latest = bot.dataprovider.ohlcv(pair)
        latest = latest.loc[latest.date < day].copy()
        research.canonical_csv(latest)
        path = directory / f"{pair.replace('/', '_')}-1d.feather"
        if path.exists():
            old = pd.read_feather(path)
            overlap = old.merge(latest, on="date", suffixes=("_old", "_new"))
            if any((overlap[f"{key}_old"] != overlap[f"{key}_new"]).any()
                   for key in ("open", "high", "low", "close", "volume")):
                raise ValueError("Captured confirmed candle revised; reconciliation required")
            latest = pd.concat([old, latest]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        temporary = path.with_suffix(".pending")
        latest.to_feather(temporary)
        temporary.replace(path)
        data[pair] = latest
    rows = decision_rows(bot.strategy, data, day)
    now = datetime.now(timezone.utc)
    if pd.Timestamp(now).floor("D") != day:
        raise ValueError("Daily boundary crossed during acquisition")
    snapshot, state = adapter.observe(prices, now)
    complete = all(r["date"] == day - pd.Timedelta(days=1) and r["can_trade"] for r in rows.values())
    if snapshot and complete and day > pd.Timestamp(FORWARD_START) and not adapter.get("nav", previous):
        account = snapshot["accounts"][0]
        equity = account["cash"] + sum(p["quantity"] * rows[p["pair"]]["close"] for p in account["positions"])
        adapter.save("nav", previous, {"date": previous, "equity": equity})
    if terminal:
        return existing
    if complete and (now - day).total_seconds() <= 300 and adapter.get("reference", day.isoformat()) is None:
        reference = {}
        for pair in bot.active_pair_whitelist:
            candle = bot.exchange._api.fetch_ohlcv(pair, "1d", since=int(day.timestamp() * 1000), limit=1)
            if not candle or candle[0][0] != int(day.timestamp() * 1000):
                raise ValueError("Missing next-open reference")
            reference[pair] = number(candle[0][1], "reference open", positive=True)
        adapter.save("reference", day.isoformat(), reference)
    now = datetime.now(timezone.utc)
    if pd.Timestamp(now).floor("D") != day:
        raise ValueError("Daily boundary crossed during acquisition")
    if existing is None or existing["status"] == "waiting_data":
        late = (now - day).total_seconds() > 300
        if late or not complete:
            batch = dict(time=day.isoformat(), status="missed" if late else "waiting_data", divergence=True,
                         risk_reason="late daily decision" if late else "waiting for confirmed prior-day candles",
                         target_weights={p: float(r["target_weight"]) if pd.notna(r["target_weight"]) else None for p, r in rows.items()})
            adapter.save("batch", day.isoformat(), batch)
            return batch

    # Pin one observed book per tick so market-fill simulation cannot outrun its cash budget.
    bot.exchange.fetch_l2_order_book = lambda pair, *a, **kw: books[pair]
    try:
        return adapter.step(rows, prices, now)
    finally:
        bot.exchange.fetch_l2_order_book = fetch_book


def status(directory):
    directory = Path(directory)
    running = False
    lock = directory / "service.lock"
    if lock.exists():
        with lock.open() as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                running = True
    result = {"running": running, "directory": str(directory)}
    heartbeat = directory / "heartbeat.json"
    if heartbeat.exists():
        result["heartbeat"] = json.loads(heartbeat.read_text())
        stamp = datetime.fromisoformat(result["heartbeat"]["asof"])
        result["heartbeat_fresh"] = 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 300
    result["ready"] = running and result.get("heartbeat_fresh", False) and result.get("heartbeat", {}).get("state") == "ready"
    return result


def start(config_path, directory, ledger, *, placeholder=False, now=None):
    now = now or datetime.now(timezone.utc)
    if now < datetime.fromisoformat(FORWARD_START):
        raise ValueError("Paper service cannot start before 2026-12-19 00:00 UTC")
    prepare_config(json.loads(Path(config_path).read_text()), directory, ledger, placeholder=placeholder)
    if status(directory)["running"]:
        raise ValueError("H1 service is already running")
    Path(directory).mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--config", str(Path(config_path).resolve()),
               "--directory", str(Path(directory).resolve()), "--ledger", str(Path(ledger).resolve())]
    if placeholder:
        command.append("--placeholder")
    with (Path(directory) / "service.log").open("a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True, cwd=ROOT)
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            result = status(directory)
            if child.poll() is not None:
                raise RuntimeError("H1 child exited; inspect service.log")
            if result.get("heartbeat", {}).get("pid") == child.pid:
                if result["ready"]:
                    return result
                if result["heartbeat"]["state"] == "halted":
                    raise RuntimeError("H1 startup halted; inspect service.log")
            time.sleep(.1)
        raise RuntimeError("Startup readiness timed out; inspect service.log")
    except BaseException:
        # A failed start must not leave a child that can begin trading later.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=STOP_TIMEOUT)
        raise


def stop(directory):
    result = status(directory)
    if not result["running"]:
        return result
    pid = result.get("heartbeat", {}).get("pid")
    if not isinstance(pid, int) or pid <= 1:
        raise ValueError("No verified H1 process identity")
    command = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True)
    if str(Path(__file__).resolve()) not in command or str(Path(directory).resolve()) not in command:
        raise ValueError("PID identity changed; refusing to signal another process")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not (result := status(directory))["running"]:
            return result
        time.sleep(.1)
    raise RuntimeError("Stop not confirmed; process still holds the H1 lock")


def heartbeat(directory, value):
    value = {**value, "pid": os.getpid(), "asof": datetime.now(timezone.utc).isoformat()}
    temporary = Path(directory) / "heartbeat.pending"
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(Path(directory) / "heartbeat.json")


def run(config_path, directory, ledger, *, placeholder=False):
    if datetime.now(timezone.utc) < datetime.fromisoformat(FORWARD_START):
        raise ValueError("Paper service cannot start before 2026-12-19 00:00 UTC")
    import ccxt
    import freqtrade
    import logging
    from threading import Event
    from freqtrade.enums import RunMode, CandleType, TradingMode
    from freqtrade.freqtradebot import FreqtradeBot
    from h1_paper import PaperAdapter
    import research
    if freqtrade.__version__ != "2026.10" or ccxt.__version__ != "4.5.90":
        raise ValueError("H1 requires Freqtrade 2026.10 and ccxt 4.5.90")
    cfg = prepare_config(json.loads(Path(config_path).read_text()), directory, ledger, placeholder=placeholder)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO)
    with (directory / "service.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop_event = Event()
        handlers = {sig: signal.signal(sig, lambda *_: stop_event.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
        identity = {"selection": cfg["h1_selection"], "rules": cfg["research_h1"],
                    "params": cfg["research_params"], "git_sha": research.git_sha(ROOT)}
        path = directory / "identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Paper identity changed; use a separate directory or reconcile explicitly")
        if not path.exists():
            path.write_text(json.dumps(identity, indent=2))
        heartbeat(directory, {"state": "starting", "selection": cfg["h1_selection"]})
        cfg.update(runmode=RunMode.DRY_RUN, candle_type_def=CandleType.SPOT, trading_mode=TradingMode.SPOT,
                   _h1_paper_adapter=PAPER_TOKEN)
        bot = None
        try:
            bot = FreqtradeBot(cfg)
            adapter = PaperAdapter(bot, directory / "journal.sqlite")
            while not stop_event.is_set():
                try:
                    batch = tick(adapter)
                    state = adapter.get("risk", "current")
                    health = "waiting" if batch["status"] == "waiting_data" else ("ready" if state["allow_new_exposure"] else "halted")
                    heartbeat(directory, {"state": health, "batch": batch["time"], "batch_status": batch["status"],
                                          "risk": state, "selection": cfg["h1_selection"]})
                except Exception as exc:
                    logging.exception("H1 tick failed closed")
                    stamp = datetime.now(timezone.utc)
                    key = stamp.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                    if adapter.get("error", key) is None:
                        adapter.save("error", key, {"time": key, "observed": stamp.isoformat(), "error": str(exc)})
                    adapter.observe({}, datetime.now(timezone.utc))
                    heartbeat(directory, {"state": "halted", "error": str(exc), "risk": adapter.get("risk", "current")})
                if stop_event.wait(60):
                    break
        finally:
            if bot is not None:
                bot.cleanup()
            heartbeat(directory, {"state": "stopped"})
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status", "run"))
    parser.add_argument("--config", default=str(ROOT / "user_data/configs/h1_dryrun.json"))
    parser.add_argument("--directory", default=str(ROOT / "user_data/data/h1-paper"))
    parser.add_argument("--ledger", default=str(ROOT / "user_data/data/trial_ledger.sqlite"))
    parser.add_argument("--placeholder", action="store_true", help="Explicit M45 non-selected test mode; never G2 evidence")
    args = parser.parse_args()
    if args.action in ("status", "stop"):
        result = (status if args.action == "status" else stop)(args.directory)
    else:
        result = (start if args.action == "start" else run)(args.config, args.directory, args.ledger, placeholder=args.placeholder)
    if result is not None:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    # Keep the identity token singular when this file is executed as a script.
    sys.modules["h1_service"] = sys.modules[__name__]
    raise SystemExit(main())
