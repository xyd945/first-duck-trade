"""Real pinned dry-run fills, SQL trades and wallets; public I/O replaced offline."""
import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pandas as pd
import pytest

pytest.importorskip("freqtrade")
from freqtrade.enums import RunMode
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from tests.test_h1_engine import ROOT, offline_exchange


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    import sys
    class Clock(datetime):
        current = datetime(2021, 12, 6, tzinfo=timezone.utc)
        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz) if tz else cls.current.replace(tzinfo=None)
    for name, module in list(sys.modules.items()):
        if name.startswith("freqtrade") and getattr(module, "datetime", None) is datetime:
            monkeypatch.setattr(module, "datetime", Clock)
    import freqtrade.util.datetime_helpers as dates
    monkeypatch.setattr(dates, "time", lambda: Clock.current.timestamp())
    monkeypatch.setattr(sys.modules[__name__], "datetime", Clock)
    return Clock


@pytest.fixture
def paper_bot(tmp_path, monkeypatch):
    from H1Trend import H1Trend
    from freqtrade.resolvers import ExchangeResolver, StrategyResolver
    from freqtrade.persistence import enable_database_use
    enable_database_use()
    cfg = json.loads((ROOT / "user_data/configs/h1_dryrun.json").read_text())
    cfg.update(runmode=RunMode.DRY_RUN, user_data_dir=tmp_path,
               strategy_path=str(ROOT / "user_data/strategies/candidates"),
               db_url=f"sqlite:///{tmp_path / 'trades.sqlite'}")
    exchange = offline_exchange(cfg)
    exchange.validate_config = Mock()
    exchange.get_tickers = Mock(return_value={p: {"symbol": p, "last": 100., "bid": 100., "ask": 100.}
                                              for p in cfg["exchange"]["pair_whitelist"]})
    exchange.get_rate = Mock(return_value=100.)
    exchange.fetch_l2_order_book = Mock(return_value={"bids": [[100., 1e9]], "asks": [[100., 1e9]]})
    monkeypatch.setattr(ExchangeResolver, "load_exchange", lambda *a, **kw: exchange)
    # Only this offline fixture bypasses the still-disabled production startup guard.
    load = StrategyResolver.load_strategy
    def strategy(config):
        instance = load(config)
        instance.bot_start = lambda **kw: None
        return instance
    monkeypatch.setattr(StrategyResolver, "load_strategy", strategy)
    bot = FreqtradeBot(cfg)
    yield bot
    Trade.session.remove()
    exchange.close()


def quotes(bot, now):
    return {p: {"price": 100., "asof": now.isoformat()} for p in bot.active_pair_whitelist}


def enter(bot, pair, quantity, now):
    bot.strategy.batch = {"time": pd.Timestamp(now).floor("D"), "orders": {pair: quantity}}
    assert bot.execute_entry(pair, quantity * 100., price=100.)


def test_reconcile_real_open_entry_deducts_deferred_fee(paper_bot):
    from h1_paper import reconcile
    now = datetime.now(timezone.utc)
    enter(paper_bot, "BTC/USDT", 20., now)
    now = datetime.now(timezone.utc)
    snapshot = reconcile(paper_bot, quotes(paper_bot, now), now)
    account = snapshot["accounts"][0]
    assert account["cash"] == pytest.approx(10000 - 2000 * 1.0015)
    assert account["positions"][0]["quantity"] == pytest.approx(20)
    assert snapshot["accounts"][1]["cash"] == 240000
    assert paper_bot.wallets.get_total("USDT") == pytest.approx(account["cash"] + 3.)


def rows(bot, weights, marks=None):
    return {p: {"can_trade": True, "rebalance": True, "signal_change": False,
                "target_weight": weights.get(p, 0.), "close": (marks or {}).get(p, 100.)}
            for p in bot.active_pair_whitelist}


def test_funded_rotation_settles_sells_before_common_factor_buys(paper_bot, tmp_path, clock):
    from h1_paper import PaperAdapter
    bot = paper_bot
    now = clock.current
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    adapter.observe(quotes(bot, now), now)
    quantity = 10000 / (4 * 100 * 1.0015)
    for pair in bot.active_pair_whitelist:
        enter(bot, pair, quantity, now)
    marks = {"BTC/USDT": 80., "ETH/USDT": 120., "SOL/USDT": 100., "XRP/USDT": 100.}
    bot.exchange.get_rate.side_effect = lambda pair, **kw: marks[pair]
    bot.exchange.fetch_l2_order_book.side_effect = lambda pair, *a, **kw: {
        "bids": [[marks[pair], 1e9]], "asks": [[marks[pair], 1e9]]}
    prices = {p: {"price": v, "asof": now.isoformat()} for p, v in marks.items()}
    batch = adapter.step(rows(bot, dict.fromkeys(marks, .25), marks), prices, now)
    assert batch["status"] == "finished"
    assert batch["attempted"] == ["ETH/USDT", "BTC/USDT"]
    assert batch["cash"] >= 0
    assert batch["unfilled"]["ETH/USDT"] == pytest.approx(0, abs=1e-7)
    assert batch["unfilled"]["BTC/USDT"] > 0
    assert batch["cash_scale"] == pytest.approx(.9985 / 1.0015, abs=1e-6)
    before = sum(len(t.orders) for t in Trade.get_trades_proxy())
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    assert adapter.step(rows(bot, dict.fromkeys(marks, .25), marks), prices, now) == batch
    assert sum(len(t.orders) for t in Trade.get_trades_proxy()) == before


def test_pending_sale_expires_before_next_decision_without_funding_buy(paper_bot, tmp_path, clock, monkeypatch):
    from h1_paper import PaperAdapter
    bot, now = paper_bot, clock.current
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    adapter.observe(quotes(bot, now), now)
    enter(bot, "ETH/USDT", 20, now)
    create = bot.exchange.create_order
    def pending_sale(**kw):
        result = create(**kw)
        if kw["side"] == "sell":
            result.update(status="open", filled=0., remaining=result["amount"], fee=None)
        return result
    monkeypatch.setattr(bot.exchange, "create_order", pending_sale)
    first = adapter.step(rows(bot, {"BTC/USDT": .2}), quotes(bot, now), now)
    assert first["status"] == "selling"
    assert first["attempted"] == ["ETH/USDT"]
    assert not any(t.pair == "BTC/USDT" for t in Trade.get_open_trades())
    clock.current += pd.Timedelta(days=1)
    for row in (next_rows := rows(bot, {})).values():
        row.update(rebalance=False)
    adapter.step(next_rows, quotes(bot, clock.current), clock.current)
    prior = adapter.get("batch", now.isoformat())
    assert prior["status"] == "expired"
    assert prior["unfilled"]["BTC/USDT"] > 0
    assert not any(t.open_orders for t in Trade.get_open_trades())
    assert not any(t.pair == "BTC/USDT" for t in Trade.get_open_trades())


@pytest.mark.parametrize("bad", ["missing", "stale", "future", "wallet", "sql"])
def test_bad_snapshot_latches_kill_and_logs_blocked_targets(paper_bot, tmp_path, clock, monkeypatch, bad):
    from h1_paper import PaperAdapter
    bot, now = paper_bot, clock.current
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    prices = quotes(bot, now)
    adapter.observe(prices, now)
    if bad == "missing":
        prices.pop("BTC/USDT")
    elif bad in ("stale", "future"):
        prices["BTC/USDT"]["asof"] = (now + pd.Timedelta(seconds=-301 if bad == "stale" else 1)).isoformat()
    elif bad == "wallet":
        original = bot.wallets.get_total
        monkeypatch.setattr(bot.wallets, "get_total", lambda currency: original(currency) + (1 if currency == "USDT" else 0))
    else:
        enter(bot, "ETH/USDT", 2, now)
        Trade.get_open_trades()[0].amount += 1
        Trade.commit()
    batch = adapter.step(rows(bot, {"BTC/USDT": .1}), prices, now)
    assert adapter.get("risk", "current")["kill_switch_active"]
    assert batch["divergence"]
    assert batch["target_weights"]["BTC/USDT"] == .1
    assert not any(t.pair == "BTC/USDT" for t in Trade.get_open_trades())


def test_restart_rechecks_risk_before_dispatching_saved_buys(paper_bot, tmp_path, clock):
    from h1_paper import PaperAdapter
    now, bot = clock.current, paper_bot
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    adapter.observe(quotes(bot, now), now)
    batch = bot.strategy.plan_batch(pd.Timestamp(now), rows(bot, {"BTC/USDT": .1}), {}, 10000.)
    batch.update(time=now.isoformat(), status="buying", attempted=[], orders={"BTC/USDT": 10.},
                 target_weights={"BTC/USDT": .1})
    adapter.save("batch", now.isoformat(), batch)
    state = adapter.get("risk", "current")
    state["kill_switch_active"] = True
    adapter.save("risk", "current", state)
    batch = adapter.step(rows(bot, {"BTC/USDT": .1}), quotes(bot, now), now)
    assert Trade.get_open_trades() == []
    assert batch["unfilled"]["BTC/USDT"] == 10.
    assert batch["risk_reason"] == "kill_latched"


def test_market_depth_is_budgeted_before_common_factor_buy(paper_bot, tmp_path, clock):
    from h1_paper import PaperAdapter
    bot, now = paper_bot, clock.current
    bot.exchange.fetch_l2_order_book.return_value = {"bids": [[100., 1e9]], "asks": [[105., 1e9]]}
    prices = {p: {"price": 100., "buy_price": 105., "sell_price": 100., "asof": now.isoformat()}
              for p in bot.active_pair_whitelist}
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    batch = adapter.step(rows(bot, dict.fromkeys(prices, .25)), prices, now)
    assert batch["status"] == "finished"
    assert 0 <= batch["cash"] < .00001
    assert all(q < 25. for q in batch["holdings"].values())


@pytest.fixture
def service_tick(paper_bot, tmp_path, clock, monkeypatch):
    import numpy as np
    import h1_service
    from h1_paper import PaperAdapter
    now = pd.Timestamp("2026-12-19", tz="UTC")
    clock.current = now.to_pydatetime()
    monkeypatch.setattr(h1_service, "datetime", clock)
    dates = pd.date_range(end=now - pd.Timedelta(days=1), periods=400)
    close = 100. + np.sin(np.arange(400))
    frame = pd.DataFrame(dict(date=dates, open=close, high=close, low=close, close=close, volume=1000.))
    bot = paper_bot
    bot.dataprovider.refresh = Mock()
    bot.dataprovider.ohlcv = Mock(return_value=frame)
    bot.exchange._api.fetch_ohlcv = Mock(return_value=[[int(now.timestamp()*1000), 100., 100., 100., 100., 1.]])
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    return adapter, frame


def test_service_tick_persists_confirmed_data_and_one_daily_batch(service_tick, tmp_path, clock):
    import h1_service
    adapter, frame = service_tick
    now = pd.Timestamp(clock.current)
    batch = h1_service.tick(adapter)
    assert batch["time"] == now.isoformat()
    assert len(batch["target_weights"]) == 4
    assert len(list((tmp_path / "data").glob("*.feather"))) == 4
    assert adapter.get("reference", now.isoformat())["BTC/USDT"] == 100.
    assert h1_service.tick(adapter) == batch


def test_late_day_is_terminal_but_keeps_prior_nav_and_next_day_running(service_tick, clock):
    import h1_service
    adapter, frame = service_tick
    clock.current = datetime(2026, 12, 20, 0, 6, tzinfo=timezone.utc)
    frame = pd.concat([frame, frame.iloc[[-1]].assign(date=pd.Timestamp("2026-12-19", tz="UTC"))], ignore_index=True)
    adapter.bot.dataprovider.ohlcv.return_value = frame
    adapter.bot.exchange._api.fetch_ohlcv.side_effect = lambda *a, **kw: [[kw["since"], 100., 100., 100., 100., 1.]]
    first = h1_service.tick(adapter)
    assert first["status"] == "missed"
    assert adapter.get("nav", "2026-12-19T00:00:00+00:00")["equity"] == 10000.
    clock.current = datetime(2026, 12, 21, 0, 1, tzinfo=timezone.utc)
    adapter.bot.dataprovider.ohlcv.return_value = pd.concat([
        frame, frame.iloc[[-1]].assign(date=pd.Timestamp("2026-12-20", tz="UTC"))], ignore_index=True)
    assert h1_service.tick(adapter)["time"] == "2026-12-21T00:00:00+00:00"
    assert adapter.get("batch", first["time"]) == first


def test_delayed_candle_retries_without_fabricating_nav(service_tick, clock):
    import h1_service
    adapter, frame = service_tick
    clock.current = datetime(2026, 12, 20, tzinfo=timezone.utc)
    adapter.bot.exchange._api.fetch_ohlcv.side_effect = lambda *a, **kw: [[kw["since"], 100., 100., 100., 100., 1.]]
    batch = h1_service.tick(adapter)
    assert batch["status"] == "waiting_data"
    assert adapter.get("nav", "2026-12-19T00:00:00+00:00") is None
    assert not Trade.get_trades_proxy()
    clock.current += pd.Timedelta(minutes=1)
    adapter.bot.dataprovider.ohlcv.return_value = pd.concat([
        frame, frame.iloc[[-1]].assign(date=pd.Timestamp("2026-12-19", tz="UTC"))], ignore_index=True)
    assert h1_service.tick(adapter)["status"] == "finished"
    assert adapter.get("nav", "2026-12-19T00:00:00+00:00")["equity"] == 10000.


def test_late_service_tick_still_expires_previous_pending_sale(service_tick, clock, monkeypatch):
    import h1_service
    adapter, frame = service_tick
    bot, first = adapter.bot, clock.current
    adapter.observe(quotes(bot, first), first)
    enter(bot, "ETH/USDT", 20, first)
    create = bot.exchange.create_order
    def pending_sale(**kwargs):
        result = create(**kwargs)
        if kwargs["side"] == "sell":
            result.update(status="open", filled=0., remaining=result["amount"], fee=None)
        return result
    monkeypatch.setattr(bot.exchange, "create_order", pending_sale)
    assert adapter.step(rows(bot, {"BTC/USDT": .2}), quotes(bot, first), first)["status"] == "selling"
    clock.current += pd.Timedelta(days=1, minutes=6)
    bot.dataprovider.ohlcv.return_value = pd.concat([
        frame, frame.iloc[[-1]].assign(date=pd.Timestamp("2026-12-19", tz="UTC"))], ignore_index=True)
    assert h1_service.tick(adapter)["status"] == "missed"
    assert adapter.get("batch", first.isoformat())["status"] == "expired"
    assert not any(t.open_orders for t in Trade.get_open_trades())


def test_reference_fetch_cannot_cross_the_decision_deadline(service_tick, clock):
    import h1_service
    adapter, frame = service_tick
    clock.current += pd.Timedelta(minutes=4, seconds=59)
    def reference(*args, **kwargs):
        clock.current += pd.Timedelta(seconds=1)
        return [[kwargs["since"], 100., 100., 100., 100., 1.]]
    adapter.bot.exchange._api.fetch_ohlcv.side_effect = reference
    assert h1_service.tick(adapter)["status"] == "missed"
    assert not Trade.get_trades_proxy()


def test_future_ticker_is_not_clipped_to_a_fresh_quote(service_tick, clock):
    import h1_service
    adapter, frame = service_tick
    for ticker in adapter.bot.exchange.get_tickers.return_value.values():
        ticker["timestamp"] = (clock.current + pd.Timedelta(minutes=1)).timestamp() * 1000
    batch = h1_service.tick(adapter)
    assert batch["status"] == "blocked"
    assert adapter.get("risk", "current")["kill_switch_active"]
    assert not Trade.get_trades_proxy()


def test_wallet_refresh_finishes_after_tick_acquisition_time(paper_bot, tmp_path, clock, monkeypatch):
    from h1_paper import PaperAdapter
    bot, now = paper_bot, clock.current
    original = bot.wallets.update
    def delayed(*a, **kw):
        clock.current += pd.Timedelta(milliseconds=1)
        return original(*a, **kw)
    monkeypatch.setattr(bot.wallets, "update", delayed)
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    batch = adapter.step(rows(bot, {"BTC/USDT": .1}), quotes(bot, now), now)
    assert batch["holdings"]["BTC/USDT"] > 0
    assert batch["risk_scale"] == 1.
    assert batch.get("finished_at") == clock.current.isoformat()
    assert pd.Timestamp(batch["finished_at"]) > pd.Timestamp(now)
    assert adapter.get("batch", batch["time"]) == batch
    clock.current += pd.Timedelta(minutes=1)
    restarted = PaperAdapter(bot, tmp_path / "journal.sqlite")
    assert restarted.step(rows(bot, {"BTC/USDT": .1}), quotes(bot, clock.current), clock.current) == batch


@pytest.mark.parametrize("bad", ["wallet", "sql", "fee", "position", "peak", "stale", "capital", "audit"])
def test_manual_reconciliation_refuses_bad_evidence_without_state_changes(paper_bot, tmp_path, clock, monkeypatch, bad):
    import sqlite3
    from h1_paper import PaperAdapter
    from tests.test_h1_recovery import records
    bot, now = paper_bot, clock.current
    adapter = PaperAdapter(bot, tmp_path / "journal.sqlite")
    adapter.observe(quotes(bot, now), now)
    enter(bot, "BTC/USDT", 20., now)
    previous = adapter.get("risk", "current")
    previous.update(kill_switch_active=True, allow_new_exposure=False)
    prices = quotes(bot, now)
    trade = Trade.get_open_trades()[0]
    if bad == "wallet":
        total = bot.wallets.get_total
        monkeypatch.setattr(bot.wallets, "get_total", lambda currency: total(currency) + (1 if currency == "USDT" else 0))
    elif bad == "sql":
        trade.amount += 1
    elif bad == "fee":
        trade.fee_open = .003
    elif bad == "position":
        trade.leverage = 2
    elif bad == "peak":
        previous["peak_equity"] = 270000.
    elif bad == "stale":
        prices["BTC/USDT"]["asof"] = (now - pd.Timedelta(seconds=301)).isoformat()
    elif bad == "capital":
        previous["capital"] = 100000.
    else:
        with sqlite3.connect(adapter.journal) as conn:
            conn.execute("CREATE TRIGGER refuse_audit BEFORE INSERT ON paper_records WHEN NEW.kind='reconcile' BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    Trade.commit()
    adapter.save("risk", "current", previous)
    before = records(tmp_path)
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        adapter.reset_risk(prices, now, "Investigated discrepancy")
    assert records(tmp_path) == before


def test_manual_reconciliation_preserves_same_day_loss_halt(paper_bot, tmp_path, clock):
    from h1_paper import PaperAdapter
    adapter = PaperAdapter(paper_bot, tmp_path / "journal.sqlite")
    now = clock.current
    adapter.observe(quotes(paper_bot, now), now)
    previous = adapter.get("risk", "current")
    previous.update(peak_equity=252000., day_start_equity=252000., daily_halt=True, kill_switch_active=True)
    adapter.save("risk", "current", previous)
    receipt = adapter.reset_risk(quotes(paper_bot, now), now, "Restored fresh marks")
    assert not receipt["new"]["kill_switch_active"]
    assert receipt["new"]["daily_halt"]
    assert not receipt["new"]["allow_new_exposure"]
    assert receipt["new"]["peak_equity"] == receipt["new"]["day_start_equity"] == 252000.
