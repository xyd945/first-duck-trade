"""Synthetic decision batches; no exchange connection or market data."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("freqtrade")
from tests.test_h1 import config
from H1Trend import H1EqualWeight, H1Trend


PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
NOW = datetime(2021, 11, 1, tzinfo=timezone.utc)


def rows(values, *, changed=False):
    return {pair: {"close": 100., "target_weight": value, "rebalance": True,
                   "signal_change": changed, "can_trade": True, "anomaly": False}
            for pair, value in zip(PAIRS, values)}


def test_monday_band_keeps_financing_sales_and_scales_buy_to_zero():
    strategy = H1EqualWeight(config())
    # NAV 10,000. Buy 400; sales 130/130/140 are each below the 250 band.
    batch = strategy.plan_batch(NOW, rows([.25] * 4), dict(zip(PAIRS, [21, 26.3, 26.3, 26.4])), 0)
    assert batch["requests"] == pytest.approx({PAIRS[0]: 4})
    strategy.fund_batch(NOW, 0, {p: 100 for p in PAIRS})
    assert strategy.batch["orders"][PAIRS[0]] == 0
    assert strategy.batch["cash_scale"] == 0


def test_signal_change_ignores_band_and_cost_scales_buys_equally():
    strategy = H1Trend(config())
    strategy.plan_batch(NOW, rows([.25] * 4, changed=True), dict(zip(PAIRS, [21, 26.3, 26.3, 26.4])), 0)
    assert strategy.batch["requests"] == pytest.approx(dict(zip(PAIRS, [4, -1.3, -1.3, -1.4])))
    strategy.plan_batch(NOW, rows([.5, .5], changed=True), {}, 10000)
    strategy.fund_batch(NOW, 10000, {p: 100 for p in PAIRS})
    # Two 5,000 buys each pay 0.3%; 10,000 / 1.003 is all affordable notional.
    assert strategy.batch["orders"][PAIRS[0]] == pytest.approx(50 / 1.003)
    assert strategy.batch["orders"][PAIRS[1]] == pytest.approx(50 / 1.003)


def test_strategy_callbacks_use_batch_and_expire_at_next_decision():
    strategy = H1EqualWeight(config())
    strategy.plan_batch(NOW, rows([.5, .5]), {PAIRS[0]: 40, PAIRS[1]: 60}, 0)
    trade = SimpleNamespace(pair=PAIRS[1], amount=60, stake_amount=3000)
    # Sell 10 units; Freqtrade negative stake is in original-cost units: -10*3000/60.
    assert strategy.adjust_trade_position(trade, NOW, 100, 0, 1, 10000) == pytest.approx(-500)
    strategy.fund_batch(NOW, 997, {p: 100 for p in PAIRS})
    amount = strategy.custom_stake_amount(PAIRS[0], NOW, 100, 9999, 1, 10000, 1, None, "long")
    assert amount == pytest.approx(997 / 1.003)
    assert strategy.custom_stake_amount(PAIRS[0], NOW.replace(day=2), 100, 9999, 1, 10000, 1, None, "long") == 0
    assert not strategy.confirm_trade_entry(PAIRS[0], "market", 100, 100, "GTC", NOW, None, "long")


def test_pending_sales_and_late_missing_context_cannot_fund_paper_buys():
    from freqtrade.enums import RunMode
    cfg = config()
    cfg["runmode"] = RunMode.DRY_RUN
    strategy = H1Trend(cfg)
    strategy.plan_batch(NOW, rows([.25], changed=True), {}, 10000)
    ctx = risk_context()
    strategy.fund_batch(NOW, 10000, {p: 100 for p in PAIRS}, risk=ctx)
    assert strategy.batch["orders"][PAIRS[0]] > 0
    strategy.fund_batch(NOW, 10000, {p: 100 for p in PAIRS}, risk={})
    assert strategy.batch["orders"][PAIRS[0]] == 0
    ctx["snapshot"]["accounts"][0]["pending"] = [
        {"id": "sell", "instance": "h1", "pair": PAIRS[1], "side": "sell", "notional": 100}]
    strategy.fund_batch(NOW, 10000, {p: 100 for p in PAIRS}, risk=ctx)
    assert strategy.batch["orders"][PAIRS[0]] == 0


def test_paper_batch_can_finish_after_sells_settle_before_next_day():
    from freqtrade.enums import RunMode
    cfg = config()
    cfg["runmode"] = RunMode.DRY_RUN
    strategy = H1Trend(cfg)
    strategy.plan_batch(NOW, rows([.25], changed=True), {}, 10000)
    later = NOW.replace(minute=1)
    strategy.fund_batch(later, 10000, {p: 100 for p in PAIRS}, risk=risk_context())
    assert strategy.custom_stake_amount(PAIRS[0], later, 100, 9999, 1, 10000, 1, None, "long") > 0


def risk_context(cash=10000, holdings=None):
    stamp = NOW.isoformat()
    holdings = holdings or {}
    snapshot = {"asof": stamp, "quote_currency": "USDT", "prices": {
        pair: {"price": 100., "asof": stamp} for pair in PAIRS}, "accounts": [
        {"id": "h1", "mode": "spot", "cash": cash, "instances": ["h1"],
         "positions": [{"id": p, "instance": "h1", "pair": p, "quantity": q} for p, q in holdings.items()], "pending": []},
        {"id": "reserve", "mode": "spot", "cash": 240000., "instances": [], "positions": [], "pending": []}]}
    total = 240000 + cash + sum(holdings.values()) * 100
    state = dict(asof=stamp, day=NOW.date().isoformat(), capital=250000., peak_equity=max(250000., total),
                 day_start_equity=250000., last_equity=total, daily_halt=False, kill_switch_active=False)
    return {"snapshot": snapshot, "state": state, "expected_instances": ["h1"], "instance": "h1"}


@pytest.mark.parametrize("name", ["H1Trend", "H1SimpleTrend", "H1EqualWeight", "H1BuyHold"])
def test_all_strategy_paper_buys_obey_fix4_coin_cap(name):
    import H1Trend as module
    from freqtrade.enums import RunMode
    cfg = config()
    cfg["runmode"] = RunMode.DRY_RUN
    strategy = getattr(module, name)(cfg)
    strategy.plan_batch(NOW, rows([1.0], changed=True), {}, 10000)
    strategy.fund_batch(NOW, 10000, {p: 100 for p in PAIRS}, risk=risk_context())
    amount = strategy.batch["orders"][PAIRS[0]] * 100
    assert amount <= 2500
    assert amount == pytest.approx(2500, abs=1e-6)
    assert strategy.batch["risk_scale"] == pytest.approx(.25075)


@pytest.mark.parametrize("fault", ["stale", "missing", "daily", "drawdown", "pending", "gross"])
def test_fail_closed_paper_gate_preserves_reductions(fault):
    from freqtrade.enums import RunMode
    cfg = config()
    cfg["runmode"] = RunMode.DRY_RUN
    strategy = H1Trend(cfg)
    strategy.plan_batch(NOW, rows([0, .5], changed=True), {PAIRS[0]: 10}, 9000)
    ctx = risk_context(9000, {PAIRS[0]: 10})
    if fault == "stale":
        ctx["snapshot"]["asof"] = "2021-10-30T00:00:00+00:00"
    elif fault == "missing":
        ctx = None
    elif fault == "daily":
        ctx["state"]["daily_halt"] = True
    elif fault == "drawdown":
        ctx["state"]["peak_equity"] = 300000
    elif fault == "pending":
        ctx["snapshot"]["accounts"][0]["pending"] = [{"id": "pending", "instance": "h1", "pair": PAIRS[1], "side": "buy", "notional": 2500}]
    elif fault == "gross":
        ctx["snapshot"]["accounts"][0]["positions"][0]["quantity"] = 200
    strategy.fund_batch(NOW, 9000, {p: 100 for p in PAIRS}, risk=ctx)
    assert strategy.batch["orders"][PAIRS[1]] == pytest.approx(0, abs=1e-8)
    assert strategy.batch["orders"][PAIRS[0]] == -10
