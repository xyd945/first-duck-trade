"""Synthetic portfolio snapshots; the risk gate never sends orders."""
import importlib
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "user_data/scripts"))
NOW = datetime(2020, 1, 2, 12, tzinfo=timezone.utc)


def risk():
    assert importlib.util.find_spec("paper_risk"), "Account-level risk gate is missing"
    return importlib.import_module("paper_risk")


def snapshot():
    return {"asof": NOW.isoformat(), "quote_currency": "USDT",
            "prices": {"BTC/USDT": {"price": 100., "asof": NOW.isoformat()},
                       "ETH/USDT": {"price": 20., "asof": NOW.isoformat()}},
            "accounts": [{"id": "shared", "mode": "spot", "cash": 9700., "instances": ["sweep", "dynamic"],
                          "positions": [
                              {"id": "one", "instance": "sweep", "pair": "BTC/USDT", "quantity": 1.},
                              {"id": "two", "instance": "dynamic", "pair": "ETH/USDT", "quantity": 10.}],
                          "pending": []}]}


def test_account_equity_counts_shared_cash_once_and_every_instance():
    marked = risk().mark_account(snapshot(), {"sweep", "dynamic"}, NOW)
    assert marked["equity"] == 10000.
    assert marked["gross_exposure"] == 300.
    assert marked["positions"] == {"BTC/USDT": 100., "ETH/USDT": 200.}
    data = snapshot()
    data["prices"]["ETH/USDT"]["price"] = 10.
    assert risk().mark_account(data, {"sweep", "dynamic"}, NOW)["equity"] == 9900.


def state():
    return {"capital": 10000., "peak_equity": 10000., "day_start_equity": 10000.,
            "last_equity": 10000., "day": "2020-01-02", "asof": NOW.isoformat(),
            "kill_switch_active": False, "daily_halt": False}


@pytest.mark.parametrize("case", ["positions_missing", "stale_price", "future_price", "nan_price", "unknown_instance",
    "missing_instance", "duplicate_account", "duplicate_position", "negative_quantity", "missing_cash", "stale_snapshot",
    "missing_state", "stale_state", "leverage", "unknown_pair", "pending_missing"])
def test_missing_or_untrusted_risk_inputs_latch_kill_switch(case):
    module = risk()
    assert hasattr(module, "evaluate"), "Fail-closed kill switch is missing"
    data, previous = snapshot(), state()
    account = data["accounts"][0]
    if case == "positions_missing": del account["positions"]
    if case == "stale_price": data["prices"]["BTC/USDT"]["asof"] = "2020-01-02T11:54:59+00:00"
    if case == "future_price": data["prices"]["BTC/USDT"]["asof"] = "2020-01-02T12:00:01+00:00"
    if case == "nan_price": data["prices"]["BTC/USDT"]["price"] = float("nan")
    if case == "unknown_instance": account["positions"][0]["instance"] = "hidden"
    if case == "missing_instance": account["instances"].remove("dynamic")
    if case == "duplicate_account": data["accounts"].append(account.copy())
    if case == "duplicate_position": account["positions"].append(account["positions"][0].copy())
    if case == "negative_quantity": account["positions"][0]["quantity"] = -1
    if case == "missing_cash": del account["cash"]
    if case == "stale_snapshot": data["asof"] = "2020-01-01T12:00:00+00:00"
    if case == "missing_state": previous = {}
    if case == "stale_state": previous["asof"] = "2020-01-01T12:00:00+00:00"
    if case == "leverage": account["positions"][0]["leverage"] = 2
    if case == "unknown_pair": account["positions"][0]["pair"] = "BTC/USDT:USDT"
    if case == "pending_missing": del account["pending"]
    result = module.evaluate(data, previous, {"sweep", "dynamic"}, NOW)
    assert not result["allow_new_exposure"]
    assert result["kill_switch_active"]
    assert result["reason"]
    recovered = module.evaluate(snapshot(), result, {"sweep", "dynamic"}, NOW)
    assert not recovered["allow_new_exposure"]  # reconciliation failure needs manual reset


def test_scheduler_jitter_does_not_make_a_fresh_snapshot_unusable():
    previous = state()
    previous["asof"] = "2020-01-02T11:54:50+00:00"
    result = risk().evaluate(snapshot(), previous, {"sweep", "dynamic"}, NOW)
    assert result["allow_new_exposure"]


def test_daily_loss_halts_new_exposure_at_exact_limit_and_latches_for_day():
    data = snapshot()
    data["accounts"][0]["cash"] -= 50.
    blocked = risk().evaluate(data, state(), {"sweep", "dynamic"}, NOW)
    assert not blocked["allow_new_exposure"]
    assert blocked["daily_loss"] == pytest.approx(.005)
    assert blocked["daily_halt"]
    assert not blocked["kill_switch_active"]
    assert not risk().evaluate(snapshot(), blocked, {"sweep", "dynamic"}, NOW)["allow_new_exposure"]


def test_drawdown_uses_high_water_mark_even_when_total_pnl_is_positive():
    previous = state()
    previous["peak_equity"] = 11000.
    data = snapshot()
    data["accounts"][0]["cash"] += 450.
    blocked = risk().evaluate(data, previous, {"sweep", "dynamic"}, NOW)
    assert not blocked["allow_new_exposure"]
    assert blocked["drawdown"] == pytest.approx(.05)
    assert blocked["kill_switch_active"]
    assert blocked["total_pnl"] == 450.


def test_daily_reset_uses_last_equity_and_never_resets_high_water_mark():
    previous = state()
    previous.update(asof="2020-01-01T23:59:00+00:00", day="2020-01-01", daily_halt=True,
                    last_equity=9970., peak_equity=10100.)
    now = datetime(2020, 1, 2, tzinfo=timezone.utc)
    data = snapshot()
    data["asof"] = now.isoformat()
    for price in data["prices"].values(): price["asof"] = now.isoformat()
    result = risk().evaluate(data, previous, {"sweep", "dynamic"}, now)
    assert result["allow_new_exposure"]
    assert result["day_start_equity"] == 9970.
    assert result["peak_equity"] == 10100.
    assert not result["daily_halt"]


@pytest.mark.parametrize("amounts,allowed", [([100.], True), ([100.01], False), ([60., 60.], False)])
def test_new_position_limit_is_aggregate_not_per_order(amounts, allowed):
    data = snapshot()
    data["prices"]["SOL/USDT"] = {"price": 10., "asof": NOW.isoformat()}
    proposals = [{"instance": "dynamic", "pair": "SOL/USDT", "notional": amount} for amount in amounts]
    result = risk().evaluate(data, state(), {"sweep", "dynamic"}, NOW, proposals=proposals)
    assert result["allow_new_exposure"] is allowed


def test_pending_buys_use_gross_capacity_and_pending_sells_do_not_release_it():
    data = snapshot()
    data["accounts"][0]["pending"] = [
        {"id": "buy", "instance": "dynamic", "pair": "BTC/USDT", "side": "buy", "notional": 201.},
        {"id": "sell", "instance": "sweep", "pair": "BTC/USDT", "side": "sell", "notional": 100.}]
    result = risk().evaluate(data, state(), {"sweep", "dynamic"}, NOW)
    assert not result["allow_new_exposure"]
    assert result["projected_gross"] == 501.
    assert result["reason"] == "gross_limit"


def test_same_coin_across_instances_cannot_evade_position_limit():
    proposal = [{"instance": "dynamic", "pair": "BTC/USDT", "notional": 1.}]
    result = risk().evaluate(snapshot(), state(), {"sweep", "dynamic"}, NOW, proposals=proposal)
    assert not result["allow_new_exposure"]
    assert result["reason"] == "position_limit"


@pytest.mark.parametrize("bad", [None, [], "broken"])
def test_non_object_state_fails_closed(bad):
    result = risk().evaluate(snapshot(), bad, {"sweep", "dynamic"}, NOW)
    assert result["kill_switch_active"] and not result["allow_new_exposure"]


def test_pending_buy_cannot_exceed_position_limit_without_a_new_proposal():
    data = snapshot()
    data["accounts"][0]["pending"] = [
        {"id": "p", "instance": "dynamic", "pair": "BTC/USDT", "side": "buy", "notional": 1.}]
    result = risk().evaluate(data, state(), {"sweep", "dynamic"}, NOW)
    assert not result["allow_new_exposure"]
    assert result["reason"] == "position_limit"


def test_latched_kill_keeps_marking_equity_without_reopening_exposure():
    previous = state()
    previous["kill_switch_active"] = True
    data = snapshot()
    data["accounts"][0]["cash"] -= 20.
    result = risk().evaluate(data, previous, {"sweep", "dynamic"}, NOW)
    assert result.get("equity") == 9980.
    assert result["kill_switch_active"] and not result["allow_new_exposure"]


@pytest.mark.parametrize("case", ["missing_proposal_price", "nan_proposal_price", "unknown_proposal_owner",
    "nan_proposal_notional", "derivative_proposal", "derivative_account", "missing_mode", "state_day_mismatch",
    "invalid_halt_flag", "invalid_peak", "nan_pending_price", "derivative_pending"])
def test_late_validation_failures_never_return_an_allow_decision(case):
    data, previous = snapshot(), state()
    data["prices"]["SOL/USDT"] = {"price": 10., "asof": NOW.isoformat()}
    proposal = {"instance": "dynamic", "pair": "SOL/USDT", "notional": 10.}
    if case == "missing_proposal_price": del data["prices"]["SOL/USDT"]
    if case == "nan_proposal_price": data["prices"]["SOL/USDT"]["price"] = float("nan")
    if case == "unknown_proposal_owner": proposal["instance"] = "hidden"
    if case == "nan_proposal_notional": proposal["notional"] = float("nan")
    if case == "derivative_proposal":
        proposal["pair"] = "SOL/USDT:USDT"
        data["prices"]["SOL/USDT:USDT"] = data["prices"]["SOL/USDT"]
    if case == "derivative_account": data["accounts"][0]["mode"] = "futures"
    if case == "missing_mode": del data["accounts"][0]["mode"]
    if case == "state_day_mismatch": previous["day"] = "1999-01-01"
    if case == "invalid_halt_flag": previous["kill_switch_active"] = None
    if case == "invalid_peak": previous["peak_equity"] = 9000.
    if case in ("nan_pending_price", "derivative_pending"):
        pair = "X/USDT" if case == "nan_pending_price" else "X/USDT:USDT"
        data["prices"][pair] = {"price": float("nan") if case == "nan_pending_price" else 1., "asof": NOW.isoformat()}
        data["accounts"][0]["pending"] = [{"id": "p", "instance": "dynamic", "pair": pair, "side": "buy", "notional": 1.}]
    result = risk().evaluate(data, previous, {"sweep", "dynamic"}, NOW, proposals=[proposal])
    assert result["kill_switch_active"]
    assert not result["allow_new_exposure"]
