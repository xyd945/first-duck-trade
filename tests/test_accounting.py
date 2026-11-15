"""Hand-calculated, synthetic spot accounting. No exchange or engine calls."""
import importlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "user_data/scripts"))


def accounting():
    assert importlib.util.find_spec("accounting"), "Marked capital accounting is missing"
    return importlib.import_module("accounting")


def marks(**prices):
    return pd.DataFrame(prices, index=pd.date_range("2020-01-01", periods=len(next(iter(prices.values()))), tz="UTC"))


def fill(day, pair, side, quantity, price):
    return {"date": pd.Timestamp(day, tz="UTC"), "pair": pair, "side": side,
            "quantity": quantity, "price": price}


def test_fixed_capital_weights_include_cash_and_open_losses():
    # 100 in A and 300 in B; A +10%, B -10%, 600 idle: -20 / 1000, not 0%.
    curve = accounting().equity_curve(
        1000., [fill("2020-01-01", "A", "buy", 1., 100.),
                fill("2020-01-01", "B", "buy", 3., 100.)],
        marks(A=[100., 110., 120.], B=[100., 90., 110.]), cost_rate=0.)
    assert curve.equity.tolist() == pytest.approx([1000., 980., 1050.])
    assert curve.cash.tolist() == pytest.approx([600., 600., 600.])
    assert curve.fixed_capital_return.tolist() == pytest.approx([0., -.02, .07])
    assert curve.net_return.tolist() == pytest.approx([0., -.02, 70. / 980.])


@pytest.mark.parametrize("fee", [.0008, .001])
@pytest.mark.parametrize("stress", [1, 2])
def test_cost_components_are_charged_on_both_sides_once(fee, stress):
    model = {"fee_rate": fee, "spread_rate": .0002, "slippage_rate": .0004,
             "stress_multiplier": stress}
    assert hasattr(accounting(), "cost_rate"), "Explicit cost components are missing"
    rate = accounting().cost_rate(model)
    assert rate == pytest.approx((fee + .0001 + .0004) * stress)
    curve = accounting().equity_curve(1000., [fill("2020-01-01", "A", "buy", 1., 100.),
        fill("2020-01-02", "A", "sell", 1., 110.)], marks(A=[105., 110.]), cost_rate=rate)
    assert curve.costs.tolist() == pytest.approx([100 * rate, 110 * rate])
    assert curve.equity.tolist() == pytest.approx([1005 - 100 * rate, 1010 - 210 * rate])


@pytest.mark.parametrize("field,value", [("fee_rate", float("nan")), ("slippage_rate", -1),
    ("spread_rate", float("inf")), ("stress_multiplier", 0), ("fee_rate", True)])
def test_invalid_costs_fail_closed(field, value):
    model = dict(fee_rate=.001, spread_rate=0., slippage_rate=.0005, stress_multiplier=1)
    model[field] = value
    with pytest.raises(ValueError):
        accounting().cost_rate(model)


@pytest.mark.parametrize("case", ["missing_mark", "negative_mark", "short", "cash", "invalid_side",
    "late_fill", "early_fill", "nan_quantity", "nan_capital", "missing_day", "duplicate_day"])
def test_invalid_paths_fail_closed(case):
    initial, prices = 1000., marks(A=[100., 110., 120.])
    fills = [fill("2020-01-01", "A", "buy", 1., 100.)]
    if case == "missing_mark": prices.iloc[1, 0] = float("nan")
    if case == "negative_mark": prices.iloc[1, 0] = -1.
    if case == "short": fills[0]["side"] = "sell"
    if case == "cash": fills[0]["quantity"] = 11.
    if case == "invalid_side": fills[0]["side"] = "other"
    if case == "late_fill": fills[0]["date"] = pd.Timestamp("2020-01-04", tz="UTC")
    if case == "early_fill": fills[0]["date"] = pd.Timestamp("2019-12-31", tz="UTC")
    if case == "nan_quantity": fills[0]["quantity"] = float("nan")
    if case == "nan_capital": initial = float("nan")
    if case == "missing_day": prices = prices.iloc[[0, 2]]
    if case == "duplicate_day": prices.index = pd.DatetimeIndex([prices.index[0]] * 3)
    with pytest.raises(ValueError):
        accounting().equity_curve(initial, fills, prices, cost_rate=.0015)


def test_split_metrics_use_marked_boundary_equity_not_a_reset_wallet():
    module = accounting()
    assert hasattr(module, "metrics"), "Split net metrics are missing"
    curve = module.equity_curve(1000., [fill("2020-01-01", "A", "buy", 1., 100.),
        fill("2020-01-03", "A", "sell", 1., 90.)], marks(A=[100., 80., 90.]), cost_rate=.003)
    result = module.metrics(curve, 1000., pd.Timestamp("2020-01-02", tz="UTC"))
    opening = 1000 - 100 * .003
    ending = 1000 - 10 - (100 + 90) * .003
    returns = pd.Series([-20 / opening, (ending - (opening - 20)) / (opening - 20)])
    assert result["net_return"] == pytest.approx(ending / opening - 1)
    assert result["net_sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * 365 ** .5)
    assert result["max_drawdown"] == pytest.approx(20 / opening)
    assert result["cost_total"] == pytest.approx(90 * .003)
    assert result["opening_equity"] == pytest.approx(opening)


def order(day, side, quantity, price):
    return {"order_filled_timestamp": int(pd.Timestamp(day, tz="UTC").timestamp() * 1000),
            "ft_order_side": side, "amount": quantity, "safe_price": price,
            "ft_is_entry": side == "buy"}


def test_freqtrade_resize_fills_exclude_only_the_terminal_force_exit():
    module = accounting()
    assert hasattr(module, "freqtrade_fills"), "Engine order accounting is missing"
    trade = {"pair": "A/USDT", "is_short": False, "leverage": 1., "exit_reason": "force_exit",
             "close_date": "2020-01-03T00:00:00Z", "orders": [
                 order("2020-01-01", "buy", 2., 100.),
                 order("2020-01-02", "sell", 1., 110.),
                 order("2020-01-03", "sell", 1., 120.)]}
    fills = module.freqtrade_fills([trade], pd.Timestamp("2020-01-03", tz="UTC"))
    assert [(f["side"], f["quantity"]) for f in fills] == [("buy", 2.), ("sell", 1.)]
    curve = module.equity_curve(1000., fills, marks(**{"A/USDT": [100., 110., 125.]}), cost_rate=.003)
    assert curve.iloc[-1].equity == pytest.approx(1000 + 10 + 25 - (200 + 110) * .003)
    assert curve.iloc[-1].gross_exposure == 125.


@pytest.mark.parametrize("case", ["early_force_exit", "short", "leverage", "missing_orders", "unfilled", "partial", "terminal_buy"])
def test_ambiguous_engine_exports_are_rejected(case):
    trade = {"pair": "A/USDT", "is_short": False, "leverage": 1., "exit_reason": "force_exit",
             "close_date": "2020-01-03T00:00:00Z", "orders": [
                 order("2020-01-01", "buy", 1., 100.), order("2020-01-03", "sell", 1., 120.)]}
    if case == "early_force_exit": trade["close_date"] = "2020-01-02T00:00:00Z"
    if case == "short": trade["is_short"] = True
    if case == "leverage": trade["leverage"] = 2.
    if case == "missing_orders": del trade["orders"]
    if case == "unfilled": trade["orders"][0]["order_filled_timestamp"] = None
    if case == "partial": trade["orders"][0]["filled"] = .5
    if case == "terminal_buy": trade["orders"][-1]["ft_order_side"] = "buy"
    with pytest.raises(ValueError):
        accounting().freqtrade_fills([trade], pd.Timestamp("2020-01-03", tz="UTC"))
