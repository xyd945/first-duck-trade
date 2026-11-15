"""Fixed initial capital, cash plus spot inventory marked at each daily close."""
import math

import pandas as pd


def number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Invalid {name}")
    return value


def cost_rate(model):
    """One-side cost on reference notional; spread_rate is the full quoted spread."""
    for key in ("fee_rate", "spread_rate", "slippage_rate", "stress_multiplier"):
        number(model.get(key), key, positive=key == "stress_multiplier")
    rate = (model["fee_rate"] + model["spread_rate"] / 2 + model["slippage_rate"]) * model["stress_multiplier"]
    if rate >= 1:
        raise ValueError("Cost rate must be below one")
    return rate


def equity_curve(initial_capital, fills, marks, *, cost_rate):
    """Replay actual quantities; never average closed-trade percentage profits."""
    number(initial_capital, "initial capital", positive=True)
    number(cost_rate, "cost rate")
    if cost_rate >= 1 or marks.empty or not isinstance(marks.index, pd.DatetimeIndex):
        raise ValueError("Invalid marks or costs")
    expected = pd.date_range(marks.index[0], marks.index[-1], freq="D", tz="UTC")
    if not marks.index.equals(expected) or not marks.index.equals(marks.index.normalize()):
        raise ValueError("Marks must cover consecutive UTC days")
    for event in fills:
        number(event["quantity"], "fill quantity", positive=True)
        number(event["price"], "fill price", positive=True)
        if (event["side"] not in ("buy", "sell") or event["pair"] not in marks or
                not marks.index[0] <= event["date"] < marks.index[-1] + pd.Timedelta(days=1)):
            raise ValueError("Invalid fill side, pair or date")
    cash = previous = initial_capital
    positions = dict.fromkeys(marks.columns, 0.)
    pending = iter(sorted(fills, key=lambda f: f["date"]))
    event = next(pending, None)
    rows = []
    for day, prices in marks.iterrows():
        costs = 0.
        while event is not None and event["date"] < day + pd.Timedelta(days=1):
            quantity, price = event["quantity"], event["price"]
            sign = 1 if event["side"] == "buy" else -1
            cost = quantity * price * cost_rate
            cash -= sign * quantity * price + cost
            costs += cost
            positions[event["pair"]] += sign * quantity
            if positions[event["pair"]] < -1e-9 or cash < -1e-8:
                raise ValueError("Short position or insufficient cash")
            event = next(pending, None)
        for pair, quantity in positions.items():
            if quantity > 1e-9:
                number(float(prices[pair]), "position mark", positive=True)
        gross = sum(quantity * prices[pair] for pair, quantity in positions.items() if quantity)
        equity = cash + gross
        number(equity, "equity", positive=True)
        rows.append({"date": day, "cash": cash, "gross_exposure": gross, "costs": costs,
                     "equity": equity, "net_return": equity / previous - 1,
                     "fixed_capital_return": (equity - previous) / initial_capital})
        previous = equity
    return pd.DataFrame(rows).set_index("date")


def freqtrade_fills(trades, last_day):
    """Backtest minified orders are full fills; terminal force_exit is not a sale."""
    fills = []
    for trade in trades:
        orders = trade.get("orders")
        if not orders or trade.get("is_short") is not False or trade.get("leverage") != 1:
            raise ValueError("Only complete long spot order exports are supported")
        terminal = trade["exit_reason"] == "force_exit"
        if terminal and (pd.to_datetime(trade["close_date"], utc=True) != last_day or
                         orders[-1].get("ft_order_side") != "sell"):
            raise ValueError("Ambiguous synthetic force_exit")
        for index, order in enumerate(orders):
            number(order.get("order_filled_timestamp"), "fill timestamp", positive=True)
            number(order.get("amount"), "fill amount", positive=True)
            number(order.get("safe_price"), "fill price", positive=True)
            side = order.get("ft_order_side")
            if (side not in ("buy", "sell") or order.get("ft_is_entry") != (side == "buy") or
                    order.get("filled", order["amount"]) != order["amount"]):
                raise ValueError("Unsupported partial or invalid order export")
            if terminal and index == len(orders) - 1:
                if pd.to_datetime(order["order_filled_timestamp"], unit="ms", utc=True) != last_day:
                    raise ValueError("Synthetic close timestamp mismatch")
                continue
            fills.append({"date": pd.to_datetime(order["order_filled_timestamp"], unit="ms", utc=True),
                          "pair": trade["pair"], "side": order["ft_order_side"],
                          "quantity": order["amount"], "price": order["safe_price"]})
    return fills


def reconcile_wallet(wallet, fills, pairs, left, right):
    """Cross-check pre-bar inventory; Freqtrade cash defers open-trade entry fees."""
    currencies = {pair.split("/")[0]: pair for pair in pairs}
    if wallet.duplicated(["date", "currency"]).any() or not set(wallet.currency) <= {*currencies, "USDT"}:
        raise ValueError("Duplicate or unknown wallet currency")
    for row in wallet.itertuples():
        number(float(row.balance), "wallet balance")
        number(float(row.rate), "wallet rate", positive=True)
    dates = pd.DatetimeIndex(sorted(wallet.date.unique()))
    if (dates.empty or dates[0] not in (left, left + pd.Timedelta(days=1)) or
            not dates.equals(pd.date_range(dates[0], right - pd.Timedelta(days=1), freq="D"))):
        raise ValueError("Incomplete wallet path")
    events = iter(sorted(fills, key=lambda fill: fill["date"]))
    event = next(events, None)
    held = dict.fromkeys(pairs, 0.)
    for day, rows in wallet.sort_values("date").groupby("date"):
        while event is not None and event["date"] < day:
            held[event["pair"]] += event["quantity"] * (1 if event["side"] == "buy" else -1)
            event = next(events, None)
        balances = rows.set_index("currency").balance
        for currency, pair in currencies.items():
            if not math.isclose(held[pair], balances.get(currency, 0.), rel_tol=1e-8, abs_tol=1e-9):
                raise ValueError("Orders do not reconcile with wallet inventory")


def metrics(curve, initial_capital, start):
    previous = curve.loc[curve.index < start, "equity"]
    opening = float(previous.iloc[-1]) if not previous.empty else initial_capital
    scored = curve.loc[curve.index >= start]
    if scored.empty:
        raise ValueError("Empty scoring window")
    returns = scored.net_return
    deviation = returns.std(ddof=1)
    sharpe = float(returns.mean() / deviation * math.sqrt(365)) if deviation > 0 else None
    equity = pd.Series([opening, *scored.equity.tolist()])
    return {"net_return": float(equity.iloc[-1] / opening - 1), "net_sharpe": sharpe,
            "max_drawdown": float((1 - equity / equity.cummax()).max()),
            "opening_equity": opening, "ending_equity": float(equity.iloc[-1]),
            "cost_total": float(scored.costs.sum())}
