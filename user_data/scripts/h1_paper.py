"""H1-only paper execution. Freqtrade owns every simulated order and trade."""
from datetime import timezone
import math
import json
import sqlite3

import pandas as pd
from freqtrade.enums import ExitType, ExitCheckTuple

from freqtrade.persistence import Trade

from accounting import number
import paper_risk


def reconcile(bot, prices, now):
    """Reconcile SQL fills, open inventory and Freqtrade's fee-deferred wallet."""
    cfg = bot.config
    for pair in cfg["exchange"]["pair_whitelist"]:
        paper_risk.quote({"prices": prices}, pair, now)
    fee = cfg["fee"]
    initial = number(cfg["dry_run_wallet"], "initial wallet", positive=True)
    capital = number(cfg["research_h1"]["designated_portfolio_notional"], "designated notional", positive=True)
    if capital < initial:
        raise ValueError("Designated notional cannot be below the paper wallet")
    paper_risk.fresh(bot.wallets._last_wallet_refresh.isoformat(), now)
    cash, holdings, pending, positions, deferred = initial, {}, [], [], 0.
    seen = set()
    for trade in Trade.get_trades_proxy():
        if trade.pair not in cfg["exchange"]["pair_whitelist"] or trade.is_short or trade.leverage != 1:
            raise ValueError("Unexpected paper position")
        if trade.fee_open != fee or trade.fee_close != fee:
            raise ValueError("Paper fee drift")
        for order in trade.orders:
            if order.order_id in seen or order.ft_order_side not in ("buy", "sell"):
                raise ValueError("Duplicate or unexpected paper order")
            seen.add(order.order_id)
            quantity = number(order.safe_filled, "filled quantity")
            price = number(order.safe_price, "fill price", positive=True)
            side = order.ft_order_side
            cash += quantity * price * ((1 - fee) if side == "sell" else -(1 + fee))
            holdings[trade.pair] = holdings.get(trade.pair, 0.) + quantity * (1 if side == "buy" else -1)
            if order.ft_is_open and order.safe_remaining:
                pending.append({"id": order.order_id, "instance": "h1", "pair": trade.pair,
                                "side": side, "notional": order.safe_remaining * price})
        if trade.is_open:
            deferred += trade.stake_amount * fee
            positions.append({"id": str(trade.id), "instance": "h1", "pair": trade.pair,
                              "quantity": trade.amount})
    for pair in cfg["exchange"]["pair_whitelist"]:
        held = sum(p["quantity"] for p in positions if p["pair"] == pair)
        if not math.isclose(holdings.get(pair, 0.), held, abs_tol=1e-8, rel_tol=1e-9):
            raise ValueError("Unreconciled filled inventory")
        if not math.isclose(bot.wallets.get_total(pair.split("/")[0]), held, abs_tol=1e-8, rel_tol=1e-9):
            raise ValueError("Unreconciled wallet inventory")
    if cash < -1e-6 or not math.isclose(bot.wallets.get_total("USDT"), cash + deferred, abs_tol=1e-6, rel_tol=1e-9):
        raise ValueError("Unreconciled settled cash")
    snapshot = {"asof": now.isoformat(), "quote_currency": "USDT", "prices": prices,
                "accounts": [{"id": "h1-dry-run", "mode": "spot", "cash": max(0., cash),
                              "instances": ["h1"], "positions": positions, "pending": pending},
                             {"id": "designated-paper-reserve", "mode": "spot", "cash": capital - initial,
                              "instances": [], "positions": [], "pending": []}]}
    paper_risk.mark_account(snapshot, ["h1"], now)
    return snapshot


def decision_rows(strategy, data, day):
    """Use confirmed history only, with the paper preload and complete universe."""
    prefix = {p: df.loc[df.date < day].copy() for p, df in data.items()}
    clipped = {p: df.tail(strategy.rules["paper_startup"]).copy() for p, df in prefix.items()}
    strategy._bars_before = {p: int((df.close.gt(0) & df.date.lt(clipped[p].date.min())).sum())
                             for p, df in prefix.items()}
    analysed = strategy.advise_all_indicators(clipped)
    result = {}
    for pair, frame in analysed.items():
        row = frame.iloc[-1].to_dict()
        if row["date"] != day - pd.Timedelta(days=1) or not row["can_trade"]:
            row.update(can_trade=False, rebalance=False, anomaly=True)
            row["close"] = frame.close.where(frame.can_trade).ffill().iloc[-1]
        result[pair] = row
    return result


class PaperAdapter:
    def __init__(self, bot, journal):
        self.bot, self.journal = bot, journal
        with sqlite3.connect(journal) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS paper_records (kind TEXT, id TEXT, payload TEXT NOT NULL, PRIMARY KEY(kind,id))")

    def get(self, kind, key):
        with sqlite3.connect(self.journal) as conn:
            row = conn.execute("SELECT payload FROM paper_records WHERE kind=? AND id=?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, kind, key, value):
        with sqlite3.connect(self.journal) as conn:
            conn.execute("INSERT OR REPLACE INTO paper_records VALUES (?,?,?)",
                         (kind, key, json.dumps(value, allow_nan=False, default=str)))

    def observe(self, prices, now):
        state = self.get("risk", "current")
        snapshot = None
        try:
            self.bot.wallets.update()
            now = max(now, self.bot.wallets._last_wallet_refresh)
            snapshot = reconcile(self.bot, prices, now)
            if state is None and not Trade.get_trades_proxy():
                capital = self.bot.config["research_h1"]["designated_portfolio_notional"]
                state = dict(capital=capital, peak_equity=capital, last_equity=capital,
                             day_start_equity=capital, day=now.date().isoformat(), asof=now.isoformat(),
                             kill_switch_active=False, daily_halt=False)
            state = paper_risk.evaluate(snapshot, state, ["h1"], now)
        except (ValueError, KeyError, TypeError) as exc:
            state = paper_risk.evaluate(None, state, ["h1"], now)
            state["reason"] = f"Reconciliation rejected: {exc}"
        self.save("risk", "current", state)
        self.save("snapshot", "current", snapshot)
        return snapshot, state

    def settle(self):
        for trade in Trade.get_open_trades():
            for order in list(trade.open_orders):
                result = self.bot.exchange.fetch_order(order.order_id, trade.pair)
                empty = self.bot.update_trade_state(trade, order.order_id, result)
                if empty:
                    self.bot.handle_cancel_order(result, order, trade, "H1 empty order")
        Trade.commit()

    def expire(self, day):
        with sqlite3.connect(self.journal) as conn:
            previous = [json.loads(row[0]) for row in conn.execute(
                "SELECT payload FROM paper_records WHERE kind='batch' AND id < ?", (day.isoformat(),))]
        for batch in previous:
            if batch["status"] not in ("selling", "buying", "waiting"):
                continue
            for trade in Trade.get_open_trades():
                for order in list(trade.open_orders):
                    result = self.bot.exchange.fetch_order(order.order_id, trade.pair)
                    self.bot.handle_cancel_order(result, order, trade, "H1 daily batch expired")
            if any(t.open_orders for t in Trade.get_open_trades()):
                raise ValueError("Expired orders remain open; no new batch")
            holdings = {t.pair: t.amount for t in Trade.get_open_trades()}
            batch.update(status="expired", divergence=True,
                         unfilled={p: q - holdings.get(p, 0.) for p, q in batch["targets"].items()})
            self.save("batch", batch["time"], batch)

    def step(self, rows, prices, now):
        day = pd.Timestamp(now).floor("D")
        key = day.isoformat()
        buy_prices = {p: number(q.get("buy_price", q["price"]), "buy price", positive=True) for p, q in prices.items()}
        sell_prices = {p: number(q.get("sell_price", q["price"]), "sell price", positive=True) for p, q in prices.items()}
        self.expire(day)
        self.settle()
        snapshot, state = self.observe(prices, now)
        batch = self.get("batch", key)
        if batch and batch["status"] == "finished":
            return batch
        if not snapshot:
            batch = batch or dict(time=key, targets={}, requests={}, orders={}, attempted=[],
                                  target_weights={p: float(r["target_weight"]) if pd.notna(r["target_weight"]) else None
                                                  for p, r in rows.items()})
            batch.update(status="blocked", risk_reason=state["reason"], divergence=True)
            self.save("batch", key, batch)
            return batch
        account = snapshot["accounts"][0]
        holdings = {p["pair"]: p["quantity"] for p in account["positions"]}
        if batch is None or batch["status"] in ("blocked", "waiting_data"):
            batch = self.bot.strategy.plan_batch(day, rows, holdings, account["cash"])
            batch.update(time=key, status="selling", attempted=[],
                         target_weights={p: float(r["target_weight"]) if pd.notna(r["target_weight"]) else None
                                         for p, r in rows.items()})
            self.save("batch", key, batch)
        self.bot.strategy.batch = {**batch, "time": pd.Timestamp(batch["time"])}
        if batch["status"] == "selling":
            for pair in sorted(p for p, q in batch["requests"].items() if q < -1e-10):
                if pair not in batch["attempted"]:
                    batch["attempted"].append(pair)
                    self.save("batch", key, batch)
                    trade = next(t for t in Trade.get_open_trades() if t.pair == pair)
                    self.bot.execute_trade_exit(trade, sell_prices[pair],
                        ExitCheckTuple(ExitType.PARTIAL_EXIT, "h1_resize"),
                        sub_trade_amt=min(trade.amount, -batch["requests"][pair]))
            self.settle()
            snapshot, state = self.observe(prices, now)
            if not snapshot or any(o["side"] == "sell" for o in snapshot["accounts"][0]["pending"]):
                batch["risk_reason"] = "waiting for settled sells" if snapshot else state["reason"]
                self.save("batch", key, batch)
                return batch
            now = pd.Timestamp(state["asof"]).to_pydatetime()
            self.bot.strategy.fund_batch(now, snapshot["accounts"][0]["cash"],
                buy_prices,
                risk={"snapshot": snapshot, "state": state, "expected_instances": ["h1"], "instance": "h1"})
            batch.update({k: v for k, v in self.bot.strategy.batch.items() if k != "time"})
            batch["status"] = "buying"
            self.save("batch", key, batch)
        now = pd.Timestamp(state["asof"]).to_pydatetime()
        proposals = [{"instance": "h1", "pair": p, "notional": q * buy_prices[p]}
                     for p, q in batch["orders"].items() if q > 1e-10 and p not in batch["attempted"]]
        verdict = paper_risk.evaluate(snapshot, state, ["h1"], now, proposals=proposals)
        if not verdict["allow_new_exposure"]:
            batch["risk_reason"] = verdict["reason"]
            batch["orders"] = {p: min(q, 0.) for p, q in batch["orders"].items()}
            self.bot.strategy.batch["orders"] = batch["orders"]
            self.save("batch", key, batch)
        for pair in sorted(p for p, q in batch["orders"].items() if q > 1e-10):
            if pair in batch["attempted"]:
                continue
            batch["attempted"].append(pair)
            self.save("batch", key, batch)
            trade = next((t for t in Trade.get_open_trades() if t.pair == pair), None)
            self.bot.execute_entry(pair, batch["orders"][pair] * buy_prices[pair],
                                   price=buy_prices[pair], trade=trade,
                                   mode="pos_adjust" if trade else "initial", enter_tag="h1_batch")
        self.settle()
        snapshot, state = self.observe(prices, now)
        if snapshot:
            account = snapshot["accounts"][0]
            holdings = {p["pair"]: p["quantity"] for p in account["positions"]}
            batch.update(cash=account["cash"], holdings=holdings,
                         unfilled={p: q - holdings.get(p, 0.) for p, q in batch["targets"].items()},
                         filled_weights={p: holdings.get(p, 0.) * rows[p]["close"] / batch["equity"] for p in rows},
                         status="waiting" if account["pending"] else "finished")
            batch["divergence"] = any(abs(q) > 1e-8 for q in batch["unfilled"].values())
        else:
            batch.update(status="waiting", risk_reason=state["reason"], divergence=True)
        self.save("batch", key, batch)
        return batch
