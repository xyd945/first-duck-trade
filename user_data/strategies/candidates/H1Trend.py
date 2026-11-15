"""G0 H1: causal daily spot trend rules and the registered baselines."""
import numpy as np
import pandas as pd


def grid_params(policy):
    from accounting import cost_rate
    return {f"{name}{target * 100:g}": {**policy["params"], "ladder": ladder,
            "sigma_target": target, "cost_rate": cost_rate(policy["costs"])}
            for name, ladder in policy["h1"]["grid"]["ladders"].items()
            for target in policy["h1"]["grid"]["sigma_targets"]}


def trend_frame(frame, ladder, vol_window, gap_days=3, annualisation_days=365):
    data = frame.set_index("date").copy()
    if data.empty or data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Expected nonempty, unique, ordered daily candles")
    data = data.reindex(pd.date_range(data.index[0], data.index[-1], freq="D", name="date"))
    confirmed = data["confirm"].astype(str).eq("1") if "confirm" in data else data["close"].notna()
    close = data["close"].where(confirmed & data["close"].gt(0))
    data["can_trade"] = close.notna()
    references = pd.concat([close.ffill(limit=gap_days).shift(days) for days in ladder], axis=1)
    valid = references.notna().all(axis=1) & close.notna()
    data["score"] = references.lt(close, axis=0).mean(axis=1).where(valid).ffill()
    data["anomaly"] = references.isna().any(axis=1) & (np.arange(len(data)) >= max(ladder))
    data["sigma"] = np.log(close / close.shift(1)).rolling(vol_window).std(ddof=1) * np.sqrt(annualisation_days)
    return data.reset_index()


from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy


class H1Trend(IStrategy):
    timeframe = "1d"
    can_short = False
    minimal_roi = {}
    stoploss = -1.0
    position_adjustment_enable = True
    use_exit_signal = False
    mode = "trend"

    def __init__(self, config):
        super().__init__(config)
        self.params = config["research_params"]
        self.rules = config["research_h1"]
        from accounting import number
        for key in ("sigma_target", "rebalance_band", "cost_rate"):
            number(self.params[key], key, positive=(key == "sigma_target"))
        if self.params["cost_rate"] >= 1 or self.params["rebalance_band"] > 1:
            raise ValueError("Invalid cost or rebalance band")
        periods = [*self.params["ladder"], self.params["vol_window"], self.rules["minimum_bars"],
                   self.rules["sma_fast"], self.rules["sma_slow"], self.rules["lookback_gap_days"]]
        if len(self.params["ladder"]) != 3 or any(type(v) is not int or v < 1 for v in periods):
            raise ValueError("Periods must be positive integers; a ladder has three votes")
        self.startup_candle_count = self.rules[
            "historical_startup" if config.get("runmode") == RunMode.BACKTEST else "paper_startup"]

    def bot_start(self, **kwargs):
        if self.config.get("runmode") == RunMode.DRY_RUN and self.config.get("dry_run") is True:
            from h1_service import PAPER_TOKEN
            if self.config.get("_h1_paper_adapter") is PAPER_TOKEN:
                return
        if self.config.get("runmode") != RunMode.BACKTEST or not getattr(self, "_h1_adapter", False):
            raise RuntimeError("H1 rebalance execution contract requires the logged historical adapter or verified paper service")

    def _eligible(self, dataframe, pair):
        confirmed = dataframe.close.notna() & dataframe.close.gt(0)
        if "confirm" in dataframe:
            confirmed &= dataframe["confirm"].astype(str).eq("1")
        count = confirmed.cumsum() + getattr(self, "_bars_before", {}).get(pair, 0)
        decision = dataframe.date + pd.Timedelta(days=1)
        return count.ge(self.rules["minimum_bars"]) & decision.ge(pd.Timestamp(self.rules["eligibility"][pair], tz="UTC"))

    def advise_all_indicators(self, data):
        if set(data) != set(self.rules["eligibility"]):
            raise ValueError("The complete frozen universe is required")
        series = [pd.Series(self._eligible(df, pair).to_numpy(), index=df.date, name=pair)
                  for pair, df in data.items()]
        self._eligible_count = pd.concat(series, axis=1).sort_index().ffill().fillna(False).astype(int).sum(axis=1)
        return super().advise_all_indicators(data)

    def populate_indicators(self, dataframe, metadata):
        data = trend_frame(dataframe, self.params["ladder"], self.params["vol_window"],
                           self.rules["lookback_gap_days"], self.rules["annualisation_days"])
        close = data.close.where(data.can_trade)
        if self.mode == "sma":
            references = close.ffill(limit=self.rules["lookback_gap_days"])
            slow = references.rolling(self.rules["sma_slow"]).mean()
            data["score"] = (references.rolling(self.rules["sma_fast"]).mean() > slow).astype(float).where(slow.notna() & data.can_trade).ffill()
        elif self.mode in ("equal", "hold"):
            data["score"] = 1.0
        decision = data.date + pd.Timedelta(days=1)
        eligible = self._eligible(data, metadata["pair"])
        monday = decision.dt.day_of_week.eq(self.rules["rebalance_weekday"])
        active = (eligible & monday & data.can_trade).cummax()
        count = sum(decision.ge(pd.Timestamp(day, tz="UTC")).astype(int) for day in self.rules["eligibility"].values())
        if hasattr(self, "_eligible_count"):
            count = pd.Series(self._eligible_count.reindex(data.date, method="ffill").to_numpy(), index=data.index)
        scaling = (self.params["sigma_target"] / data.sigma).clip(upper=1) if self.mode == "trend" else 1.0
        data["target_weight"] = (data.score * scaling / count.replace(0, np.nan)).where(active, 0.0)
        data["signal_change"] = data.score.ne(data.score.shift(1)) & data.score.notna()
        data["rebalance"] = active & data.can_trade & (monday | data.signal_change)
        if self.mode == "hold":
            data["target_weight"] = float(metadata["pair"] == self.rules["buy_hold_pair"])
            data["rebalance"] = (metadata["pair"] == self.rules["buy_hold_pair"]) & eligible & data.can_trade & decision.eq(pd.Timestamp(self.rules["path_start"], tz="UTC"))
        return data

    def plan_batch(self, current_time, rows, holdings, cash):
        from accounting import number
        equity = number(cash, "cash") + sum(
            number(quantity, "quantity") * number(rows[pair]["close"], "mark", positive=True)
            for pair, quantity in holdings.items())
        number(equity, "equity", positive=True)
        requests, targets = {}, {}
        for pair, row in rows.items():
            if not row["can_trade"] or not row["rebalance"] or not np.isfinite(row["target_weight"]):
                continue
            price = number(row["close"], "close", positive=True)
            weight = number(row["target_weight"], "weight")
            if weight > 1:
                raise ValueError("No leverage")
            target = weight * equity / price
            delta = target - holdings.get(pair, 0.)
            targets[pair] = target
            if row["signal_change"] or self.mode == "hold" or abs(delta * price) + 1e-9 >= self.params["rebalance_band"] * equity:
                requests[pair] = delta
        self.batch = {"time": current_time, "equity": equity, "targets": targets,
                      "requests": requests, "orders": {p: min(q, 0.) for p, q in requests.items()},
                      "anomalies": [p for p, row in rows.items() if row.get("anomaly")]}
        return self.batch

    def fund_batch(self, current_time, cash, prices, *, risk=None):
        from accounting import number
        import paper_risk
        self.batch["orders"] = {p: min(q, 0.) for p, q in self.batch["orders"].items()}
        if not self.batch["time"] <= current_time < self.batch["time"] + pd.Timedelta(days=1):
            raise ValueError("Expired decision batch")
        fee = self.params["cost_rate"]
        buys = {p: q for p, q in self.batch["requests"].items() if q > 0}
        cost = sum(q * number(prices[p], "execution price", positive=True) * (1 + fee) for p, q in buys.items())
        available = number(cash, "settled cash")
        paper = self.config.get("runmode") != RunMode.BACKTEST
        reason, risk_scale = "historical; paper limits not applied", 1.
        if paper:
            reason, risk_scale = "missing reconciled paper risk context", 0.
            if risk:
                state = paper_risk.evaluate(risk.get("snapshot"), risk.get("state"), risk.get("expected_instances"), current_time)
                reason = state["reason"]
                if state["allow_new_exposure"]:
                    account = state["instance_accounts"][risk["instance"]]
                    if any(o["side"] == "sell" for a in risk["snapshot"]["accounts"]
                           if a["id"] == account for o in a["pending"]):
                        self.batch.update(cash_scale=0., risk_scale=0., risk_reason="waiting for settled sells")
                        self.batch["orders"].update({p: 0. for p in buys})
                        return self.batch
                    pending_fees = sum(o["notional"] * fee for a in risk["snapshot"]["accounts"]
                                       if a["id"] == account for o in a["pending"] if o["side"] == "buy")
                    available = max(0., min(available, state["available_cash"][account] - pending_fees))
                    risk_scale = 1.
        cash_scale = min(1., available / cost) if cost else 1.
        if paper and risk_scale and buys:
            def allowed(factor):
                proposals = [{"instance": risk["instance"], "pair": p,
                              "notional": q * prices[p] * cash_scale * factor} for p, q in buys.items()
                             if q * cash_scale * factor > 0]
                return paper_risk.evaluate(risk["snapshot"], risk["state"], risk["expected_instances"],
                                           current_time, proposals=proposals)
            verdict = allowed(1.)
            reason = verdict["reason"]
            if not verdict["allow_new_exposure"]:
                low, high = 0., 1.
                for _ in range(50):
                    middle = (low + high) / 2
                    if allowed(middle)["allow_new_exposure"]:
                        low = middle
                    else:
                        high = middle
                risk_scale = low if low > 1e-10 else 0.
        self.batch.update(cash_scale=cash_scale, risk_scale=risk_scale, risk_reason=reason)
        self.batch["orders"].update({p: q * cash_scale * risk_scale for p, q in buys.items()})
        return self.batch

    def _order_quantity(self, pair, current_time):
        batch = getattr(self, "batch", {})
        if batch and batch["time"] <= current_time < batch["time"] + pd.Timedelta(days=1):
            return batch["orders"].get(pair, 0.)
        return 0.

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        amount = self._order_quantity(pair, current_time) * current_rate
        if side != "long" or leverage != 1 or not np.isfinite(amount):
            return 0.
        return amount if (min_stake or 0) <= amount <= max_stake else 0.

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, **kwargs):
        quantity = self._order_quantity(trade.pair, current_time)
        if quantity < 0:
            return -min(-quantity, trade.amount) * trade.stake_amount / trade.amount
        amount = quantity * current_rate
        return amount if amount > 0 and (min_stake or 0) <= amount <= max_stake else None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        wanted = self._order_quantity(pair, current_time)
        return side == "long" and wanted > 0 and 0 < amount <= wanted + 1e-10

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        if self._order_quantity(pair, current_time):
            self.batch["orders"][pair] = 0.

    def populate_entry_trend(self, dataframe, metadata):
        dataframe["enter_long"] = (dataframe.rebalance & dataframe.target_weight.gt(0)).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe["exit_long"] = 0
        return dataframe


class H1SimpleTrend(H1Trend):
    mode = "sma"


class H1EqualWeight(H1Trend):
    mode = "equal"


class H1BuyHold(H1Trend):
    mode = "hold"
