"""G0 sell-first historical execution; never starts a bot or sends exchange orders."""
import json
import logging
from copy import deepcopy

import freqtrade
import ccxt
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import LocalTrade

import research


class H1Backtesting(Backtesting):
    def __init__(self, config, *args, **kwargs):
        if (freqtrade.__version__ != config["research_h1"]["engine_version"] or
                ccxt.__version__ != config["research_h1"]["ccxt_version"]):
            raise ValueError("H1 adapter requires the tested engine version")
        if config.get("runmode") != RunMode.BACKTEST or config.get("timeframe_detail"):
            raise ValueError("H1 adapter supports daily backtests only")
        fixed = {"timeframe": "1d", "dry_run": True, "trading_mode": "spot", "minimal_roi": {},
                 "stoploss": -1., "trailing_stop": False, "position_adjustment_enable": True,
                 "use_exit_signal": False, "position_stacking": False,
                 "startup_candle_count": config["research_h1"]["historical_startup"]}
        if (any(key in config and config[key] != value for key, value in fixed.items()) or
                config.get("fee") != config["research_params"]["cost_rate"] or
                config.get("max_open_trades", 0) < len(config["research_h1"]["eligibility"]) or
                any(config.get("order_types", {}).get(side) != "market" for side in ("entry", "exit"))):
            raise ValueError("H1 execution contract cannot be overridden")
        with research.ledger() as conn:
            trial = conn.execute("SELECT notes FROM trial_ledger WHERE trial_id=?",
                                 (config.get("h1_trial_id"),)).fetchone()
        if not trial or json.loads(trial["notes"]).get("state") != "running":
            raise ValueError("A running ledger trial is required before engine creation")
        super().__init__(config, *args, **kwargs)

    def _set_strategy(self, strategy):
        strategy._h1_adapter = True
        super()._set_strategy(strategy)

    def backtest_loop(self, row, pair, current_time, trade_dir, can_enter):
        # The final registered open is executable; only the later terminal liquidation is synthetic.
        return super().backtest_loop(row, pair, current_time, trade_dir, True)

    def backtest_one_strategy(self, strat, data, timerange):
        first = pd.Timestamp(strat.rules["path_start"], tz="UTC")
        load_start = first - pd.Timedelta(days=strat.startup_candle_count + 1)
        strat._bars_before = {p: int(((df.date < load_start) & df.close.notna() & df.close.gt(0) &
                                  (df["confirm"].astype(str).eq("1") if "confirm" in df else True)).sum())
                              for p, df in data.items()}
        data = {p: df.loc[df.date >= load_start].copy() for p, df in data.items()}
        return super().backtest_one_strategy(strat, data, timerange)

    def backtest(self, processed, start_date, end_date):
        self.frames = {p: df.set_index("date").copy() for p, df in processed.items()}
        self.cash = float(self.config["dry_run_wallet"])
        self.seen_orders = set()
        self.batches = []
        return super().backtest(processed, start_date, end_date)

    def _settled(self):
        for trade in LocalTrade.bt_trades + LocalTrade.bt_trades_open:
            for order in trade.orders:
                if order.id in self.seen_orders or order.ft_is_open or not order.safe_filled:
                    continue
                notional = order.safe_filled * order.safe_price
                fee = self.config["fee"]
                self.cash += notional * (1 - fee) if order.ft_order_side == "sell" else -notional * (1 + fee)
                self.seen_orders.add(order.id)
        if self.cash < -1e-6:
            raise ValueError("Historical execution borrowed cash")
        self.cash = max(0., self.cash)
        return {trade.pair: trade.amount for trade in LocalTrade.bt_trades_open}

    def _time_pair_generator_det(self, current_time, pairs):
        holdings = self._settled()
        previous = pd.Timestamp(current_time) - pd.Timedelta(days=1)
        rows, prices = {}, {}
        for pair in sorted(pairs):
            frame = self.frames[pair]
            earlier = frame.loc[frame.index <= previous]
            if earlier.empty:
                continue
            row = earlier.iloc[-1].to_dict()
            if earlier.index[-1] != previous or not row["can_trade"]:
                row.update(can_trade=False, rebalance=False, anomaly=True)
                row["close"] = earlier.close.where(earlier.can_trade).ffill().iloc[-1]
            rows[pair] = row
            if current_time in frame.index and pd.notna(frame.loc[current_time, "open"]):
                prices[pair] = float(frame.loc[current_time, "open"])
            else:
                row["rebalance"] = False
        batch = self.strategy.plan_batch(current_time, rows, holdings, self.cash)
        sells = sorted(p for p, quantity in batch["requests"].items() if quantity < 0)
        order = sells + sorted(set(pairs) - set(sells))
        for pair in sells:
            yield current_time, True, False, 0, pair
        self._settled()
        self.strategy.fund_batch(current_time, self.cash, prices)
        for pair in order[len(sells):]:
            yield current_time, True, False, 0, pair
        after = self._settled()
        record = deepcopy(self.strategy.batch)
        record.update(time=current_time.isoformat(), execution_order=order,
                      target_weights={p: float(row["target_weight"]) if pd.notna(row["target_weight"]) else None
                                      for p, row in rows.items()},
                      cash=self.cash, holdings=after,
                      unfilled={p: target - after.get(p, 0.) for p, target in batch["targets"].items()})
        self.batches.append(record)


def run_development(base, policy, strategy, config_path, *, exchange=None, sensitivity=False):
    """Every attempt gets a fix-2/4 ledger row; there is no selection/eval option."""
    from pathlib import Path
    from freqtrade.configuration import TimeRange
    from freqtrade.enums import CandleType, TradingMode, MarginMode
    from freqtrade.optimize.optimize_reports import generate_backtest_stats, store_backtest_results
    from backtest_runner import parse_backtest_artifact
    base = Path(base)
    trial = None
    timerange = None
    descriptor = {"hypothesis": "H1", "strategy": strategy}
    sha = ""
    owned_engine = None
    verdict = "sensitivity" if sensitivity else ("ok" if strategy == "H1Trend" else "baseline")
    try:
        sha = research.git_sha(Path(__file__).resolve().parents[2])
        policy = research.load_config(deepcopy(policy))
        if strategy not in ("H1Trend", "H1SimpleTrend", "H1EqualWeight", "H1BuyHold"):
            raise ValueError("Unknown H1 strategy")
        timerange = research.window(policy, "dev")
        descriptor, cfg = research.describe_run(base, policy, strategy, "1d", config_path, verdict=verdict)
        trial = research.start_trial(descriptor, "dev", "backtest", timerange, sha)
        research.check_not_evaluated("H1")
        run_dir = research.prepare_run(base, policy, descriptor, cfg, trial, timerange, "dev")
        first, end = map(research.date, timerange.split("-"))
        if pd.Timestamp(cfg["research_h1"]["path_start"], tz="UTC") != first:
            raise ValueError("H1 path_start must equal development start")
        cfg["original_config"] = deepcopy(cfg)
        cfg.update(h1_trial_id=trial, strategy=strategy, runmode=RunMode.BACKTEST,
                   trading_mode=TradingMode.SPOT, margin_mode=MarginMode.NONE, candle_type_def=CandleType.SPOT,
                   timeframe="1d", user_data_dir=run_dir, datadir=run_dir / "data",
                   strategy_path=str(run_dir / "strategies/candidates"), exportdirectory=run_dir,
                   timerange=f"{(first - pd.Timedelta(days=1)):%Y%m%d}-{end:%Y%m%d}")
        data = {p: pd.read_feather(run_dir / "data" / f"{p.replace('/', '_')}-1d.feather")
                for p in cfg["exchange"]["pair_whitelist"]}
        engine = H1Backtesting(cfg, exchange=exchange)
        if exchange is None:
            owned_engine = engine
        engine.backtest_one_strategy(engine.strategylist[0], data, TimeRange.parse_timerange(cfg["timerange"]))
        if not engine.batches or pd.Timestamp(engine.batches[0]["time"]) != first or pd.Timestamp(engine.batches[-1]["time"]) != end - pd.Timedelta(days=1):
            raise ValueError("Engine did not execute the complete registered development window")
        stats = generate_backtest_stats(data, engine.all_bt_content, first, end - pd.Timedelta(days=1))
        archive = store_backtest_results(cfg, stats, trial,
                    wallet_summary={strategy: engine.all_bt_content[strategy]["wallet_summary"]},
                    strategy_files={strategy: engine.strategylist[0].__file__})
        batch_path = run_dir / "decision-batches.json"
        batch_path.write_text(json.dumps(engine.batches, indent=2, allow_nan=False))
        result = parse_backtest_artifact(archive, strategy, timerange)
        result.update(success=True, trades_export_path=str(archive), params={
            "batch_path": str(batch_path), "decisions": len(engine.batches),
            "divergent_decisions": sum(any(abs(q) > 1e-8 for q in b["unfilled"].values()) for b in engine.batches)})
        result = research.slice_result(result, strategy, timerange, timerange, run_dir)
    except Exception as exc:
        logging.getLogger(__name__).exception("H1 development attempt failed")
        result = {"success": False, "strategy": strategy, "error": str(exc)}
    finally:
        if owned_engine is not None:
            owned_engine.exchange.close()
    if trial is None:
        trial = research.start_trial(descriptor, "dev", "backtest", timerange, sha)
    result.setdefault("params", {}).update(engine_version=freqtrade.__version__, ccxt_version=ccxt.__version__)
    research.finish_trial(trial, result, verdict)
    result.update(trial_id=trial, split="dev", config_hash=research.config_hash(descriptor))
    return result


def main():
    import argparse
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser(description="H1 development only; no bot or sealed-window entry point")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--strategy", choices=("H1Trend", "H1SimpleTrend", "H1EqualWeight", "H1BuyHold"), required=True)
    parser.add_argument("--config-id", help="Grid ID from policy, required for H1Trend")
    parser.add_argument("--sensitivity", action="store_true", help="Label a cost sensitivity; no new statistical trial")
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(base / "strategies/candidates"))
    from H1Trend import grid_params
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text())
    for snapshot in policy["snapshots"].values():
        snapshot["manifest"] = str((policy_path.parent / snapshot["manifest"]).resolve())
    if args.strategy == "H1Trend":
        if args.config_id not in grid_params(policy):
            parser.error("--config-id must name a registered grid configuration")
        policy["params"] = grid_params(policy)[args.config_id]
    result = run_development(base, policy, args.strategy, args.engine_config, sensitivity=args.sensitivity)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
