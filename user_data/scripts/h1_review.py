"""One cumulative H1 forward ledger row per weekly review, never backfilled."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid

import pandas as pd

import research
from h1_service import FORWARD_START


def records(directory, kind):
    with sqlite3.connect(f"file:{Path(directory).resolve() / 'journal.sqlite'}?mode=ro", uri=True) as conn:
        return [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM paper_records WHERE kind=? ORDER BY id", (kind,))]


def score(directory, first, end, data, nav, result):
    import math
    from accounting import equity_curve, metrics
    days = pd.date_range(first, end)
    marks = pd.DataFrame({p: df.set_index("date").close.reindex(days) for p, df in data.items()}, index=days)
    with sqlite3.connect(f"file:{directory.resolve() / 'trades.sqlite'}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        orders = [dict(r) for r in conn.execute("""SELECT o.*, t.fee_open, t.fee_close, t.leverage, t.is_short
            FROM orders o JOIN trades t ON t.id=o.ft_trade_id WHERE o.filled > 0 ORDER BY o.order_filled_date, o.id""")]
    fills, cost_sum, notional = [], 0., 0.

    with sqlite3.connect(f"file:{directory.resolve() / 'journal.sqlite'}?mode=ro", uri=True) as conn:
        references = {key: json.loads(value) for key, value in conn.execute("SELECT id,payload FROM paper_records WHERE kind='reference'")}
    for order in orders:
        stamp = pd.to_datetime(order["order_filled_date"], utc=True)
        if pd.isna(stamp) or stamp < first:
            raise ValueError("Unexpected pre-forward or undated fill")
        if stamp >= end + pd.Timedelta(days=1):
            continue
        if order["fee_open"] != .0015 or order["fee_close"] != .0015 or order["leverage"] != 1 or order["is_short"]:
            raise ValueError("Forward trade execution contract drift")
        price = order["average"] or order["price"]
        quantity, side, pair = order["filled"], order["ft_order_side"], order["ft_pair"]
        fills.append(dict(date=stamp, pair=pair, quantity=quantity, price=price, side=side))
        reference = references[stamp.floor("D").isoformat()][pair]
        cost_sum += quantity * ((price - reference) * (1 if side == "buy" else -1) + price * .0015)
        notional += quantity * reference
    curve = equity_curve(10000., fills, marks, cost_rate=.0015)
    for day, row in curve.iterrows():
        if not math.isclose(row.equity, nav[day.isoformat()]["equity"], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Forward NAV does not reconcile to filled orders")
    result.update(metrics(curve, 10000., first))
    result.update(n_trades_or_rebalances=len(fills), average_cost_per_side=cost_sum / notional if notional else None,
                  cost_check_pass=(cost_sum / notional <= .00225) if notional else None)
    return result


def comparison(directory, identity, first, end, batches, trial, result):
    import hashlib
    from h1_service import ROOT
    from h1_replay import offline_exchange, compare
    from h1_backtest import H1Backtesting
    from freqtrade.configuration import TimeRange
    from freqtrade.enums import RunMode, TradingMode, CandleType
    from freqtrade.persistence import enable_database_use
    pairs = list(identity["rules"]["eligibility"])
    data = {p: pd.read_feather(directory / "data" / f"{p.replace('/', '_')}-1d.feather") for p in pairs}
    data = {p: df.loc[df.date <= end].copy() for p, df in data.items()}
    digest = research.composite_hash({p.replace("/", "-"): {
        "sha256": hashlib.sha256(research.canonical_csv(df)).hexdigest()} for p, df in data.items()})
    days = pd.date_range(first, end)
    cfg = json.loads((ROOT / "user_data/configs/h1_dryrun.json").read_text())
    cfg.update(research_params=identity["params"], research_h1={**identity["rules"], "path_start": first.strftime("%Y-%m-%d")},
               startup_candle_count=365, runmode=RunMode.BACKTEST, trading_mode=TradingMode.SPOT,
               candle_type_def=CandleType.SPOT, h1_trial_id=trial, user_data_dir=directory, datadir=directory / "data",
               strategy_path=str(ROOT / "user_data/strategies/candidates"), export="trades")
    cfg["timerange"] = f"{first - pd.Timedelta(days=1):%Y%m%d}-{end + pd.Timedelta(days=1):%Y%m%d}"
    exchange = offline_exchange(cfg)
    try:
        engine = H1Backtesting(cfg, exchange=exchange)
        engine.backtest_one_strategy(engine.strategylist[0], data, TimeRange.parse_timerange(cfg["timerange"]))
        expected = engine.batches
        if [r["time"] for r in expected] != [d.isoformat() for d in days]:
            raise ValueError("Weekly backtest did not cover every forward decision")
        result.update(comparison=compare(list(batches.values()), expected, pairs), comparison_available=True,
                      data_snapshot_hash=digest, engine_version="2026.10", ccxt_version="4.5.90")
    finally:
        enable_database_use()
        exchange.close()
    return data


def write_forward(directory, through, *, now=None):
    from h1_replay import compare
    from h1_service import strategy_hash
    now = now or datetime.now(timezone.utc)
    directory = Path(directory)
    first = pd.Timestamp(FORWARD_START)
    end = pd.Timestamp(through, tz="UTC")
    if end < first or end >= pd.Timestamp(now).floor("D") or through != end.strftime("%Y-%m-%d"):
        raise ValueError("Forward review needs completed days from 2026-12-19")
    identity = json.loads((directory / "identity.json").read_text())
    week = (end + pd.Timedelta(days=1)).strftime("%G-W%V")
    days = pd.date_range(first, end)
    expected = [{"time": d.isoformat(), "target_weights": {}} for d in days]
    verdict, result = "missed", {"state": "finished", "week": week, "placeholder": identity["selection"]["placeholder"],
                                "comparison": compare([], expected, list(identity["rules"]["eligibility"])),
                                "comparison_available": False}
    with research.ledger() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS h1_weekly_reviews (week TEXT PRIMARY KEY, trial_id TEXT NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT trial_id FROM h1_weekly_reviews WHERE week=?", (week,)).fetchone()
        if existing:
            row = dict(conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (existing[0],)).fetchone())
            if row["window_end"] != end.strftime("%Y%m%d") or json.loads(row["params_json"])["config"] != identity:
                raise ValueError("This review week already has a different cutoff or identity")
            return row
        trial = uuid.uuid4().hex
        conn.execute("INSERT INTO h1_weekly_reviews VALUES (?,?)", (week, trial))
        conn.execute("""INSERT INTO trial_ledger
            (trial_id,created_utc,hypothesis,repo,git_sha,config_hash,data_snapshot_hash,split,
             window_start,window_end,params_json,verdict,notes)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (trial, now.isoformat(), "H1", "first-duck-trade", identity["git_sha"],
             identity["selection"]["config_hash"], "", "forward", first.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
             json.dumps({"config": identity, "kind": "weekly_review"}), "error", json.dumps({**result, "state": "running"})))
    try:
        if identity["selection"]["placeholder"]:
            verdict = "invalid"
        elif strategy_hash(identity["params"]) != identity["selection"]["config_hash"]:
            verdict = "invalid"
            result["error"] = "Forward strategy changed after selection"
        else:
            batches = {r["time"]: r for r in records(directory, "batch")}
            errors = [r for r in records(directory, "error") if first <= pd.Timestamp(r["time"]) <= end]
            result["operational_errors"] = errors
            for failure in errors:
                day = failure["time"]
                if batches.get(day, {}).get("status") != "finished":
                    batches[day] = {**batches.get(day, {}), "time": day, "status": "error", "risk_reason": failure["error"]}
            result["operational_divergences"] = [dict(kind=kind, **r)
                for kind in ("latch", "reconcile", "reidentify") for r in records(directory, kind)
                if first <= pd.Timestamp(r["time"]) <= end]
            for receipt in result["operational_divergences"]:
                day = receipt["time"]
                batch = batches.get(day, {})
                finished_before = (batch.get("status") == "finished" and batch.get("finished_at")
                                   and receipt.get("observed")
                                   and pd.Timestamp(batch["finished_at"]) < pd.Timestamp(receipt["observed"]))
                if finished_before:
                    if receipt["kind"] != "reidentify":
                        continue
                    day = (pd.Timestamp(day) + pd.Timedelta(days=1)).isoformat()
                batches[day] = {**batches.get(day, {"status": "missed"}), "time": day,
                                "risk_reason": receipt["kind"]}
            nav = {r["date"]: r for r in records(directory, "nav")}
            days = pd.date_range(first, end)
            missing = [day.isoformat() for day in days if day.isoformat() not in batches or day.isoformat() not in nav]
            result["missing_days"] = missing
            result["failed_days"] = {d.isoformat(): batches[d.isoformat()].get("status") for d in days
                                     if d.isoformat() in batches and batches[d.isoformat()].get("status") != "finished"}
            result["comparison"] = compare(list(batches.values()), expected, list(identity["rules"]["eligibility"]))
            if not missing or (directory / "data").exists():
                data = comparison(directory, identity, first, end, batches, trial, result)
                if all(d.isoformat() in nav for d in days):
                    score(directory, first, end, data, nav, result)
            verdict = "error" if "error" in result["failed_days"].values() else (
                "missed" if missing or result["failed_days"] else "ok")
    except Exception as exc:
        verdict = "error"
        result.update(state="finished", error=str(exc))
    with research.ledger() as conn:
        conn.execute("""UPDATE trial_ledger SET verdict=?, notes=?, data_snapshot_hash=?,
            n_trades_or_rebalances=?, net_return=?, net_sharpe=?, max_drawdown=? WHERE trial_id=?""",
            (verdict, json.dumps(result, allow_nan=False), result.get("data_snapshot_hash", ""),
             result.get("n_trades_or_rebalances"), result.get("net_return"), result.get("net_sharpe"),
             result.get("max_drawdown"), trial))
        return dict(conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (trial,)).fetchone())


def main():
    import argparse
    from h1_service import ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(ROOT / "user_data/data/h1-paper"))
    parser.add_argument("--through", required=True, help="Last completed UTC day, YYYY-MM-DD (inclusive)")
    args = parser.parse_args()
    try:
        row = write_forward(args.directory, args.through)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(row, indent=2, allow_nan=False))
    return 0 if row["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
