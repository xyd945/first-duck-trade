"""Verify actual H1 dev exports and publish derived evidence, not exchange OHLCV."""
import csv
import io
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "user_data"
sys.path.insert(0, str(BASE / "scripts"))
import research
from accounting import freqtrade_fills, metrics

out = Path(__file__).resolve().parent
attempts = [json.loads(line) for line in (BASE / "data/h1-development/attempts.jsonl").read_text().splitlines()]
assert len(attempts) == len({a["result"]["trial_id"] for a in attempts}) == 24
assert Counter(a["cost"] for a in attempts) == {"gate": 12, "base": 12}
with research.ledger() as conn:
    rows = [dict(conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (a["result"]["trial_id"],)).fetchone()) for a in attempts]
    sealed_rows = conn.execute("SELECT count(*) FROM trial_ledger WHERE split IN ('selection','eval','forward')").fetchone()[0]
    diagnostics = [dict(row) for row in conn.execute("SELECT * FROM trial_ledger WHERE verdict='invalid' ORDER BY created_utc")]
assert sealed_rows == 0
assert Counter(row["verdict"] for row in rows) == {"ok": 9, "baseline": 3, "sensitivity": 12}
assert len({row["config_hash"] for row in rows if row["verdict"] == "ok"}) == 9
for label in {attempt["label"] for attempt in attempts}:
    matching = [row for attempt, row in zip(attempts, rows) if attempt["label"] == label]
    assert len(matching) == 2 and matching[0]["config_hash"] == matching[1]["config_hash"]
assert all(row["split"] == "dev" and row["net_return"] is None for row in diagnostics)
snapshot_path = BASE / "data/h1-development/snapshot/manifest.json"
snapshot = json.loads(snapshot_path.read_text())
research.verify_snapshot(snapshot_path, snapshot["data_snapshot_hash"])
assert snapshot["window_end"] == "20211231"
expected_days = pd.date_range("2019-01-14", "2021-12-31", tz="UTC")
summary = []
max_nav_error = 0.0
receipt_hashes = {}
with zipfile.ZipFile(out / "execution-evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as evidence:
    for attempt, row in zip(attempts, rows):
        label, cost = attempt["label"], attempt["cost"]
        assert row["split"] == "dev" and row["window_start"] == "20190114" and row["window_end"] == "20211231"
        assert row["data_snapshot_hash"] == snapshot["data_snapshot_hash"]
        notes = json.loads(row["notes"])
        assert notes["state"] == "finished"
        assert notes["params"]["engine_version"] == "2026.10"
        assert notes["params"]["ccxt_version"] == "4.5.90"
        run_dir = Path(notes["equity_path"]).parent
        config = json.loads((run_dir / "config.json").read_text())
        batches = json.loads((run_dir / "decision-batches.json").read_text())
        curve = pd.read_feather(notes["equity_path"]).set_index("date")
        assert list(curve.index) == list(expected_days)
        assert [pd.Timestamp(b["time"]) for b in batches] == list(expected_days)
        assert (curve.cash >= -1e-6).all()
        marks = {}
        for pair in config["exchange"]["pair_whitelist"]:
            data = pd.read_feather(run_dir / "data" / f"{pair.replace('/', '_')}-1d.feather")
            assert data.date.max() < pd.Timestamp("2022-01-01", tz="UTC")
            marks[pair] = data.set_index("date").close
        for batch in batches:
            day = pd.Timestamp(batch["time"])
            nav = batch["cash"] + sum(quantity * marks[pair].loc[day] for pair, quantity in batch["holdings"].items())
            error = abs(nav - curve.loc[day, "equity"])
            max_nav_error = max(max_nav_error, error)
            assert math.isclose(nav, curve.loc[day, "equity"], rel_tol=1e-8, abs_tol=1e-6), (label, cost, day, error)
        with zipfile.ZipFile(notes["trades_export_path"]) as archive:
            stats = [json.loads(archive.read(name)) for name in archive.namelist() if name.endswith(".json")]
            strategy = json.loads(row["params_json"])["config"]["strategy"]
            result = next(s["strategy"][strategy] for s in stats if isinstance(s.get("strategy"), dict) and strategy in s["strategy"])
        fills = freqtrade_fills(result["trades"], expected_days[-1])
        turnover_notional = sum(fill["quantity"] * fill["price"] for fill in fills)
        assert len(fills) == row["n_trades_or_rebalances"]
        assert math.isclose(turnover_notional * config["fee"], notes["cost_total"], rel_tol=1e-9)
        first_mondays = {"BTC/USDT": "2019-01-14", "ETH/USDT": "2019-01-14", "XRP/USDT": "2021-01-04", "SOL/USDT": "2021-10-04"}
        assert all(pd.Timestamp(first_mondays[f["pair"]], tz="UTC") <= f["date"] <= expected_days[-1] for f in fills)
        recalculated = metrics(curve, 10000, expected_days[0])
        assert all(math.isclose(row[key], recalculated[key], rel_tol=1e-10, abs_tol=1e-10) for key in ("net_return", "net_sharpe", "max_drawdown"))
        summary.append(dict(label=label, cost=cost, trial_id=row["trial_id"], net_return=row["net_return"], max_drawdown=row["max_drawdown"],
                            net_sharpe=row["net_sharpe"], fills=len(fills), turnover_notional=turnover_notional,
                            turnover_x_mean_equity=turnover_notional / curve.equity.mean(), cost_total=notes["cost_total"],
                            ending_equity=notes["ending_equity"], divergent_decisions=notes["params"]["divergent_decisions"]))
        receipt_hashes[row["trial_id"]] = {}
        for filename in ("decision-batches.json", "net-equity.feather", "run.json"):
            path = run_dir / filename
            receipt_hashes[row["trial_id"]][filename] = research.file_hash(path)
            evidence.write(path, f"{cost}/{label}/{filename}")
        evidence.writestr(f"{cost}/{label}/fills.json", json.dumps(fills, default=str, indent=2))
        buffer = io.StringIO()
        curve.reset_index().to_csv(buffer, index=False)
        evidence.writestr(f"{cost}/{label}/net-equity.csv", buffer.getvalue())
# Replace machine-local path prefixes only; leave every numeric ledger field and ID unchanged.
public_rows = [{k: v.replace(str(ROOT) + "/", "") if isinstance(v, str) else v for k, v in row.items()} for row in rows]
for filename, data in (("ledger.csv", public_rows), ("results.csv", summary)):
    with (out / filename).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
if diagnostics:
    with (out / "prior-diagnostics.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(diagnostics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: v.replace(str(ROOT) + "/", "") if isinstance(v, str) else v for k, v in row.items()} for row in diagnostics)
(out / "snapshot-manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n")
coverage = json.loads((BASE / "data/h1-dev-bounded/coverage.json").read_text())
(out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
verification = dict(attempts=len(attempts), ledger_rows=len(rows), verdicts=dict(Counter(r["verdict"] for r in rows)),
                    scored_days=len(expected_days), snapshot_hash=snapshot["data_snapshot_hash"],
                    market_code_shas=sorted({r["git_sha"] for r in rows}),
                    engine="2026.10", ccxt="4.5.90", max_daily_nav_reconciliation_error=max_nav_error,
                    sealed_window_rows=sealed_rows, budget=research.search_budget(), receipt_hashes=receipt_hashes)
(out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
lines = ["| Configuration | Cost | Net return | Max drawdown | Net Sharpe | Turnover | Costs (USDT) | Fills |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
for row in summary:
    lines.append(f"| {row['label']} | {row['cost']} | {row['net_return']:.2%} | {row['max_drawdown']:.2%} | {row['net_sharpe']:.3f} | {row['turnover_x_mean_equity']:.3f}× | {row['cost_total']:,.2f} | {row['fills']} |")
(out / "results-table.md").write_text("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in verification.items() if k != "receipt_hashes"}, indent=2))
print("\n".join(lines))
