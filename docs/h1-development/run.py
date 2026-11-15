"""Reproduce the registered development grid and base-cost checks, never selection/eval."""
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "user_data"
sys.path[:0] = [str(BASE / "scripts"), str(BASE / "strategies/candidates")]
import research
from H1Trend import grid_params
from h1_backtest import run_development

work = BASE / "data/h1-development"
work.mkdir(parents=True, exist_ok=False)
manifest = research.freeze_snapshot(BASE / "data/h1-dev-bounded", work / "snapshot", "20180111", "20211231")
snapshot = json.loads(manifest.read_text())
policy = json.loads((BASE / "configs/research-h1.example.json").read_text())
assert policy["splits"]["dev"] == ["20190114", "20211231"]
assert policy["warmup"] == ["20180111", "20190113"]
policy["snapshots"]["S1"] = {"manifest": str(manifest), "hash": snapshot["data_snapshot_hash"]}
config = json.loads((BASE / "configs/h1_dryrun.json").read_text())
config.pop("startup_candle_count")
config["export"] = "trades"
assert config["dry_run"] is True
assert not any(config["exchange"][key] for key in ("key", "secret", "password"))
results = work / "attempts.jsonl"
expected = len(grid_params(policy)) + 3
count = 0
for cost, multiplier in (("gate", 2), ("base", 1)):
    variant = deepcopy(policy)
    variant["costs"]["stress_multiplier"] = multiplier
    grid = grid_params(variant)
    engine = deepcopy(config)
    engine["fee"] = grid["M45"]["cost_rate"]
    engine_path = work / f"engine-{cost}.json"
    engine_path.write_text(json.dumps(engine, indent=2) + "\n")
    jobs = [(name, "H1Trend", params) for name, params in grid.items()]
    jobs += [(name, name, grid["M45"]) for name in ("H1BuyHold", "H1SimpleTrend", "H1EqualWeight")]
    assert len(jobs) == expected == 12
    for name, strategy, params in jobs:
        variant["params"] = params
        result = run_development(BASE, variant, strategy, engine_path, sensitivity=(cost == "base"))
        item = dict(label=name, cost=cost, result=result)
        with results.open("a") as out:
            out.write(json.dumps(item, default=str) + "\n")
        with research.ledger() as conn:
            row = dict(conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (result["trial_id"],)).fetchone())
        assert row["split"] == "dev" and row["window_end"] == "20211231"
        assert json.loads(row["notes"])["params"]["engine_version"] == "2026.10"
        assert result["success"], item
        count += 1
        print(cost, name, row["trial_id"], row["verdict"], flush=True)
assert count == 24
assert research.search_budget()["H1"]["N"] == 0
print("24 attempts succeeded, 24 ledger rows verified; N=0", flush=True)
