"""Frozen research inputs, split policy and the shared trial ledger."""
import os
import hashlib
import json
import re
import sqlite3
import uuid
import ast
import io
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def enabled(flag: str) -> bool:
    return os.environ.get(flag, "false").lower() in ("1", "true", "yes")


def file_hash(path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def date(raw: str) -> datetime:
    if not isinstance(raw, str) or not re.fullmatch(r"\d{8}", raw):
        raise ValueError("Dates must be YYYYMMDD UTC")
    return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)


PAIRS = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT")
COLUMNS = ("ts", "open", "high", "low", "close", "volume")


def canonical_csv(frame: pd.DataFrame) -> bytes:
    """UTF-8/LF CSV: Unix milliseconds and plain decimals, no trailing zeroes."""
    from decimal import Decimal
    stamps = pd.to_datetime(frame["date"], utc=True)
    if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        raise ValueError("Invalid candle dates")
    if (stamps != stamps.dt.normalize()).any():
        raise ValueError("Daily candles must open at 00:00 UTC")
    if "confirm" in frame and not frame["confirm"].astype(str).eq("1").all():
        raise ValueError("Unconfirmed candles")
    lines = [",".join(COLUMNS)]
    for stamp, values in zip(stamps, frame[list(COLUMNS[1:])].itertuples(index=False, name=None)):
        numbers = [Decimal(str(v)) for v in values]
        if any(not v.is_finite() for v in numbers):
            raise ValueError("Non-finite OHLCV")
        o, h, low, c, volume = numbers
        if min(o, h, low, c) <= 0 or volume < 0 or h < max(o, c, low) or low > min(o, c):
            raise ValueError("Invalid OHLCV prices or volume")
        fields = [format(v, "f").rstrip("0").rstrip(".") if "." in format(v, "f")
                  else format(v, "f") for v in numbers]
        lines.append(",".join([str(stamp.value // 1_000_000), *fields]))
    return ("\n".join(lines) + "\n").encode()


def csv_frame(data: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(data), dtype=str)
    if tuple(frame.columns) != COLUMNS:
        raise ValueError("Invalid canonical CSV columns")
    frame["date"] = pd.to_datetime(pd.to_numeric(frame.pop("ts")), unit="ms", utc=True)
    return frame


def composite_hash(pairs: dict) -> str:
    return hashlib.sha256("".join(pairs[p]["sha256"] for p in sorted(pairs)).encode()).hexdigest()


def freeze_snapshot(source, destination, start: str, end: str, *, previous=None) -> Path:
    """Freeze the four OKX daily pairs; start and end are inclusive UTC days."""
    first, last = date(start), date(end)
    if first > last:
        raise ValueError("Snapshot start must not follow end")
    cutoff = last + timedelta(days=1, minutes=10)
    if datetime.now(timezone.utc) < cutoff:
        raise ValueError("Snapshot cutoff has not closed plus ten minutes")
    source, destination = Path(source).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 2, "window_start": start, "window_end": end,
                "exchange": "okx", "bar": "1Dutc", "pairs": {}}
    for pair in PAIRS:
        path = source / f"{pair.replace('-', '_')}-1d.feather"
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("Snapshot files must not escape the source")
        frame = pd.read_feather(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        cropped = frame.loc[(frame.date >= first) & (frame.date <= last)]
        if cropped.empty:
            raise ValueError(f"No candles in snapshot window: {pair}")
        data = canonical_csv(cropped)
        filename = f"{pair}.csv"
        (destination / filename).write_bytes(data)
        manifest["pairs"][pair] = {"file": filename, "sha256": hashlib.sha256(data).hexdigest()}
    manifest["data_snapshot_hash"] = composite_hash(manifest["pairs"])
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if previous:
        verify_prefix(*previous, manifest_path, manifest["data_snapshot_hash"])
    return manifest_path


def verify_snapshot(manifest_path, expected_hash: str) -> dict:
    """Recompute canonical pair hashes and their pinned composite, not a manifest digest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest["version"] != 2 or set(manifest["pairs"]) != set(PAIRS)
            or manifest["exchange"] != "okx" or manifest["bar"] != "1Dutc"):
        raise ValueError("Invalid snapshot manifest")
    first, last = date(manifest["window_start"]), date(manifest["window_end"])
    if first > last:
        raise ValueError("Invalid snapshot window")
    if composite_hash(manifest["pairs"]) != expected_hash or manifest["data_snapshot_hash"] != expected_hash:
        raise ValueError("Snapshot composite hash mismatch")
    root = manifest_path.parent.resolve()
    for pair, item in manifest["pairs"].items():
        path = root / item["file"]
        if item["file"] != f"{pair}.csv" or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Snapshot path escapes its directory")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Snapshot data hash mismatch: {pair}")
        frame = csv_frame(data)
        if frame.empty or (frame.date < first).any() or (frame.date > last).any():
            raise ValueError("Candles outside snapshot window")
        if canonical_csv(frame) != data:
            raise ValueError("Non-canonical snapshot CSV")
    return manifest


def verify_prefix(s1_path, s1_hash, s2_path, s2_hash):
    s1 = verify_snapshot(s1_path, s1_hash)
    s2 = verify_snapshot(s2_path, s2_hash)
    if s1["window_start"] != s2["window_start"] or s1["window_end"] >= s2["window_end"]:
        raise ValueError("S2 must extend the S1 prefix window")
    cutoff = date(s1["window_end"])
    for pair in PAIRS:
        prior = (Path(s1_path).parent / s1["pairs"][pair]["file"]).read_bytes()
        later = (Path(s2_path).parent / s2["pairs"][pair]["file"]).read_bytes()
        frame = csv_frame(later)
        if canonical_csv(frame.loc[frame.date <= cutoff]) != prior:
            raise ValueError(f"S2 prefix differs from S1: {pair}")


def read_config() -> dict:
    raw = os.environ.get("RESEARCH_CONFIG")
    if not raw:
        raise ValueError("RESEARCH_CONFIG is required; unrestricted backtests are disabled")
    path = Path(raw).resolve()
    config = json.loads(path.read_text())
    for snapshot in config["snapshots"].values():
        snapshot["manifest"] = str((path.parent / snapshot["manifest"]).resolve())
    return config


def load_config(config=None) -> dict:
    config = read_config() if config is None else config
    if config.get("hypothesis") not in ("H1", "H2", "other"):
        raise ValueError("hypothesis must be H1, H2 or other")
    warm_start, previous_end = map(date, config["warmup"])
    if warm_start > previous_end:
        raise ValueError("Invalid warm-up window")
    if set(config["splits"]) != {"dev", "selection", "eval"}:
        raise ValueError("splits must contain dev, selection and eval")
    for split in ("dev", "selection", "eval"):
        start, end = map(date, config["splits"][split])
        if start > end or start != previous_end + timedelta(days=1):
            raise ValueError("Split windows overlap, have gaps or are not chronological")
        previous_end = end
    if set(config["snapshots"]) != {"S1", "S2"}:
        raise ValueError("Configure S1 and S2 snapshots separately")
    return config


def window(config: dict, split: str, requested: str | None = None) -> str:
    if split not in ("dev", "selection", "eval"):
        raise ValueError("Unknown research split")
    start, end = config["splits"][split]
    end = (date(end) + timedelta(days=1)).strftime("%Y%m%d")
    if requested is None:
        return f"{start}-{end}"
    left, right = requested.split("-")
    if not (date(start) <= date(left) < date(right) <= date(end)):
        raise ValueError("Requested timerange escapes its split")
    if split == "eval" and requested != f"{start}-{end}":
        raise ValueError("Evaluation must use its entire preregistered window")
    return requested


def path_window(config, split, requested=None):
    end = window(config, split, requested).split("-")[1]
    return f"{config['splits']['dev'][0]}-{end}"


def config_hash(descriptor: dict) -> str:
    if "config_hash" in descriptor:
        return descriptor["config_hash"]
    return json_hash(descriptor)


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@contextmanager
def ledger():
    path = Path(os.environ.get("TRIAL_LEDGER_PATH", str(
        Path(__file__).resolve().parents[1] / "data" / "trial_ledger.sqlite")))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS trial_ledger (
            trial_id TEXT PRIMARY KEY, created_utc TEXT NOT NULL,
            hypothesis TEXT NOT NULL, repo TEXT NOT NULL, git_sha TEXT,
            config_hash TEXT, data_snapshot_hash TEXT, split TEXT,
            window_start TEXT, window_end TEXT, params_json TEXT NOT NULL,
            n_trades_or_rebalances INTEGER, net_return REAL, net_sharpe REAL,
            max_drawdown REAL, verdict TEXT NOT NULL CHECK (verdict IN
            ('ok','selected','g1_pass','g1_fail','missed','error','baseline','sensitivity','rerun','invalid')),
            notes TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS evaluation_registrations (
            registration TEXT PRIMARY KEY, hypothesis TEXT NOT NULL, config_hash TEXT NOT NULL,
            descriptor_json TEXT NOT NULL, registered_utc TEXT NOT NULL, opened_utc TEXT)""")
        conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_selection
            ON trial_ledger(hypothesis) WHERE verdict='selected'""")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def registration_key(descriptor):
    # S2 does not exist at registration. The S1 hash is bound by policy_hash.
    return json_hash({k: v for k, v in descriptor.items() if k != "data_snapshot_hash"})


def preregister(descriptor: dict) -> str:
    digest = registration_key(descriptor)
    with ledger() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM evaluation_registrations WHERE registration=?", (digest,)).fetchone():
            return digest
        if conn.execute("SELECT 1 FROM trial_ledger WHERE hypothesis=? AND verdict='selected'",
                        (descriptor["hypothesis"],)).fetchone():
            raise ValueError("Preregister every variant before selection is locked")
        conn.execute("INSERT INTO evaluation_registrations VALUES (?, ?, ?, ?, ?, NULL)",
                     (digest, descriptor["hypothesis"], config_hash(descriptor),
                      json.dumps(descriptor, sort_keys=True), datetime.now(timezone.utc).isoformat()))
    return digest


def record_selection(trial_id):
    """Copy one completed ordinary selection result into the immutable choice row."""
    selected = uuid.uuid4().hex
    with ledger() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (trial_id,)).fetchone()
        if not row or row["split"] != "selection" or row["verdict"] not in ("ok", "rerun"):
            raise ValueError("Choose a successful ordinary selection trial")
        if conn.execute("SELECT 1 FROM trial_ledger WHERE hypothesis=? AND verdict='selected'",
                        (row["hypothesis"],)).fetchone():
            raise ValueError("Hypothesis already selected")
        descriptor = json.loads(row["params_json"])["config"]
        deadline = date(descriptor["splits"]["eval"][1]) + timedelta(days=1)
        if datetime.now(timezone.utc) >= deadline:
            raise ValueError("Selection deadline passed; never backfill the forward choice")
        if not conn.execute("SELECT 1 FROM evaluation_registrations WHERE registration=?",
                            (registration_key(descriptor),)).fetchone():
            raise ValueError("Preregister the chosen config before selection")
        values = dict(row)
        values.update(trial_id=selected, created_utc=datetime.now(timezone.utc).isoformat(),
                      verdict="selected", notes=json.dumps({"selected_trial": trial_id}))
        conn.execute("INSERT INTO trial_ledger VALUES (" + ",".join("?" for _ in values) + ")",
                     tuple(values.values()))
    return selected


def claim_evaluation(registration: str, descriptor):
    """Consume once, after a matching selection, before reading any S2 candles."""
    if registration != registration_key(descriptor):
        raise ValueError("Config changed since preregistration")
    with ledger() as conn:
        conn.execute("BEGIN IMMEDIATE")
        selected = conn.execute("SELECT params_json FROM trial_ledger WHERE hypothesis=? AND verdict='selected'",
                                (descriptor["hypothesis"],)).fetchone()
        if not selected or json.loads(selected["params_json"])["config"]["policy_hash"] != descriptor["policy_hash"]:
            raise ValueError("A matching selection row is required before evaluation")
        cutoff = date(descriptor["splits"]["eval"][1]) + timedelta(days=1, minutes=10)
        if datetime.now(timezone.utc) < cutoff:
            raise ValueError("Evaluation snapshot cutoff has not closed plus ten minutes")
        cursor = conn.execute("""UPDATE evaluation_registrations SET opened_utc = ?
            WHERE registration = ? AND opened_utc IS NULL""",
            (datetime.now(timezone.utc).isoformat(), registration))
        if cursor.rowcount != 1:
            raise ValueError("Evaluation is not preregistered or was already opened")


def check_not_evaluated(hypothesis: str):
    with ledger() as conn:
        if conn.execute("SELECT 1 FROM evaluation_registrations WHERE hypothesis=? AND opened_utc IS NOT NULL",
                        (hypothesis,)).fetchone():
            raise ValueError("Evaluation already opened; this hypothesis cannot return to search")


def start_trial(descriptor, split, kind, timerange, git_sha, epochs=0, search=None) -> str:
    trial_id = uuid.uuid4().hex
    start, _, end = (timerange or "-").partition("-")
    if end:
        try:
            end = (date(end) - timedelta(days=1)).strftime("%Y%m%d")
        except ValueError:
            pass  # Retain malformed requests in the error ledger too.
    params = {**descriptor.get("params", {}), "kind": kind, "epochs_requested": epochs,
              "config": descriptor, "search": search}
    with ledger() as conn:
        conn.execute("""INSERT INTO trial_ledger (
            trial_id, created_utc, hypothesis, repo, git_sha, config_hash,
            data_snapshot_hash, split, window_start, window_end, params_json, verdict, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trial_id, datetime.now(timezone.utc).isoformat(), descriptor.get("hypothesis", "other"),
             "first-duck-trade", git_sha, config_hash(descriptor),
             descriptor.get("data_snapshot_hash", ""), split, start, end,
             json.dumps(params, sort_keys=True), "error", json.dumps({"state": "running"})))
    return trial_id


def finish_trial(trial_id, result, verdict="ok"):
    if verdict not in ("ok", "baseline", "sensitivity", "rerun", "invalid", "error"):
        raise ValueError("Invalid run verdict")
    # Keep whole-path engine output separate from independently marked net metrics.
    notes = {key: result[key] for key in (
        "error", "profit_total_pct", "sharpe", "max_drawdown_pct", "params",
        "best_epoch", "total_epochs", "trades_export_path", "params_path", "path_timerange",
        "path_metrics", "slice_path", "equity_path", "net_path_metrics", "opening_equity",
        "ending_equity", "cost_total", "cost_model") if key in result}
    notes["state"] = "finished"
    with ledger() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM trial_ledger WHERE trial_id=?", (trial_id,)).fetchone()
        if not row or json.loads(row["notes"]).get("state") != "running":
            raise ValueError("Trial missing or already finalised")
        if not result.get("success"):
            verdict = "error"
        elif verdict == "ok" and conn.execute("""SELECT 1 FROM trial_ledger
                WHERE hypothesis=? AND config_hash=? AND data_snapshot_hash=? AND split=?
                AND window_start=? AND window_end=? AND params_json=? AND verdict IN ('ok','rerun')""",
                tuple(row[k] for k in ("hypothesis", "config_hash", "data_snapshot_hash", "split",
                                      "window_start", "window_end", "params_json"))).fetchone():
            verdict = "rerun"
        cursor = conn.execute("""UPDATE trial_ledger SET
            n_trades_or_rebalances=?, net_return=?, net_sharpe=?, max_drawdown=?,
            verdict=?, notes=? WHERE trial_id=?""",
            (result.get("total_trades"), result.get("net_return"), result.get("net_sharpe"),
             result.get("max_drawdown"), verdict,
             json.dumps(notes, sort_keys=True), trial_id))
        if cursor.rowcount != 1:
            raise ValueError("Trial missing or already finalised")


def search_budget() -> dict:
    """G0 N counts distinct selection/evaluation hashes, not rows or dev epochs."""
    with ledger() as conn:
        rows = conn.execute("SELECT * FROM trial_ledger").fetchall()
    report = {}
    counted = {}
    for row in rows:
        entry = report.setdefault(row["hypothesis"], dict(
            trials=0, failed=0, unfinished=0, backtests=0, evaluations=0,
            hyperopt_batches=0, hyperopt_epochs_requested=0))
        entry["trials"] += 1
        entry["failed"] += row["verdict"] == "error"
        try:
            notes = json.loads(row["notes"] or "{}")
        except json.JSONDecodeError:
            notes = {}
        entry["unfinished"] += isinstance(notes, dict) and notes.get("state") == "running"
        hashes = counted.setdefault(row["hypothesis"], set())
        if row["split"] in ("selection", "eval") and row["verdict"] not in ("baseline", "sensitivity"):
            hashes.add(row["config_hash"])
        entry["N"] = len(hashes)
        params = json.loads(row["params_json"])
        kind = params.get("kind")
        if kind == "hyperopt":
            entry["hyperopt_batches"] += 1
            entry["hyperopt_epochs_requested"] += params["epochs_requested"]
        elif kind == "evaluation":
            entry["evaluations"] += 1
        elif kind == "backtest":
            entry["backtests"] += 1
    return report


def git_sha(root) -> str:
    value = os.environ.get("RESEARCH_GIT_SHA") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("RESEARCH_GIT_SHA must be a full commit SHA")
    return value


def describe_run(base, policy, strategy, timeframe, config_path=None, freqai_model=None, verdict="ok"):
    """Bind the effective engine config, code and parameter sidecars before a run."""
    if verdict not in ("ok", "baseline", "sensitivity"):
        raise ValueError("Run verdict must be ok, baseline or sensitivity; reruns are detected")
    if not re.fullmatch(r"[A-Za-z_]\w*", strategy):
        raise ValueError("Invalid strategy class name")
    path = Path(config_path or "config.json")
    container_root = Path("/freqtrade/user_data")
    if path.is_relative_to(container_root):
        path = path.relative_to(container_root)
    path = (base / path).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Engine config must be inside user_data")
    config_bytes = path.read_bytes()
    engine_config = json.loads(config_bytes)
    if engine_config.get("add_config_files"):
        raise ValueError("Flatten engine config; includes are not frozen")
    pending = [engine_config.get("exchange", {})]
    while pending:
        settings = pending.pop()
        for key, value in settings.items():
            if key.lower() in ("key", "apikey", "secret", "password", "privatekey") and value:
                raise ValueError("Research config must not contain exchange credentials")
            if isinstance(value, dict):
                pending.append(value)
    code = {}
    matches = []
    for directory in ("strategies/candidates", "indicators", "hyperopts"):
        for source in sorted((base / directory).rglob("*")):
            if source.suffix not in (".py", ".json") or not source.is_file():
                continue
            if source.is_symlink() or not source.resolve().is_relative_to(base.resolve()):
                raise ValueError("Code paths must stay inside user_data")
            relative = source.relative_to(base).as_posix()
            code[relative] = file_hash(source)
            if directory == "strategies/candidates" and source.suffix == ".py":
                if any(isinstance(n, ast.ClassDef) and n.name == strategy
                       for n in ast.parse(source.read_text()).body):
                    matches.append(relative)
    if len(matches) != 1:
        raise ValueError("Strategy class must occur once in strategies/candidates")
    params = policy["params"]
    from accounting import cost_rate
    costs = policy.get("costs", {})
    rate = cost_rate(costs)
    if abs(rate - params.get("cost_rate", -1)) > 1e-12 or engine_config.get("fee") != params["cost_rate"]:
        raise ValueError("Engine fee and cost_rate must equal the explicit cost model")
    engine_config["research_costs"] = costs.copy()
    if policy["hypothesis"] == "H1":
        if set(params) != {"ladder", "sigma_target", "vol_window", "rebalance_band", "cost_rate"}:
            raise ValueError("H1 params must contain exactly the four G0 fields and cost_rate")
        exchange = engine_config.get("exchange", {})
        if (timeframe != "1d" or freqai_model or exchange.get("name") != "okx"
                or set(exchange.get("pair_whitelist", [])) != {p.replace('-', '/') for p in PAIRS}
                or engine_config.get("trading_mode", "spot") != "spot"
                or engine_config.get("dry_run") is not True
                or engine_config.get("pairlists") != [{"method": "StaticPairList"}]):
            raise ValueError("H1 requires the four OKX spot daily pairs, static pairlist and dry_run")
        if engine_config.get("fee") != params.get("cost_rate") or params.get("cost_rate", -1) < 0:
            raise ValueError("Engine fee must equal the recorded cost_rate")
        if (base / matches[0]).with_suffix(".json").exists():
            raise ValueError("H1 uses registered params, not hidden hyperopt sidecars")
    engine_config["research_params"] = params
    if "h1" in policy:
        engine_config["research_h1"] = policy["h1"]
        for filename in ("h1_backtest.py", "research.py", "accounting.py", "paper_risk.py"):
            source = base / "scripts" / filename
            if source.exists():
                code[f"scripts/{filename}"] = file_hash(source)
    hash_params = {k: params[k] for k in ("ladder", "sigma_target", "vol_window", "rebalance_band")}
    source_bytes = (base / matches[0]).read_bytes()
    digest = hashlib.sha256(source_bytes + json.dumps(
        hash_params, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    policy_hash = json_hash({"warmup": policy["warmup"], "splits": policy["splits"],
                             "S1": policy["snapshots"]["S1"]["hash"]})
    descriptor = {
        "hypothesis": policy["hypothesis"], "data_snapshot_hash": policy["snapshots"]["S1"]["hash"],
        "config_hash": digest, "params": params, "policy_hash": policy_hash, "run_verdict": verdict,
        "splits": policy["splits"], "strategy": strategy, "timeframe": timeframe,
        "freqai_model": freqai_model, "engine_config_hash": json_hash(engine_config),
        "code_hashes": code, "strategy_file": matches[0],
    }
    return descriptor, engine_config


def prepare_run(base, policy, descriptor, engine_config, trial_id, timerange, split):
    """Mount only this run's frozen prefix, never the live data or sealed results."""
    run_dir = base / "research_runs" / trial_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative, expected in descriptor["code_hashes"].items():
        data = (base / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Code changed while preparing run")
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    snapshot = policy["snapshots"]["S2" if split == "eval" else "S1"]
    manifest_path = Path(snapshot["manifest"])
    manifest = verify_snapshot(manifest_path, snapshot["hash"])
    expected_end = policy["splits"]["eval" if split == "eval" else "selection"][1]
    accepted_ends = {expected_end, policy["splits"]["dev"][1]} if split == "dev" else {expected_end}
    if manifest["window_start"] != policy["warmup"][0] or manifest["window_end"] not in accepted_ends:
        raise ValueError("Snapshot window does not match the configured split policy")
    if split == "eval":
        s1 = policy["snapshots"]["S1"]
        verify_prefix(s1["manifest"], s1["hash"], manifest_path, snapshot["hash"])
    end = date(timerange.split("-")[1])
    for pair, item in manifest["pairs"].items():
        data = (manifest_path.parent / item["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Snapshot data hash mismatch during staging")
        frame = csv_frame(data)
        frame = frame.loc[frame.date < end].copy()
        for column in COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column])
        target = run_dir / "data" / f"{pair.replace('-', '_')}-1d.feather"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_feather(target)
    (run_dir / "config.json").write_text(json.dumps(engine_config))
    (run_dir / "run.json").write_text(json.dumps(descriptor, sort_keys=True, indent=2))
    return run_dir


def slice_result(result, strategy, timerange, path_timerange, run_dir):
    """Slice the exported path, retaining entering positions and prior wallet state."""
    import zipfile
    start, end = map(date, timerange.split("-"))
    with zipfile.ZipFile(result["trades_export_path"]) as archive:
        summaries = []
        for name in archive.namelist():
            if name.endswith(".json"):
                payload = json.loads(archive.read(name))
                strategies = payload.get("strategy", {})
                if isinstance(strategies, dict) and strategy in strategies:
                    summaries.append(strategies[strategy])
        wallets = [n for n in archive.namelist() if n.endswith(f"_{strategy}_wallet.feather")]
        if len(summaries) != 1 or len(wallets) != 1:
            raise ValueError("A continuous-path result and wallet export are required")
        summary = summaries[0]
        wallet = pd.read_feather(io.BytesIO(archive.read(wallets[0])))
    wallet["date"] = pd.to_datetime(wallet["date"], utc=True)
    if not {"currency", "rate", "balance"} <= set(wallet):
        raise ValueError("Invalid wallet export")
    left, right = map(date, path_timerange.split("-"))
    if (pd.to_datetime(summary["backtest_start"], utc=True) != left or
            pd.to_datetime(summary["backtest_end"], utc=True) < right - timedelta(days=1)):
        raise ValueError("Engine shortened the requested continuous path")
    sliced = wallet.loc[(wallet.date >= start) & (wallet.date < end)]
    if sliced.empty or sliced.date.max() < end - timedelta(days=1):
        raise ValueError("Incomplete split wallet export")
    previous = wallet.loc[wallet.date < start]
    opening = previous.loc[previous.date == previous.date.max()]
    if start > left and opening.empty:
        raise ValueError("Missing wallet state at split boundary")
    trades = [t for t in summary["trades"] if pd.to_datetime(t["open_date"], utc=True) < end
              and pd.to_datetime(t["close_date"], utc=True) >= start]
    target = run_dir / "split.json"
    target.write_text(json.dumps({"timerange": timerange, "path_timerange": path_timerange,
                                 "trades": trades, "opening_wallet": opening.to_dict("records")},
                                default=str, sort_keys=True))
    sliced.reset_index(drop=True).to_feather(target.with_suffix(".feather"))
    # Wallet captures are pre-bar and defer open-entry fees, not daily closing NAV.
    from accounting import cost_rate, equity_curve, freqtrade_fills, metrics, reconcile_wallet
    config = json.loads((run_dir / "config.json").read_text())
    initial = config["dry_run_wallet"]
    fills = freqtrade_fills(summary["trades"], right - timedelta(days=1))
    if "research_h1" in config:
        # Minified exports group by trade, losing the H1 adapter's same-open sell-first ordering.
        fills.sort(key=lambda fill: (fill["date"], fill["side"] == "buy", fill["pair"]))
    marks = pd.DataFrame(index=pd.date_range(left, right - timedelta(days=1), freq="D"))
    for pair in config["exchange"]["pair_whitelist"]:
        frame = pd.read_feather(run_dir / "data" / f"{pair.replace('/', '_')}-1d.feather")
        marks[pair] = frame.set_index("date")["close"].reindex(marks.index)
    curve = equity_curve(initial, fills, marks, cost_rate=cost_rate(config["research_costs"]))
    reconcile_wallet(wallet, fills, config["exchange"]["pair_whitelist"], left, right)
    equity_path = run_dir / "net-equity.feather"
    curve.reset_index(names="date").to_feather(equity_path)
    keys = ("total_trades", "profit_total_pct", "profit_total_abs", "profit_avg_pct", "max_drawdown_pct",
            "max_drawdown_abs", "sharpe", "sortino", "profit_factor", "win_rate", "avg_duration",
            "backtest_days", "starting_balance")
    result["path_metrics"] = {key: result.pop(key) for key in keys if key in result}
    result.update(metrics(curve, initial, start))
    result.update(slice_path=str(target), path_timerange=path_timerange,
                  equity_path=str(equity_path), net_path_metrics=metrics(curve, initial, left),
                  cost_model=config["research_costs"],
                  total_trades=sum(start <= f["date"] < end for f in fills))
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Frozen snapshots and the shared trial ledger")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("snapshot", choices=("S1", "S2"))
    freeze.add_argument("--source", required=True, help="Directory of four OKX *-1d.feather files")
    freeze.add_argument("--destination", required=True, help="New directory; never overwrite a snapshot")
    commands.add_parser("budget")
    select = commands.add_parser("select")
    select.add_argument("trial_id")
    args = parser.parse_args(argv)
    try:
        if args.command == "budget":
            result = search_budget()
        elif args.command == "select":
            result = {"selected_trial_id": record_selection(args.trial_id)}
        else:
            policy = load_config()
            previous = None
            if args.snapshot == "S2":
                with ledger() as conn:
                    if not conn.execute("SELECT 1 FROM trial_ledger WHERE hypothesis=? AND verdict='selected'",
                                        (policy["hypothesis"],)).fetchone():
                        raise ValueError("Lock selection before freezing S2")
                s1 = policy["snapshots"]["S1"]
                previous = (s1["manifest"], s1["hash"])
            end = policy["splits"]["eval" if args.snapshot == "S2" else "selection"][1]
            manifest = freeze_snapshot(args.source, args.destination, policy["warmup"][0], end, previous=previous)
            result = {"manifest": str(manifest.resolve()),
                      "data_snapshot_hash": json.loads(manifest.read_text())["data_snapshot_hash"]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
