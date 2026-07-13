"""
Backtest Runner — Automated Freqtrade backtesting wrapper.

Two-stage evaluation:
  1. Mini-backtest (30 days) — quick filter, rejects obviously broken strategies
  2. Full backtest (6+ months) — proper evaluation for candidates that pass stage 1

Uses the sandboxed backtest container (no network, resource limits) for safety
when testing LLM-generated strategies.

Results are parsed and returned as structured dicts for the strategy registry.

Result parsing has two paths:

  * ``parse_backtest_artifact`` reads the strongly-typed JSON Freqtrade writes
    into its export zip. Preferred when ``export_trades=True``.
  * ``parse_backtest_output`` scrapes the console table as a fallback for
    runs that didn't request an export (e.g. mini-backtests).

The console scraper has been the source of two silent-zero bugs (the
``Tot Profit %`` column-header collision fix in PR #30 most recently). The
artifact path is far more robust because the values come typed from
Freqtrade itself, not regex-extracted from a Rich table layout.
"""

import json
import logging
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("backtest_runner")

BASE_DIR = Path(__file__).resolve().parent.parent  # user_data/
PROJECT_ROOT = BASE_DIR.parent

# When running inside Docker, volume paths in docker-compose.yml are relative
# to the host project directory, not the container's filesystem.
# The compose file is mounted at /app/docker-compose.yml; we tell compose
# to resolve relative paths against the host project dir via --project-directory.
HOST_PROJECT_DIR = os.environ.get("HOST_PROJECT_DIR", str(PROJECT_ROOT))
RESULTS_DIR = BASE_DIR / "backtest_results"


def run_backtest(
    strategy_name: str,
    timeframe: str = "1h",
    timerange: str = None,
    config_path: str = None,
    use_sandbox: bool = True,
    timeout_seconds: int = 300,
    export_trades: bool = False,
    freqai_model: str = None,
) -> dict:
    """
    Run a Freqtrade backtest via Docker and return parsed results.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy class to backtest.
    timeframe : str
        Candle timeframe (default "1h").
    timerange : str
        Freqtrade timerange string (e.g., "20250701-20260101").
        If None, backtests all available data.
    config_path : str
        Path to config file inside the container.
    use_sandbox : bool
        If True, use the sandboxed container (no network, resource limits).
    timeout_seconds : int
        Max time for the backtest to complete.
    export_trades : bool
        If True, write the per-trade list to a deterministic path inside
        user_data/backtest_results/ and return its host path under
        result['trades_export_path']. Used by R2d trade attribution.
    freqai_model : str
        FreqAI model class (e.g. "LightGBMRegressor"). When set, the run
        uses the freqtrade-freqai service (stable_freqai image — ships the
        ML deps and higher resource limits) and passes --freqaimodel.
        FreqAI backtests REQUIRE an explicit timerange (training windows
        slide across it), so timerange=None is rejected up front. Callers
        should also pass a per-candidate config (freqai block + unique
        identifier) and a much larger timeout — the run retrains once per
        backtest_period_days per pair.

    Returns
    -------
    dict with keys:
        - success: bool
        - strategy: str
        - timerange: str
        - total_trades: int
        - profit_total_pct: float
        - profit_total_abs: float
        - max_drawdown_pct: float
        - max_drawdown_abs: float
        - sharpe: float
        - sortino: float
        - profit_factor: float
        - win_rate: float
        - avg_duration: str
        - backtest_days: int
        - starting_balance: float
        - raw_output: str (last 50 lines)
        - trades_export_path: str (host path to .zip; only if export_trades=True)
        - error: str (if failed)
    """
    if config_path is None:
        config_path = "/freqtrade/user_data/config.json"

    if freqai_model and not timerange:
        return {
            "success": False,
            "strategy": strategy_name,
            "error": "FreqAI backtests require an explicit timerange "
                     "(the training window slides across it)",
        }

    # Compose file is mounted at /app/docker-compose.yml inside the orchestrator.
    # --project-directory must point to the HOST path so volume mounts resolve correctly.
    compose_file = str(PROJECT_ROOT / "docker-compose.yml")
    cmd = ["docker", "compose",
           "-f", compose_file,
           "--project-directory", HOST_PROJECT_DIR]

    if use_sandbox or freqai_model:
        # Use the sandboxed profile (--profile must come before 'run')
        cmd.extend(["--profile", "backtest"])

    cmd.extend(["run", "--rm"])

    if freqai_model:
        cmd.append("freqtrade-freqai")
    elif use_sandbox:
        cmd.append("freqtrade-backtest")
    else:
        cmd.append("freqtrade-sweep")  # Use any running instance

    cmd.extend([
        "backtesting",
        "--strategy", strategy_name,
        "--strategy-path", "/freqtrade/user_data/strategies/candidates",
        "--timeframe", timeframe,
        "--config", config_path,
    ])

    if freqai_model:
        cmd.extend(["--freqaimodel", freqai_model])

    # R2d: per-trade attribution needs Freqtrade's trade export. Recent
    # Freqtrade deprecated --export-filename — we now pass an isolated
    # per-call subdirectory via --backtest-directory and let FT auto-name
    # the file inside it. Avoids collisions between concurrent backtests
    # and lets us locate the .zip without ambiguity afterwards.
    export_host_dir: Path | None = None
    if export_trades:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        subdir_name = f"{strategy_name}-{run_id}"
        export_host_dir = BASE_DIR / "backtest_results" / subdir_name
        export_host_dir.mkdir(parents=True, exist_ok=True)
        export_container_dir = f"/freqtrade/user_data/backtest_results/{subdir_name}"
        cmd.extend([
            "--export", "trades",
            "--backtest-directory", export_container_dir,
        ])
    else:
        cmd.extend(["--export", "none"])

    if timerange:
        cmd.extend(["--timerange", timerange])

    log.info(f"Running backtest: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(PROJECT_ROOT),
        )

        output = proc.stdout + proc.stderr
        last_lines = "\n".join(output.strip().split("\n")[-50:])

        if proc.returncode != 0:
            return {
                "success": False,
                "strategy": strategy_name,
                "error": f"Backtest exited with code {proc.returncode}",
                "raw_output": last_lines,
            }

        # Parse results. Prefer the JSON artifact when an export was
        # requested — it ships typed values straight from Freqtrade and
        # sidesteps the entire console-regex bug class. Fall back to
        # console parsing if the artifact is missing/corrupt OR if no
        # export was requested (the mini-backtest path).
        result = None
        if export_host_dir is not None:
            zips = sorted(export_host_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
            if zips:
                zip_path = zips[-1]
                try:
                    result = parse_backtest_artifact(zip_path, strategy_name, timerange)
                    result["trades_export_path"] = str(zip_path)
                    # Preserve the tail of console output for debugging — the
                    # artifact JSON has the metrics, but logs/warnings live in
                    # stdout/stderr only.
                    result.setdefault("raw_output", last_lines)
                except Exception as e:
                    log.warning(
                        f"artifact parse failed for {strategy_name} "
                        f"({type(e).__name__}: {e}); falling back to console parse"
                    )
                    result = None
            else:
                log.warning(f"export requested but no zip in {export_host_dir}")

        if result is None:
            result = parse_backtest_output(output, strategy_name, timerange)
            if export_host_dir is not None:
                zips = sorted(export_host_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
                if zips:
                    result["trades_export_path"] = str(zips[-1])

        return result

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "strategy": strategy_name,
            "error": f"Backtest timed out after {timeout_seconds}s",
        }
    except Exception as e:
        return {
            "success": False,
            "strategy": strategy_name,
            "error": str(e),
        }


def parse_backtest_artifact(
    zip_path: Path | str,
    strategy_name: str,
    timerange: str | None = None,
) -> dict:
    """Parse Freqtrade's exported backtest JSON inside its result zip.

    Freqtrade writes ``backtest-result-<ts>.json`` into the zip alongside
    the trades feather. That JSON contains every metric we previously
    scraped from the console table — but typed, in canonical units,
    straight from Freqtrade's own writer. No regex layer to drift when
    a column header changes upstream.

    The returned dict has the same shape as ``parse_backtest_output``
    so downstream consumers (orchestrator, strategy_registry,
    strategy_generator's diagnostic) don't need to branch.

    Raises if the zip is missing, corrupted, or the JSON doesn't contain
    the expected strategy entry. The caller is expected to catch and
    fall back to console parsing.
    """
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        json_names = [
            n for n in zf.namelist()
            if n.endswith(".json") and "_config" not in n
        ]
        if not json_names:
            raise ValueError(f"no result JSON inside {zip_path.name}")
        with zf.open(json_names[0]) as fh:
            data = json.load(fh)

    strategies = data.get("strategy", {})
    if strategy_name not in strategies:
        # Freqtrade names the entry by the class, which is what we passed
        # as --strategy. If it's missing, the run probably failed to load
        # or registered under a different name — surface that explicitly.
        raise ValueError(
            f"strategy {strategy_name!r} not in artifact (found: {list(strategies)})"
        )
    s = strategies[strategy_name]

    # Freqtrade stores most ratios as decimals (0.0188 = 1.88%, 0.74 = 74%).
    # Our pre-existing API exposes these as percentages — keep that contract
    # so the registry / orchestrator / generator diagnostics don't see a
    # silent 100× change when the parser path switches.
    def _pct(decimal_value):
        return float(decimal_value) * 100.0 if decimal_value is not None else 0.0

    # Drawdown: prefer max_drawdown_account (true equity-curve drawdown);
    # fall back to max_relative_drawdown if Freqtrade ever stops emitting
    # the first one. Both are decimals.
    dd_decimal = s.get("max_drawdown_account")
    if dd_decimal is None:
        dd_decimal = s.get("max_relative_drawdown", 0.0)

    return {
        "success": True,
        "strategy": strategy_name,
        "timerange": timerange or s.get("timerange") or "all",
        "total_trades": int(s.get("total_trades", 0)),
        "profit_total_pct": _pct(s.get("profit_total")),
        "profit_total_abs": float(s.get("profit_total_abs", 0.0)),
        "profit_avg_pct": _pct(s.get("profit_mean")),
        "max_drawdown_pct": _pct(dd_decimal),
        "max_drawdown_abs": float(s.get("max_drawdown_abs", 0.0)),
        "sharpe": float(s.get("sharpe", 0.0)),
        "sortino": float(s.get("sortino", 0.0)),
        "profit_factor": float(s.get("profit_factor", 0.0)),
        # winrate is decimal in JSON (0.74), pct elsewhere (74)
        "win_rate": _pct(s.get("winrate")),
        "avg_duration": s.get("holding_avg", ""),
        "backtest_days": int(s.get("backtest_days", 0)),
        "starting_balance": float(s.get("starting_balance", 0.0)),
        # End of the data this backtest ran on. Distinct from when the
        # backtest was invoked (created_at on the registry row) — see
        # docs/deployment-lifecycle.md for why this matters: a "fresh
        # backtest" run yesterday on stale candles is not the same as
        # one run against current data. The deployment-eligibility
        # filter checks both.
        "backtest_data_end_at": s.get("backtest_end", ""),
    }


def parse_backtest_output(output: str, strategy_name: str, timerange: str = None) -> dict:
    """Parse Freqtrade backtest console output into structured results."""
    result = {
        "success": True,
        "strategy": strategy_name,
        "timerange": timerange or "all",
    }

    def extract_value(pattern: str, text: str, default=None, cast=float):
        match = re.search(pattern, text)
        if match:
            try:
                return cast(match.group(1).strip())
            except (ValueError, TypeError):
                return default
        return default

    import re

    # Try to parse from the STRATEGY SUMMARY line
    # Format: │ StrategyName │ N │ X.XX │ Y.YY USDT │ ...
    summary_pattern = (
        rf"│\s*{re.escape(strategy_name)}\s*│\s*(\d+)\s*│\s*([-\d.]+)\s*│\s*([-\d.]+)"
    )
    summary_match = re.search(summary_pattern, output)
    if summary_match:
        result["total_trades"] = int(summary_match.group(1))
        result["profit_avg_pct"] = float(summary_match.group(2))
        result["profit_total_abs"] = float(summary_match.group(3))
    else:
        # Fallback: look for total trades in the TOTAL row
        result["total_trades"] = extract_value(
            r"│\s*TOTAL\s*│\s*(\d+)\s*│", output, default=0, cast=int
        )

    # Profit. The summary table at the top uses ABBREVIATED column headers
    # ("Tot Profit %", "Tot Profit USDT") — those strings appear in the
    # output but as table headers, not data rows, so matching them grabs
    # the *next column header* and the regex group capture fails silently
    # → default 0.0 → every strategy looks "unprofitable".
    # The summary metrics table at the bottom uses different labels:
    # "Total profit %" and "Absolute profit". Those are the data rows.
    result["profit_total_pct"] = extract_value(
        r"Total profit %\s*│\s*([-\d.]+)", output, default=0.0
    )
    result["profit_total_abs"] = result.get("profit_total_abs") or extract_value(
        r"Absolute profit\s*│\s*([-\d.]+)", output, default=0.0
    )

    # Drawdown
    result["max_drawdown_pct"] = extract_value(
        r"Max % of account underwater\s*│\s*([-\d.]+)", output, default=0.0
    )
    result["max_drawdown_abs"] = extract_value(
        r"Absolute drawdown\s*│\s*([-\d.]+)", output, default=0.0
    )

    # Risk metrics. Freqtrade 2026.x labels these rows "Sharpe (closed
    # trades)" / "Sortino (closed trades)" — the bare "Sharpe │" pattern
    # stopped matching and silently returned 0.0 for every walk-forward
    # window (the third silent-zero scraper bug; found during the issue #47
    # shakedown). Tolerate an optional parenthetical suffix; the first
    # matching row is the closed-trades metric, same source the JSON
    # artifact path reports.
    result["sharpe"] = extract_value(
        r"Sharpe(?:\s*\([^)]*\))?\s*│\s*([-\d.]+)", output, default=0.0
    )
    result["sortino"] = extract_value(
        r"Sortino(?:\s*\([^)]*\))?\s*│\s*([-\d.]+)", output, default=0.0
    )
    result["profit_factor"] = extract_value(
        r"Profit factor\s*│\s*([-\d.]+)", output, default=0.0
    )

    # Win rate from STRATEGY SUMMARY — last column shows "Win  Draw  Loss  Win%"
    # Match the Win% value at the end of the strategy summary row
    win_pattern = rf"{re.escape(strategy_name)}.*?(\d+(?:\.\d+)?)\s*│\s*[\d.]+\s*(?:USDT|%)?\s*│?\s*$"
    win_match = re.search(win_pattern, output, re.MULTILINE)
    if win_match:
        result["win_rate"] = float(win_match.group(1))
    else:
        # Fallback: look for "Win%" pattern anywhere
        result["win_rate"] = extract_value(
            r"(\d+(?:\.\d+)?)\s*│\s*[\d.]+\s*(?:USDT|%)", output, default=0.0
        )

    # Duration
    duration_match = re.search(r"Avg Duration\s*│\s*(.+?)│", output)
    if duration_match:
        result["avg_duration"] = duration_match.group(1).strip()

    # Starting balance
    result["starting_balance"] = extract_value(
        r"dry_run_wallet.*?(\d+)", output, default=1000.0
    )

    # Backtest period
    period_match = re.search(r"Backtested\s+([\d-]+\s[\d:]+)\s*->\s*([\d-]+\s[\d:]+)", output)
    if period_match:
        try:
            start = datetime.strptime(period_match.group(1), "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(period_match.group(2), "%Y-%m-%d %H:%M:%S")
            result["backtest_days"] = (end - start).days
            # Capture the end timestamp so eligibility can distinguish
            # "ran a backtest yesterday" from "ran a backtest on stale
            # candles". JSON-artifact path emits the same field name
            # populated from the typed source — see parse_backtest_artifact.
            result["backtest_data_end_at"] = period_match.group(2)
        except ValueError:
            result["backtest_days"] = 0
    else:
        result["backtest_days"] = 0

    result["raw_output"] = "\n".join(output.strip().split("\n")[-30:])

    return result


def run_mini_backtest(
    strategy_name: str,
    days: int = 90,
    skip_recent_days: int = 30,
    **kwargs,
) -> dict:
    """Stage 1: out-of-sample backtest to filter overfit strategies.

    Window: [today - skip_recent_days - days, today - skip_recent_days].
    Default = [today - 120, today - 30], a 90-day slice ending one month
    before the full backtest's most-recent data.

    Why out-of-sample. The orchestrator's full backtest uses all available
    data (~200 days). When mini also covered the most recent 90 days,
    strategies overfit to that slice scored well in mini and then blew up
    in full — observed in trials #5 and #6 (e.g. cell 19 mini sharpe 1.91,
    full sharpe -2.74). Skipping the most recent 30 days keeps that slice
    out of the mini's view, so a strategy that wins both runs has actually
    generalized across two non-overlapping regimes (in expectation) rather
    than just memorized one.

    90-day length is preserved because the rendered strategy declares
    startup_candle_count=200; a 30-day window leaves too little prelude
    and Freqtrade exits with "no data left after adjusting for startup
    candles". 90 days = 2160 1h candles, comfortable headroom.
    """
    end = datetime.now(timezone.utc) - timedelta(days=skip_recent_days)
    start = end - timedelta(days=days)
    timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

    log.info(f"Mini-backtest: {strategy_name} ({timerange})")
    return run_backtest(strategy_name, timerange=timerange, **kwargs)


def run_full_backtest(strategy_name: str, months: int = 6, **kwargs) -> dict:
    """Stage 2: Full 6-month backtest for serious evaluation."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)
    timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"

    log.info(f"Full backtest: {strategy_name} ({timerange})")
    return run_backtest(strategy_name, timerange=timerange, **kwargs)


def evaluate_candidate(strategy_name: str) -> dict:
    """
    Two-stage evaluation of a candidate strategy.

    Stage 1: Mini-backtest (30 days) — must produce trades and not crash.
    Stage 2: Full backtest (6 months) — evaluated on Sharpe, drawdown, profit factor.

    Returns dict with stage results and overall verdict.
    """
    log.info(f"=== Evaluating candidate: {strategy_name} ===")

    # Import validation
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from validation_pipeline import validate_backtest_results

    # Stage 1: Mini-backtest
    mini = run_mini_backtest(strategy_name)
    if not mini.get("success"):
        return {
            "strategy": strategy_name,
            "verdict": "FAIL_MINI",
            "reason": mini.get("error", "Mini-backtest failed"),
            "mini_result": mini,
        }

    # Check mini-backtest sanity
    mini_validation = validate_backtest_results(mini)
    if not mini_validation.passed:
        return {
            "strategy": strategy_name,
            "verdict": "FAIL_SANITY",
            "reason": str(mini_validation),
            "mini_result": mini,
        }

    # Stage 2: Full backtest
    full = run_full_backtest(strategy_name)
    if not full.get("success"):
        return {
            "strategy": strategy_name,
            "verdict": "FAIL_FULL",
            "reason": full.get("error", "Full backtest failed"),
            "mini_result": mini,
            "full_result": full,
        }

    full_validation = validate_backtest_results(full)
    if not full_validation.passed:
        return {
            "strategy": strategy_name,
            "verdict": "FAIL_SANITY",
            "reason": str(full_validation),
            "mini_result": mini,
            "full_result": full,
        }

    return {
        "strategy": strategy_name,
        "verdict": "PASS",
        "mini_result": mini,
        "full_result": full,
    }


# ---------------------------------------------------------------------------
# Hyperopt — parameter search via Freqtrade's IHyperOptLoss
# ---------------------------------------------------------------------------

def run_hyperopt(
    strategy_name: str,
    timeframe: str = "1h",
    timerange: str = None,
    epochs: int = 50,
    spaces: tuple = ("buy", "sell"),
    loss: str = "SampleHyperOptLoss",
    config_path: str = None,
    use_sandbox: bool = True,
    timeout_seconds: int = 1800,
) -> dict:
    """Run Freqtrade hyperopt via Docker and return parsed best-epoch results.

    Mirror of run_backtest but for parameter search. The strategy must already
    use IntParameter/DecimalParameter etc. — base_generated.py enforces this
    in generated strategies. Hyperopt writes its result file to
    user_data/hyperopt_results/ and also auto-loads the best params on the
    NEXT backtest run of the same strategy (via the <StrategyName>.json
    Freqtrade auto-discovers next to the .py file when --hyperopt-export is on).

    Returns a dict with the same shape as run_backtest plus:
      - best_epoch: int               which epoch won
      - total_epochs: int             how many were actually run
      - params: dict                  best-epoch buy/sell/roi/stoploss params
      - loss: float                   loss function value at best epoch
    """
    if config_path is None:
        config_path = "/freqtrade/user_data/config.json"

    compose_file = str(PROJECT_ROOT / "docker-compose.yml")
    cmd = ["docker", "compose",
           "-f", compose_file,
           "--project-directory", HOST_PROJECT_DIR]
    if use_sandbox:
        cmd.extend(["--profile", "backtest"])
    cmd.extend(["run", "--rm"])
    cmd.append("freqtrade-backtest" if use_sandbox else "freqtrade-sweep")

    cmd.extend([
        "hyperopt",
        "--strategy", strategy_name,
        "--strategy-path", "/freqtrade/user_data/strategies/candidates",
        "--hyperopt-path", "/freqtrade/user_data/hyperopts",
        "--hyperopt-loss", loss,
        "--spaces", *spaces,
        "--epochs", str(epochs),
        "--timeframe", timeframe,
        "--config", config_path,
        # auto-write <StrategyName>.json next to the .py file so the next
        # backtest run picks up the optimized params transparently
        "--print-json",
    ])
    if timerange:
        cmd.extend(["--timerange", timerange])

    log.info(f"Running hyperopt ({epochs} epochs): {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(PROJECT_ROOT),
        )
        output = proc.stdout + proc.stderr
        last_lines = "\n".join(output.strip().split("\n")[-100:])

        if proc.returncode != 0:
            return {
                "success": False,
                "strategy": strategy_name,
                "error": f"Hyperopt exited with code {proc.returncode}",
                "raw_output": last_lines,
            }

        return parse_hyperopt_output(output, strategy_name, timerange, epochs)

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "strategy": strategy_name,
            "error": f"Hyperopt timed out after {timeout_seconds}s",
        }
    except Exception as e:
        return {
            "success": False,
            "strategy": strategy_name,
            "error": str(e),
        }


def parse_hyperopt_output(
    output: str, strategy_name: str, timerange: str = None, epochs: int = 0
) -> dict:
    """Extract best-epoch metrics + params from Freqtrade hyperopt stdout.

    Freqtrade's --print-json flag emits a one-line JSON block with the best
    epoch's params — that's the most reliable thing to parse. Per-epoch
    summary numbers are parsed from the conventional "Best result" section.

    Returns the same dict shape as parse_backtest_output, plus best_epoch,
    total_epochs, params, loss.
    """
    import re

    result = {
        "success": True,
        "strategy": strategy_name,
        "timerange": timerange or "",
        "total_epochs": epochs,
        "raw_output": "\n".join(output.strip().split("\n")[-100:]),
    }

    # --print-json emits a single-line JSON payload of the winning params.
    # Format: { "params": {...}, "minimal_roi": {...}, "stoploss": ... }
    # Scan lines instead of regex — nested braces don't matter line-by-line.
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or '"params"' not in stripped:
            continue
        try:
            result["params"] = json.loads(stripped)
            break
        except json.JSONDecodeError:
            continue
    else:
        log.debug("No --print-json payload found in hyperopt output.")

    # "Best result was reached in epoch N/T"
    best_match = re.search(
        r"Best result.*?epoch\s+(\d+)\s*/\s*(\d+)", output, re.IGNORECASE | re.DOTALL
    )
    if best_match:
        result["best_epoch"] = int(best_match.group(1))
        result["total_epochs"] = int(best_match.group(2))

    # Common per-epoch metrics — same regex shapes as backtest output
    for key, pattern in [
        ("total_trades", r"(\d+)\s+trades"),
        ("profit_total_pct", r"Total profit.*?([-\d.]+)\s*%"),
        ("sharpe", r"Sharpe:\s*([-\d.]+)"),
        ("sortino", r"Sortino:\s*([-\d.]+)"),
        ("profit_factor", r"Profit factor:\s*([-\d.]+)"),
        ("max_drawdown_pct", r"(?:Max\s+)?[Dd]rawdown.*?([-\d.]+)\s*%"),
        ("loss", r"\bLoss:\s*([-\d.]+)"),
    ]:
        m = re.search(pattern, output)
        if m:
            val = m.group(1)
            try:
                result[key] = float(val) if "." in val or key in ("sharpe", "sortino", "profit_total_pct", "max_drawdown_pct", "profit_factor", "loss") else int(val)
            except ValueError:
                pass

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Run Freqtrade backtests or hyperopt")
    parser.add_argument("strategy", help="Strategy class name")
    parser.add_argument("--mini", action="store_true", help="Run mini-backtest only (30 days)")
    parser.add_argument("--full", action="store_true", help="Run full backtest only (6 months)")
    parser.add_argument("--evaluate", action="store_true", help="Run full 2-stage evaluation")
    parser.add_argument("--hyperopt", action="store_true", help="Run hyperopt parameter search")
    parser.add_argument("--epochs", type=int, default=50, help="Hyperopt epochs (default 50)")
    parser.add_argument(
        "--timerange",
        default=None,
        help="Freqtrade timerange (e.g. 20251101-20260501). "
             "Defaults to last ~6 months for hyperopt.",
    )
    args = parser.parse_args()

    if args.hyperopt:
        timerange = args.timerange
        if timerange is None:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=180)
            timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
        result = run_hyperopt(args.strategy, timerange=timerange, epochs=args.epochs)
    elif args.evaluate:
        result = evaluate_candidate(args.strategy)
    elif args.mini:
        result = run_mini_backtest(args.strategy)
    elif args.full:
        result = run_full_backtest(args.strategy)
    else:
        result = run_backtest(args.strategy)

    # Print results (exclude raw_output for readability)
    display = {k: v for k, v in result.items() if k != "raw_output"}
    print(json.dumps(display, indent=2, default=str))
