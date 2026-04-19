"""
Backtest Runner — Automated Freqtrade backtesting wrapper.

Two-stage evaluation:
  1. Mini-backtest (30 days) — quick filter, rejects obviously broken strategies
  2. Full backtest (6+ months) — proper evaluation for candidates that pass stage 1

Uses the sandboxed backtest container (no network, resource limits) for safety
when testing LLM-generated strategies.

Results are parsed and returned as structured dicts for the strategy registry.
"""

import json
import logging
import os
import subprocess
import sys
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
        - error: str (if failed)
    """
    if config_path is None:
        config_path = "/freqtrade/user_data/config.json"

    # Compose file is mounted at /app/docker-compose.yml inside the orchestrator.
    # --project-directory must point to the HOST path so volume mounts resolve correctly.
    compose_file = str(PROJECT_ROOT / "docker-compose.yml")
    cmd = ["docker", "compose",
           "-f", compose_file,
           "--project-directory", HOST_PROJECT_DIR]

    if use_sandbox:
        # Use the sandboxed profile (--profile must come before 'run')
        cmd.extend(["--profile", "backtest"])

    cmd.extend(["run", "--rm"])

    if use_sandbox:
        cmd.append("freqtrade-backtest")
    else:
        cmd.append("freqtrade-sweep")  # Use any running instance

    cmd.extend([
        "backtesting",
        "--strategy", strategy_name,
        "--strategy-path", "/freqtrade/user_data/strategies/candidates",
        "--timeframe", timeframe,
        "--config", config_path,
        "--export", "none",  # Don't export trade details to save disk
    ])

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

        # Parse results from output
        return parse_backtest_output(output, strategy_name, timerange)

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

    # Profit
    result["profit_total_pct"] = extract_value(
        r"Tot Profit %\s*│\s*([-\d.]+)", output, default=0.0
    )
    result["profit_total_abs"] = result.get("profit_total_abs") or extract_value(
        r"Tot Profit USDT\s*│\s*([-\d.]+)", output, default=0.0
    )

    # Drawdown
    result["max_drawdown_pct"] = extract_value(
        r"Max % of account underwater\s*│\s*([-\d.]+)", output, default=0.0
    )
    result["max_drawdown_abs"] = extract_value(
        r"Absolute drawdown\s*│\s*([-\d.]+)", output, default=0.0
    )

    # Risk metrics
    result["sharpe"] = extract_value(r"Sharpe\s*│\s*([-\d.]+)", output, default=0.0)
    result["sortino"] = extract_value(r"Sortino\s*│\s*([-\d.]+)", output, default=0.0)
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
        except ValueError:
            result["backtest_days"] = 0
    else:
        result["backtest_days"] = 0

    result["raw_output"] = "\n".join(output.strip().split("\n")[-30:])

    return result


def run_mini_backtest(strategy_name: str, days: int = 30, **kwargs) -> dict:
    """Stage 1: Quick 30-day backtest to filter obviously broken strategies."""
    end = datetime.now(timezone.utc)
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
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Run Freqtrade backtests")
    parser.add_argument("strategy", help="Strategy class name")
    parser.add_argument("--mini", action="store_true", help="Run mini-backtest only (30 days)")
    parser.add_argument("--full", action="store_true", help="Run full backtest only (6 months)")
    parser.add_argument("--evaluate", action="store_true", help="Run full 2-stage evaluation")
    args = parser.parse_args()

    if args.evaluate:
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
