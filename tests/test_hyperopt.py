"""Tests for R4: hyperopt primitive + parser."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# A realistic-shape Freqtrade hyperopt stdout snippet. Built from actual
# Freqtrade output format with metric lines + the --print-json payload.
SAMPLE_STDOUT = """\
2026-05-17 19:00:00 - freqtrade - INFO - Starting hyperopt
... (many lines of progress) ...

  Epoch details:

  * Best result was reached in epoch 42/50:    105 trades. Avg profit   0.82%. Total profit  36.9%.
       Sharpe: 1.21  Sortino: 1.85  Profit factor: 1.32  Max Drawdown: 5.20%
       Trade duration: 12h 30m  Loss: -0.234

  Best result params (print-json):
  {"params": {"buy": {"rsi_oversold": 28, "bb_pct_threshold": 0.06}, "sell": {"rsi_overbought_exit": 72}}, "minimal_roi": {"0": 0.15, "60": 0.08}, "stoploss": -0.045}

Done.
"""


def test_parse_best_epoch_metrics():
    from backtest_runner import parse_hyperopt_output

    result = parse_hyperopt_output(SAMPLE_STDOUT, "FooStrategy", "20251101-20260501")
    assert result["success"] is True
    assert result["strategy"] == "FooStrategy"
    assert result["best_epoch"] == 42
    assert result["total_epochs"] == 50
    assert result["total_trades"] == 105
    assert result["profit_total_pct"] == 36.9
    assert result["sharpe"] == 1.21
    assert result["sortino"] == 1.85
    assert result["max_drawdown_pct"] == 5.20
    assert result["loss"] == -0.234


def test_parse_extracts_params_payload():
    from backtest_runner import parse_hyperopt_output

    result = parse_hyperopt_output(SAMPLE_STDOUT, "FooStrategy")
    assert "params" in result
    payload = result["params"]
    assert payload["params"]["buy"]["rsi_oversold"] == 28
    assert payload["params"]["sell"]["rsi_overbought_exit"] == 72
    assert payload["stoploss"] == -0.045


def test_parse_handles_missing_json_block():
    """If --print-json wasn't on or the JSON is malformed, the parser still
    returns the per-epoch metrics it can find — no exception."""
    from backtest_runner import parse_hyperopt_output

    truncated = SAMPLE_STDOUT.split("Best result params")[0]
    result = parse_hyperopt_output(truncated, "FooStrategy")
    assert result["success"] is True
    assert "params" not in result
    assert result["best_epoch"] == 42
    assert result["sharpe"] == 1.21


# ---------------------------------------------------------------------------
# run_hyperopt — mocked subprocess
# ---------------------------------------------------------------------------

def test_run_hyperopt_success_path():
    from backtest_runner import run_hyperopt

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = SAMPLE_STDOUT
    mock_proc.stderr = ""

    with patch("backtest_runner.subprocess.run", return_value=mock_proc) as mock_run:
        result = run_hyperopt(
            "FooStrategy", timerange="20251101-20260501", epochs=50,
        )

    assert result["success"] is True
    assert result["best_epoch"] == 42
    assert result["total_trades"] == 105

    # Verify the subprocess command included the right freqtrade hyperopt flags
    cmd = mock_run.call_args.args[0]
    assert "hyperopt" in cmd
    assert "--strategy" in cmd
    assert "FooStrategy" in cmd
    assert "--epochs" in cmd
    assert "50" in cmd
    assert "--hyperopt-loss" in cmd
    assert "SampleHyperOptLoss" in cmd
    assert "--print-json" in cmd
    # Default spaces
    assert "buy" in cmd
    assert "sell" in cmd
    # Timerange was forwarded
    assert "20251101-20260501" in cmd


def test_run_hyperopt_returns_failure_on_nonzero_exit():
    from backtest_runner import run_hyperopt

    mock_proc = MagicMock()
    mock_proc.returncode = 2
    mock_proc.stdout = ""
    mock_proc.stderr = "freqtrade: error: strategy not found"

    with patch("backtest_runner.subprocess.run", return_value=mock_proc):
        result = run_hyperopt("MissingStrategy", timerange="20251101-20260501")

    assert result["success"] is False
    assert "exited with code 2" in result["error"]
    assert "strategy not found" in result["raw_output"]


def test_run_hyperopt_returns_failure_on_timeout():
    import subprocess as sp
    from backtest_runner import run_hyperopt

    def raise_timeout(*a, **kw):
        raise sp.TimeoutExpired(cmd="docker", timeout=60)

    with patch("backtest_runner.subprocess.run", side_effect=raise_timeout):
        result = run_hyperopt("FooStrategy", timeout_seconds=60)

    assert result["success"] is False
    assert "timed out" in result["error"]


def test_run_hyperopt_custom_spaces_and_loss():
    from backtest_runner import run_hyperopt

    mock_proc = MagicMock(returncode=0, stdout=SAMPLE_STDOUT, stderr="")

    with patch("backtest_runner.subprocess.run", return_value=mock_proc) as mock_run:
        run_hyperopt(
            "FooStrategy",
            spaces=("buy", "sell", "roi", "stoploss"),
            loss="CustomLoss",
        )

    cmd = mock_run.call_args.args[0]
    assert "CustomLoss" in cmd
    for s in ("buy", "sell", "roi", "stoploss"):
        assert s in cmd
