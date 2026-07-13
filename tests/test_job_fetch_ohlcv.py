"""Tests for orchestrator.job_fetch_ohlcv — weekly OKX OHLCV refresh.

Background: pipeline silently produced 0-trade strategies because OKX feathers
were 5 weeks stale, truncating backtests below the walk-forward window. This
job refreshes them weekly before the Saturday generation cron.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))

import orchestrator


@pytest.fixture
def cfg_with_pairs(tmp_path, monkeypatch):
    """Point BASE_DIR at a temp dir holding a minimal config.json."""
    cfg = {
        "exchange": {"name": "okx", "pair_whitelist": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
        "timeframe": "1h",
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(orchestrator, "BASE_DIR", tmp_path)
    return tmp_path


def test_fetch_ohlcv_builds_correct_docker_command(cfg_with_pairs):
    with patch("orchestrator.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        orchestrator.job_fetch_ohlcv()

    assert mock_run.call_count == 1
    cmd = mock_run.call_args[0][0]
    assert cmd[0:2] == ["docker", "compose"]
    assert "--profile" in cmd and cmd[cmd.index("--profile") + 1] == "backtest"
    assert "run" in cmd and "--rm" in cmd
    assert "freqtrade-backtest" in cmd
    assert "download-data" in cmd
    # All configured pairs propagated
    assert "BTC/USDT" in cmd and "ETH/USDT" in cmd and "SOL/USDT" in cmd
    # 400 days: 180 (longest backtest/walk-forward window) + 180 (max
    # freqai train_period_days of pre-window training data) + slack
    assert "--days" in cmd and cmd[cmd.index("--days") + 1] == "400"
    assert "--timeframes" in cmd and cmd[cmd.index("--timeframes") + 1] == "1h"


def _write_freqai_config(base_dir: Path, pairs: list):
    cfg_dir = base_dir / "configs"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config-freqai-base.json").write_text(
        json.dumps({"exchange": {"pair_whitelist": pairs}})
    )


def test_fetch_ohlcv_unions_freqai_config_pairs(cfg_with_pairs):
    """A pair present only in config-freqai-base.json must still be
    downloaded — otherwise FreqAI backtests reference data the refresh
    never fetches (issue #47 review follow-up)."""
    _write_freqai_config(cfg_with_pairs, ["BTC/USDT", "DOGE/USDT"])
    with patch.object(orchestrator.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        orchestrator.job_fetch_ohlcv()
    cmd = mock_run.call_args[0][0]
    assert "DOGE/USDT" in cmd
    # No duplicate for the shared pair
    assert cmd.count("BTC/USDT") == 1
    # config.json ordering preserved, union appended
    assert cmd.index("BTC/USDT") < cmd.index("DOGE/USDT")


def test_fetch_ohlcv_works_without_freqai_config(cfg_with_pairs):
    """The freqai base config is optional — its absence must not block
    the refresh of the main pairs."""
    with patch.object(orchestrator.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        orchestrator.job_fetch_ohlcv()
    cmd = mock_run.call_args[0][0]
    assert "BTC/USDT" in cmd and "SOL/USDT" in cmd


def test_fetch_ohlcv_survives_corrupt_freqai_config(cfg_with_pairs):
    """A malformed freqai config is logged, never fatal — the main
    refresh must still run."""
    cfg_dir = cfg_with_pairs / "configs"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "config-freqai-base.json").write_text("{not json")
    with patch.object(orchestrator.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        orchestrator.job_fetch_ohlcv()
    cmd = mock_run.call_args[0][0]
    assert "BTC/USDT" in cmd


def test_fetch_ohlcv_handles_missing_config(tmp_path, monkeypatch):
    """No config.json → log error, do not raise (don't crash orchestrator)."""
    monkeypatch.setattr(orchestrator, "BASE_DIR", tmp_path)
    with patch("orchestrator.subprocess.run") as mock_run:
        orchestrator.job_fetch_ohlcv()  # should not raise
    assert mock_run.call_count == 0


def test_fetch_ohlcv_handles_empty_pair_whitelist(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"exchange": {"pair_whitelist": []}}))
    monkeypatch.setattr(orchestrator, "BASE_DIR", tmp_path)
    with patch("orchestrator.subprocess.run") as mock_run:
        orchestrator.job_fetch_ohlcv()
    assert mock_run.call_count == 0


def test_fetch_ohlcv_nonzero_exit_does_not_raise(cfg_with_pairs):
    """Network/exchange outage → log error, swallow (next weekly run retries)."""
    with patch("orchestrator.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="OKX API down")
        orchestrator.job_fetch_ohlcv()  # should not raise
    assert mock_run.called


def test_fetch_ohlcv_timeout_does_not_raise(cfg_with_pairs):
    import subprocess
    with patch("orchestrator.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=600)
        orchestrator.job_fetch_ohlcv()  # should not raise


def test_fetch_ohlcv_uses_host_project_dir_env(cfg_with_pairs, monkeypatch):
    """HOST_PROJECT_DIR env var must be passed via --project-directory so
    docker compose volume mounts resolve correctly when invoked from inside
    the orchestrator container."""
    monkeypatch.setenv("HOST_PROJECT_DIR", "/host/path/to/repo")
    with patch("orchestrator.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        orchestrator.job_fetch_ohlcv()
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--project-directory")
    assert cmd[idx + 1] == "/host/path/to/repo"


def test_fetch_ohlcv_scheduled_weekly_before_generation():
    """Schedule must fire before job_generate_strategies (Sat 20:00) so the
    Saturday mini-backtests see fresh data."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    # Confirm both schedule lines coexist and ordering is correct (fetch < generate)
    fetch_idx = src.index('id="fetch_ohlcv"')
    gen_idx = src.index('id="generate_strategies"')
    assert fetch_idx < gen_idx, "fetch_ohlcv must be scheduled before generate_strategies in source"
    # Confirm the cron expression itself
    assert 'day_of_week="sat", hour=19, minute=30, id="fetch_ohlcv"' in src


def test_backtest_cap_covers_phase_6_matrix():
    """job_backtest_candidates must process enough strategies to cover one
    full Phase 6 generation (20-cell coherence matrix). Earlier cap was 10
    and silently dropped 5+ candidates from a 15-strategy registration
    batch, leaving them with no full backtest scoring."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    # Find the for-loop that iterates candidates
    import re
    m = re.search(r"for cand in candidates\[:(\d+)\]:", src)
    assert m, "expected `for cand in candidates[:N]:` slicing in job_backtest_candidates"
    cap = int(m.group(1))
    assert cap >= 20, (
        f"backtest cap is {cap}; must be >= 20 to cover Phase 6's 20-cell matrix output"
    )
