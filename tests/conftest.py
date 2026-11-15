"""Shared test fixtures for First Duck Trade tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolated_trial_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAL_LEDGER_PATH", str(tmp_path / "trial_ledger.sqlite"))


@pytest.fixture(autouse=True)
def legacy_opt_in(request, monkeypatch):
    """These suites exercise the retained, explicitly enabled factory paths."""
    if request.path.name == "test_freqai_generator.py":
        monkeypatch.setenv("ENABLE_LLM_GENERATION", "true")
    if request.path.name == "test_strategy_critic.py":
        monkeypatch.setenv("ENABLE_STRATEGY_CRITIC", "true")
    if request.path.name == "test_hyperopt.py":
        monkeypatch.setenv("ENABLE_HYPEROPT_RESCUE", "true")
    if request.path.name in {"test_deployment_manager.py", "test_phase3_reconciler_acting.py"}:
        # Retained Docker behaviour is tested only against mocks; production remains locked.
        import paper_risk
        monkeypatch.setattr(paper_risk, "EXECUTION_ENABLED", True)


@pytest.fixture
def sample_ohlcv():
    """Generate a simple OHLCV dataframe for testing indicators."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2025-01-01', periods=n, freq='1h', tz='UTC')

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.abs(np.random.randn(n) * 1000) + 500

    return pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }).reset_index(drop=True)


@pytest.fixture
def trending_ohlcv():
    """Generate a clearly trending (upward) OHLCV dataframe."""
    n = 200
    dates = pd.date_range('2025-01-01', periods=n, freq='1h', tz='UTC')

    # Strong uptrend
    close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    high = close + 0.3
    low = close - 0.3
    open_ = close - 0.1
    volume = np.full(n, 1000.0)

    return pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }).reset_index(drop=True)


@pytest.fixture
def choppy_ohlcv():
    """Generate a ranging/choppy OHLCV dataframe."""
    n = 200
    dates = pd.date_range('2025-01-01', periods=n, freq='1h', tz='UTC')

    # Oscillating around 100
    close = 100 + np.sin(np.arange(n) * 0.3) * 2 + np.random.randn(n) * 0.1
    high = close + 0.5
    low = close - 0.5
    open_ = close + np.random.randn(n) * 0.1
    volume = np.full(n, 1000.0)

    return pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }).reset_index(drop=True)
