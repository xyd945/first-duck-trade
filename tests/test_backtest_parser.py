"""Regression tests for parse_backtest_output.

The parser was matching abbreviated column headers ("Tot Profit %") instead
of the actual metrics-table data labels ("Total profit %"), so every
strategy reported 0.0% profit and got retired as unprofitable — even when
sharpe was positive and win rate was 87%.

These tests use a captured real Freqtrade output as a fixture so the
parser is verified against the exact format Freqtrade emits, not a
hand-stubbed string that might paper over the bug.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))

from backtest_runner import parse_backtest_output


# Captured 2026-05-20 from `freqtrade backtesting` against
# DonchianADXMomentumContinuation over 20260218-20260519 on 4 pairs.
# Trimmed to the parser-relevant sections.
REAL_OUTPUT = """\
Result for strategy DonchianADXMomentumContinuation
┃     Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
│ ETH/USDT │      8 │         0.51 │           4.062 │         0.41 │     17:38:00 │    5     0     3  62.5 │
│ SOL/USDT │      6 │        -0.11 │          -0.680 │        -0.07 │     17:40:00 │    3     0     3  50.0 │
│ XRP/USDT │     13 │        -0.11 │          -1.497 │        -0.15 │      1:42:00 │    5     0     8  38.5 │
│ BTC/USDT │     12 │        -0.25 │          -3.007 │         -0.3 │     14:40:00 │    4     0     8  33.3 │
│    TOTAL │     39 │        -0.03 │          -1.122 │        -0.11 │     11:25:00 │   17     0    22  43.6 │
│ Backtesting from              │ 2026-02-18 00:00:00            │
│ Backtesting to                │ 2026-05-19 00:00:00            │
│ Total/Daily Avg Trades        │ 39 / 0.43                      │
│ Starting balance              │ 1000 USDT                      │
│ Final balance                 │ 998.878 USDT                   │
│ Absolute profit               │ -1.122 USDT                    │
│ Total profit %                │ -0.11%                         │
│ CAGR %                        │ -0.45%                         │
│ Sortino                       │ -0.44                          │
│ Sharpe                        │ -0.22                          │
│ Profit factor                 │ 0.94                           │
│ Max % of account underwater   │ 0.60%                          │
│ Absolute drawdown             │ 6.018 USDT (0.60%)             │
│ DonchianADXMomentumContinuation │     39 │        -0.03 │          -1.122 │        -0.11 │     11:25:00 │   17     0    22  43.6 │ 6.018 USDT  0.60% │
Backtested 2026-02-18 00:00:00 -> 2026-05-19 00:00:00 | Max open trades : 3
"""


# Captured from a winning strategy (high win rate + positive sharpe + small
# positive profit). This is the case the old parser misreported as 0.0%
# profit and incorrectly retired as unprofitable.
WINNING_OUTPUT = """\
Result for strategy WinningStrategy
┃     Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
│ BTC/USDT │     46 │         0.30 │          14.250 │         1.42 │     11:25:00 │   40     0     6  87.0 │
│    TOTAL │     46 │         0.30 │          14.250 │         1.42 │     11:25:00 │   40     0     6  87.0 │
│ Absolute profit               │ 14.250 USDT                    │
│ Total profit %                │ 1.42%                          │
│ Sharpe                        │ 0.17                           │
│ Profit factor                 │ 3.42                           │
│ Max % of account underwater   │ 0.85%                          │
│ Absolute drawdown             │ 8.500 USDT (0.85%)             │
│ WinningStrategy │     46 │         0.30 │          14.250 │         1.42 │     11:25:00 │   40     0     6  87.0 │ 8.500 USDT  0.85% │
Backtested 2026-02-18 00:00:00 -> 2026-05-19 00:00:00 | Max open trades : 3
"""


def test_profit_total_pct_extracts_from_metrics_table():
    """Regression: was matching 'Tot Profit %' (column header, no number
    follows) instead of 'Total profit %' (metrics table data row)."""
    r = parse_backtest_output(REAL_OUTPUT, "DonchianADXMomentumContinuation")
    assert r["profit_total_pct"] == -0.11, f"got {r['profit_total_pct']}"


def test_profit_total_abs_extracts_from_absolute_profit():
    r = parse_backtest_output(REAL_OUTPUT, "DonchianADXMomentumContinuation")
    # Summary match (group 3) wins first; sign and value must both be right.
    assert r["profit_total_abs"] == -1.122, f"got {r['profit_total_abs']}"


def test_profit_pct_positive_value_extracted():
    """The bug masked positive profits — 87% win rate → 1.42% profit had
    been reported as 0.0% and the strategy got retired."""
    r = parse_backtest_output(WINNING_OUTPUT, "WinningStrategy")
    assert r["profit_total_pct"] == 1.42, f"got {r['profit_total_pct']}"
    assert r["profit_total_abs"] == 14.25, f"got {r['profit_total_abs']}"


def test_parser_still_extracts_sharpe_sortino_factor():
    """Confirm we didn't break the regexes that were already working."""
    r = parse_backtest_output(REAL_OUTPUT, "DonchianADXMomentumContinuation")
    assert r["sharpe"] == -0.22
    assert r["sortino"] == -0.44
    assert r["profit_factor"] == 0.94


def test_parser_extracts_drawdown():
    r = parse_backtest_output(REAL_OUTPUT, "DonchianADXMomentumContinuation")
    assert r["max_drawdown_pct"] == 0.60
    assert r["max_drawdown_abs"] == 6.018


def test_parser_extracts_total_trades():
    r = parse_backtest_output(REAL_OUTPUT, "DonchianADXMomentumContinuation")
    assert r["total_trades"] == 39


def test_parser_handles_missing_profit_section_gracefully():
    """If freqtrade truncates output (timeout, crash), parser must default
    cleanly instead of throwing."""
    truncated = "Result for strategy X\n│    TOTAL │     0 │  0.0 │  0.000 │  0.0 │"
    r = parse_backtest_output(truncated, "X")
    assert r["profit_total_pct"] == 0.0
    assert r["profit_total_abs"] == 0.0


# ---------------------------------------------------------------------------
# parse_backtest_artifact — JSON-from-zip path
#
# Captured 2026-05-20 from the actual Freqtrade artifact for
# RsiBbVolumeMeanReversion (the strategy that auto-promoted in trial #6).
# Only the fields the parser reads are kept; structure matches Freqtrade's
# real output verbatim so a future SDK change is caught by these tests.
# ---------------------------------------------------------------------------

import json as _json
import zipfile as _zipfile

from backtest_runner import parse_backtest_artifact


REAL_ARTIFACT_STRATEGY = {
    "strategy_name": "RsiBbVolumeMeanReversion",
    "total_trades": 27,
    "profit_total": 0.01884796886,        # decimal — must surface as 1.88 (pct)
    "profit_total_abs": 18.84796886,
    "profit_mean": 0.006970329926738167,  # decimal
    "max_drawdown_account": 0.011587980598572837,  # decimal — must surface as 1.16 (pct)
    "max_drawdown_abs": 11.807714939999997,
    "sharpe": 0.6535700690953908,
    "sortino": 0.5395559754062101,
    "profit_factor": 1.987207967642965,
    "winrate": 0.7407407407407407,        # decimal — must surface as 74.07 (pct)
    "holding_avg": "2 days, 23:40:00",
    "backtest_days": 209,
    "starting_balance": 1000,
}


def _make_artifact_zip(tmp_path, strategy_payload, strategy_name="RsiBbVolumeMeanReversion"):
    """Build a tiny zip in the exact shape Freqtrade emits."""
    zip_path = tmp_path / "backtest-result.zip"
    payload = {
        "strategy": {strategy_name: strategy_payload},
        "strategy_comparison": [],
    }
    with _zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("backtest-result.json", _json.dumps(payload))
        zf.writestr("backtest-result_config.json", "{}")  # parser must ignore
    return zip_path


def test_artifact_parses_canonical_real_strategy(tmp_path):
    """End-to-end: every field the previous regex parser produced must
    be present and correctly typed when sourced from the JSON."""
    zip_path = _make_artifact_zip(tmp_path, REAL_ARTIFACT_STRATEGY)
    r = parse_backtest_artifact(zip_path, "RsiBbVolumeMeanReversion")
    assert r["success"] is True
    assert r["total_trades"] == 27
    # Decimals → percentages (matches what parse_backtest_output returns)
    assert round(r["profit_total_pct"], 2) == 1.88
    assert r["profit_total_abs"] == 18.84796886
    assert round(r["max_drawdown_pct"], 2) == 1.16
    assert r["max_drawdown_abs"] == 11.807714939999997
    assert r["sharpe"] == 0.6535700690953908
    assert r["sortino"] == 0.5395559754062101
    assert r["profit_factor"] == 1.987207967642965
    assert round(r["win_rate"], 2) == 74.07
    assert r["avg_duration"] == "2 days, 23:40:00"
    assert r["backtest_days"] == 209


def test_artifact_unit_conversion_matches_console_parser(tmp_path):
    """Critical invariant: both parsers must report the same unit for
    profit_total_pct (percentage), so callers don't see a silent 100×
    change when the source flips between artifact and console."""
    # Console parser returned -0.11 (percentage) for the same magnitude
    # of decimal -0.0011. Artifact parser must do the same conversion.
    zip_path = _make_artifact_zip(tmp_path, {
        **REAL_ARTIFACT_STRATEGY,
        "profit_total": -0.0011,
    })
    r = parse_backtest_artifact(zip_path, "RsiBbVolumeMeanReversion")
    assert round(r["profit_total_pct"], 2) == -0.11


def test_artifact_winrate_converted_to_percentage(tmp_path):
    """JSON gives winrate as a decimal; the established API surfaces a
    percentage. Regression check — earlier console parser returned win%
    like 43.6, not 0.436."""
    zip_path = _make_artifact_zip(tmp_path, {**REAL_ARTIFACT_STRATEGY, "winrate": 0.436})
    r = parse_backtest_artifact(zip_path, "RsiBbVolumeMeanReversion")
    assert round(r["win_rate"], 1) == 43.6


def test_artifact_falls_back_to_max_relative_drawdown(tmp_path):
    """If max_drawdown_account is absent (older Freqtrade or edge case),
    use max_relative_drawdown without crashing."""
    payload = {**REAL_ARTIFACT_STRATEGY}
    del payload["max_drawdown_account"]
    payload["max_relative_drawdown"] = 0.0234
    zip_path = _make_artifact_zip(tmp_path, payload)
    r = parse_backtest_artifact(zip_path, "RsiBbVolumeMeanReversion")
    assert round(r["max_drawdown_pct"], 2) == 2.34


def test_artifact_raises_when_strategy_not_in_payload(tmp_path):
    """Misnamed strategy → caller should hit the fallback path with a
    clear error rather than getting garbage zeros."""
    zip_path = _make_artifact_zip(tmp_path, REAL_ARTIFACT_STRATEGY,
                                  strategy_name="DifferentStrategy")
    with pytest.raises(ValueError, match="not in artifact"):
        parse_backtest_artifact(zip_path, "RsiBbVolumeMeanReversion")


def test_artifact_raises_on_empty_zip(tmp_path):
    """Corrupt / no-JSON zip → caller falls back to console parsing."""
    zip_path = tmp_path / "empty.zip"
    with _zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("README.txt", "not a backtest result")
    with pytest.raises(ValueError, match="no result JSON"):
        parse_backtest_artifact(zip_path, "RsiBbVolumeMeanReversion")


def test_artifact_ignores_config_json(tmp_path):
    """Freqtrade also writes <ts>_config.json into the same zip. The
    parser must skip it and find the actual result JSON instead."""
    # _make_artifact_zip already includes a _config.json — if the parser
    # wrongly grabbed it the test would fail at the strategy lookup.
    zip_path = _make_artifact_zip(tmp_path, REAL_ARTIFACT_STRATEGY)
    r = parse_backtest_artifact(zip_path, "RsiBbVolumeMeanReversion")
    assert r["total_trades"] == 27


def test_artifact_handles_zero_trade_strategy(tmp_path):
    """A strategy that compiles but fires no entries — Freqtrade still
    writes a result JSON, but everything is zero-valued. Parser must
    not divide-by-zero or NoneType-explode."""
    zip_path = _make_artifact_zip(tmp_path, {
        "strategy_name": "Empty",
        "total_trades": 0,
        "profit_total": 0.0,
        "profit_total_abs": 0.0,
        "profit_mean": 0.0,
        "max_drawdown_account": 0.0,
        "max_drawdown_abs": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "profit_factor": 0.0,
        "winrate": 0.0,
        "holding_avg": "",
        "backtest_days": 90,
        "starting_balance": 1000,
    }, strategy_name="Empty")
    r = parse_backtest_artifact(zip_path, "Empty")
    assert r["total_trades"] == 0
    assert r["profit_total_pct"] == 0.0
    assert r["sharpe"] == 0.0
