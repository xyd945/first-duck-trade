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
