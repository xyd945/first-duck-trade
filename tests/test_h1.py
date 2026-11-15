"""Hand-calculated synthetic checks; no exchange or historical prices."""
import math
import json
from copy import deepcopy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("freqtrade")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "user_data/strategies/candidates"))


def test_hand_computed_votes_and_sample_volatility():
    from H1Trend import trend_frame
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=6, tz="UTC"),
                          "close": [100, 110, 121, 108.9, 119.79, 107.811]})
    result = trend_frame(frame, [1, 2, 3], 3)
    assert result.score.iloc[3:].tolist() == pytest.approx([1 / 3, 2 / 3, 0])
    up, down = math.log(1.1), math.log(.9)
    mean = (up + 2 * down) / 3
    sigma = math.sqrt(((up - mean) ** 2 + 2 * (down - mean) ** 2) / 2) * math.sqrt(365)
    assert result.sigma.iloc[-1] == pytest.approx(sigma)


def test_calendar_lookback_uses_earlier_close_within_three_days():
    from H1Trend import trend_frame
    dates = pd.date_range("2020-01-01", periods=9, tz="UTC")
    frame = pd.DataFrame({"date": dates, "close": [100, 200, 80, 90, 110, 115, 120, 125, 130]})
    frame = frame.drop(index=[2, 3, 4])
    result = trend_frame(frame, [3, 3, 3], 2).set_index("date")
    # On Jan 6 the Jan 3 close is missing: use Jan 2's 200, not the third prior row.
    assert result.loc[dates[5], "score"] == 0
    assert result.loc[dates[4], "can_trade"] == False


def test_long_gap_keeps_previous_score_and_logs_anomaly():
    from H1Trend import trend_frame
    dates = pd.date_range("2020-01-01", periods=12, tz="UTC")
    frame = pd.DataFrame({"date": dates, "close": np.arange(100, 112, dtype=float)})
    frame = frame.drop(index=[4, 5, 6, 7])
    frame.loc[11, "close"] = 10
    result = trend_frame(frame, [4, 4, 4], 2).set_index("date")
    assert result.loc[dates[11], "score"] == 1
    assert result.loc[dates[11], "anomaly"]


def test_future_prices_cannot_change_any_prefix():
    from H1Trend import trend_frame
    dates = pd.date_range("2020-01-01", periods=25, tz="UTC")
    frame = pd.DataFrame({"date": dates, "close": np.arange(100, 125, dtype=float)})
    whole = trend_frame(frame, [1, 3, 7], 3)
    for length in range(8, len(frame)):
        pd.testing.assert_frame_equal(trend_frame(frame.iloc[:length], [1, 3, 7], 3), whole.iloc[:length])


POLICY_PATH = Path(__file__).resolve().parents[1] / "user_data/configs/research-h1.example.json"
PARAMS = {"ladder": [20, 60, 180], "sigma_target": .45, "vol_window": 60,
          "rebalance_band": .025, "cost_rate": .003}


def config():
    from freqtrade.enums import RunMode
    policy = json.loads(POLICY_PATH.read_text())
    return {"research_params": deepcopy(PARAMS), "research_h1": policy["h1"],
            "runmode": RunMode.BACKTEST, "fee": .003, "dry_run": True}


def test_registered_grid_and_rule_settings_come_from_config():
    from H1Trend import grid_params
    policy = json.loads(POLICY_PATH.read_text())
    grid = grid_params(policy)
    assert list(grid) == ["F30", "F45", "F60", "M30", "M45", "M60", "S30", "S45", "S60"]
    assert grid["S60"] == {**PARAMS, "ladder": [40, 120, 365], "sigma_target": .6}
    changed = deepcopy(policy)
    changed["h1"]["grid"]["ladders"] = {"X": [1, 2, 3]}
    changed["h1"]["grid"]["sigma_targets"] = [.2]
    assert list(grid_params(changed)) == ["X20"]


@pytest.mark.parametrize("pair,first,expected", [
    ("BTC/USDT", "2018-01-11", "2019-01-14"),
    ("ETH/USDT", "2018-01-11", "2019-01-14"),
    ("XRP/USDT", "2020-01-01", "2021-01-04"),
    ("SOL/USDT", "2020-10-01", "2021-10-04"),
])
@pytest.mark.parametrize("name", ["H1Trend", "H1SimpleTrend", "H1EqualWeight"])
def test_first_trade_waits_for_366_confirmed_bars_and_monday(pair, first, expected, name):
    import H1Trend
    from freqtrade.enums import RunMode
    cls = getattr(H1Trend, name)
    strategy = cls(config())
    dates = pd.date_range(first, periods=400, tz="UTC")
    frame = pd.DataFrame({"date": dates, "open": np.arange(100, 500),
                          "close": np.arange(100, 500), "volume": 10})
    frame["high"], frame["low"] = frame.close, frame.close
    data = strategy.populate_indicators(frame, {"pair": pair})
    result = strategy.populate_entry_trend(data, {"pair": pair})
    entries = result.loc[result.enter_long.eq(1), "date"] + pd.Timedelta(days=1)
    assert entries.iloc[0] == pd.Timestamp(expected, tz="UTC")
    assert strategy.startup_candle_count == 365
    assert result.target_weight.max() <= .5


def test_buy_hold_only_enters_btc_at_registered_start():
    from H1Trend import H1BuyHold
    from freqtrade.enums import RunMode
    strategy = H1BuyHold(config())
    frame = pd.DataFrame({"date": pd.date_range("2018-01-11", periods=400, tz="UTC"),
                          "close": np.arange(100, 500), "volume": 10})
    for pair in ("BTC/USDT", "ETH/USDT"):
        result = strategy.populate_entry_trend(strategy.populate_indicators(frame, {"pair": pair}), {"pair": pair})
        entries = result.loc[result.enter_long.eq(1), "date"] + pd.Timedelta(days=1)
        if pair.startswith("BTC"):
            assert entries.tolist() == [pd.Timestamp("2019-01-14", tz="UTC")]
        else:
            assert entries.empty


@pytest.mark.parametrize("pair,first,expected", [
    ("BTC/USDT", "2018-01-11", "2019-01-14"),
    ("ETH/USDT", "2018-01-11", "2019-01-14"),
    ("XRP/USDT", "2020-01-01", "2021-01-04"),
    ("SOL/USDT", "2020-10-01", "2021-10-04"),
])
def test_real_loader_trim_and_next_bar_conversion(tmp_path, pair, first, expected):
    from types import SimpleNamespace
    from H1Trend import H1EqualWeight
    from freqtrade.configuration import TimeRange
    from freqtrade.data.history import load_pair_history
    from freqtrade.enums import RunMode, CandleType
    from freqtrade.optimize.backtesting import Backtesting, LONG_IDX
    window = TimeRange.parse_timerange("20190113-20220101")
    strategy = H1EqualWeight(config())
    frame = pd.DataFrame({"date": pd.date_range(first, periods=420, tz="UTC")})
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = 100.0
    frame.to_feather(tmp_path / f"{pair.replace('/', '_')}-1d.feather")
    loaded = load_pair_history(pair, "1d", tmp_path, timerange=window,
                               startup_candles=365, data_format="feather", fill_up_missing=False)
    processed = strategy.populate_indicators(loaded, {"pair": pair})
    converter = SimpleNamespace(
        _set_progress_step=lambda *args: None, check_abort=lambda: None,
        _increment_progress=lambda: None, strategy=strategy,
        dataprovider=SimpleNamespace(_set_cached_df=lambda *args: None), timeframe="1d",
        config={"candle_type_def": CandleType.SPOT}, timerange=window, required_startup=365)
    rows = Backtesting._get_ohlcv_as_lists(converter, {pair: processed})[pair]
    first_entry = next(row[0] for row in rows if row[LONG_IDX] == 1)
    assert first_entry == pd.Timestamp(expected, tz="UTC")
    if pair.startswith(("BTC", "ETH")):
        assert rows[0][0] == pd.Timestamp("2019-01-14", tz="UTC")


def test_configured_sma_and_all_strategy_prefixes():
    import H1Trend
    cfg = config()
    cfg["research_h1"].update(minimum_bars=1, sma_fast=2, sma_slow=3,
                              eligibility={"BTC/USDT": "2020-01-01"})
    cfg["research_params"].update(ladder=[1, 2, 3], vol_window=3)
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=9, tz="UTC"),
                          "close": [100, 110, 121, 108.9, 119.79, 107.811, 120, 110, 130]})
    # SMA2 vs SMA3 from index 2: 115.5>110.333, 114.95>113.3,
    # 114.345<116.5633, 113.8005>112.167: votes 1,1,0,1.
    sma = H1Trend.H1SimpleTrend(cfg).populate_indicators(frame, {"pair": "BTC/USDT"})
    assert sma.score.iloc[2:6].tolist() == [1, 1, 0, 1]
    for name in ("H1Trend", "H1SimpleTrend", "H1EqualWeight", "H1BuyHold"):
        strategy = getattr(H1Trend, name)(deepcopy(cfg))
        whole = strategy.populate_indicators(frame, {"pair": "BTC/USDT"})
        for length in range(4, len(frame)):
            part = strategy.populate_indicators(frame.iloc[:length], {"pair": "BTC/USDT"})
            pd.testing.assert_frame_equal(part, whole.iloc[:length])


def test_actual_eligible_count_and_missing_monday_are_causal():
    from H1Trend import H1EqualWeight
    cfg = config()
    cfg["research_h1"].update(minimum_bars=3, eligibility={"BTC/USDT": "2020-01-01", "ETH/USDT": "2020-01-01"})
    dates = pd.date_range("2020-01-01", periods=12, tz="UTC")
    btc = pd.DataFrame({"date": dates, "close": 100., "confirm": 1})
    eth = btc.copy()
    eth.loc[:5, "confirm"] = 0
    strategy = H1EqualWeight(cfg)
    result = strategy.advise_all_indicators({"BTC/USDT": btc, "ETH/USDT": eth})
    # Monday Jan 6: only BTC has 3 confirmed bars, so it gets the full sleeve.
    assert result["BTC/USDT"].iloc[4].target_weight == 1
    eth.loc[4, "confirm"] = 0
    assert not result["ETH/USDT"].iloc[7].rebalance
    prefix = H1EqualWeight(cfg).advise_all_indicators({"BTC/USDT": btc.iloc[:8], "ETH/USDT": eth.iloc[:8]})
    for pair in prefix:
        pd.testing.assert_frame_equal(prefix[pair], result[pair].iloc[:8])


def test_buy_hold_does_not_bypass_confirmed_bar_eligibility():
    from H1Trend import H1BuyHold
    frame = pd.DataFrame({"date": pd.date_range("2019-01-12", periods=3, tz="UTC"), "close": 100.})
    strategy = H1BuyHold(config())
    result = strategy.populate_indicators(frame, {"pair": "BTC/USDT"})
    assert not result.rebalance.any()


def test_sma_fills_short_reference_gaps_but_never_trades_missing_latest():
    from H1Trend import H1SimpleTrend
    cfg = config()
    cfg["research_h1"].update(sma_fast=2, sma_slow=3)
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, tz="UTC"),
                          "close": [100., 110., 999., 120., 130.], "confirm": [1, 1, 0, 1, 1]})
    result = H1SimpleTrend(cfg).populate_indicators(frame, {"pair": "BTC/USDT"})
    # On Jan 4 SMA2=(110+120)/2=115; SMA3=(110+110+120)/3=113 1/3.
    assert result.score.iloc[3] == 1
    assert not result.rebalance.iloc[2]


@pytest.mark.parametrize("field,value", [("sigma_target", -1), ("vol_window", 0),
                                         ("rebalance_band", float("nan")), ("cost_rate", -1)])
def test_invalid_money_or_rule_parameters_fail_at_construction(field, value):
    from H1Trend import H1Trend
    cfg = config()
    cfg["research_params"][field] = value
    with pytest.raises(ValueError):
        H1Trend(cfg)


def test_incomplete_checkpoint_cannot_start_an_engine():
    from H1Trend import H1Trend
    from freqtrade.enums import RunMode
    with pytest.raises(RuntimeError, match="rebalance execution contract"):
        H1Trend(config()).bot_start()
