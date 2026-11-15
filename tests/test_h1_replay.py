"""Daily comparisons include missing/blocked records in their denominator."""
import pytest
pytest.importorskip("freqtrade")


def test_compare_counts_missing_and_blocked_days_as_mismatches():
    from h1_replay import compare
    expected = [{"time": f"2021-12-{day:02d}T00:00:00+00:00", "target_weights": {"BTC/USDT": .2}}
                for day in range(1, 5)]
    actual = [{**row, "risk_scale": 1.} for row in expected[:3]]
    actual[0]["target_weights"] = {"BTC/USDT": .204}
    actual[1]["target_weights"] = {"BTC/USDT": .206}
    actual[2]["risk_scale"] = 0.
    report = compare(actual, expected, ["BTC/USDT"])
    assert report["matched"] == 1
    assert report["total"] == 4
    assert report["match_percent"] == 25.
    assert not report["pass"]


def test_real_dry_run_replay_matches_backtest_targets(tmp_path):
    import json
    import pandas as pd
    import numpy as np
    from tests.test_h1_engine import ROOT
    from h1_replay import replay
    cfg = json.loads((ROOT / "user_data/configs/h1_dryrun.json").read_text())
    dates = pd.date_range("2018-01-11", "2019-02-04", tz="UTC")
    price = 100 + np.arange(len(dates)) * .1 + np.sin(np.arange(len(dates)))
    data = {p: pd.DataFrame(dict(date=dates, open=price, high=price, low=price, close=price, volume=1000.))
            for p in cfg["exchange"]["pair_whitelist"]}
    report = replay(cfg, data, "2019-01-14", "2019-02-04", tmp_path / "replay")
    assert report["target_match_percent"] == 100.
    # Two eligible coins in 2019 exceed the four-sleeve paper cap; do not hide those blocks.
    assert report["match_percent"] < 100.
    assert report["paper_order_count"] > 0
    assert report["paper_runmode"] == "dry_run"
    assert report["days"] == 22

