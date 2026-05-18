"""Tests for R2d: per-trade macro-bucket attribution."""

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))

from trade_attribution import (
    BUCKETS,
    bucket_value,
    load_trades_from_zip,
    attribute_trades,
    summarize_attribution,
    format_attributions_for_reflector,
    aggregate_attributions_by_bucket,
    format_aggregate_for_generator,
)


# ---------------------------------------------------------------------------
# bucket_value
# ---------------------------------------------------------------------------

def test_bucket_value_for_each_named_signal():
    """Every signal in BUCKETS should produce a labeled bucket for a mid value."""
    samples = {
        "fgi": (-20, "fgi_fear"),
        "vix": (15, "vix_low"),
        "btc_funding_rate": (-0.0001, "btc_funding_rate_shorts_pay"),
        "btc_oi_pct_change_24h": (10, "btc_oi_pct_change_24h_building"),
        "alt_strength_zscore_30d": (-1.5, "alt_strength_zscore_30d_btc_dominant"),
    }
    for name, (val, expected) in samples.items():
        assert bucket_value(name, val) == expected


def test_bucket_value_returns_none_for_nan():
    assert bucket_value("vix", float("nan")) is None


def test_bucket_value_returns_none_for_unknown_signal():
    assert bucket_value("not_a_signal", 1.0) is None


def test_bucket_value_boundary_behavior():
    """Boundaries are strictly < — values at the upper bound spill into next bucket."""
    # vix bucket: low (<18), mid (<25), high (rest)
    assert bucket_value("vix", 17.99) == "vix_low"
    assert bucket_value("vix", 18) == "vix_mid"
    assert bucket_value("vix", 25) == "vix_high"


# ---------------------------------------------------------------------------
# load_trades_from_zip
# ---------------------------------------------------------------------------

def _make_export_zip(tmp_path: Path, strategy_name: str, trades: list[dict]) -> Path:
    """Build a minimal Freqtrade-shaped export zip with the given trades."""
    base = f"backtest-result-2026-05-18_19-00-00"
    payload = {
        "strategy": {strategy_name: {"trades": trades}},
        "strategy_comparison": [],
    }
    json_path = tmp_path / f"{base}.json"
    json_path.write_text(json.dumps(payload))
    # Sidecar config (Freqtrade always writes one) — should NOT be picked up
    cfg_path = tmp_path / f"{base}_config.json"
    cfg_path.write_text("{}")

    zip_path = tmp_path / f"{base}.json.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(json_path, arcname=json_path.name)
        zf.write(cfg_path, arcname=cfg_path.name)
    return zip_path


def test_load_trades_from_zip_round_trips(tmp_path):
    trades = [
        {"open_date": "2026-04-01T12:00:00", "profit_ratio": 0.02, "pair": "BTC/USDT"},
        {"open_date": "2026-04-02T08:00:00", "profit_ratio": -0.01, "pair": "BTC/USDT"},
    ]
    zip_path = _make_export_zip(tmp_path, "MyStrat", trades)
    loaded = load_trades_from_zip(zip_path, "MyStrat")
    assert loaded == trades


def test_load_trades_skips_config_sidecar(tmp_path):
    """The _config.json sidecar must not be picked up as the main JSON."""
    zip_path = _make_export_zip(tmp_path, "S", [{"open_date": "2026-01-01", "profit_ratio": 0.0}])
    # Add a _config.json with bogus structure — load must ignore it
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr("_config.json", '{"strategy": {"S": {"trades": [{"WRONG": 1}]}}}')
    trades = load_trades_from_zip(zip_path, "S")
    assert trades[0].get("open_date") == "2026-01-01"
    assert "WRONG" not in trades[0]


def test_load_trades_missing_file_returns_empty(tmp_path):
    assert load_trades_from_zip(tmp_path / "nope.zip", "X") == []


def test_load_trades_malformed_zip_returns_empty(tmp_path):
    bad = tmp_path / "garbage.zip"
    bad.write_text("not actually a zip")
    assert load_trades_from_zip(bad, "X") == []


def test_load_trades_fallback_to_first_strategy(tmp_path):
    """When the requested strategy name isn't in the export, fall back to
    the first strategy present (typical single-strategy backtests)."""
    zip_path = _make_export_zip(
        tmp_path, "ActualName", [{"open_date": "2026-01-01", "profit_ratio": 0.01}]
    )
    # Caller asks for a different name
    trades = load_trades_from_zip(zip_path, "WrongName")
    assert len(trades) == 1


# ---------------------------------------------------------------------------
# attribute_trades — core math
# ---------------------------------------------------------------------------

def _macro_df(rows: list[tuple]) -> pd.DataFrame:
    """rows: [(date_str, vix, btc_funding_rate, alt_strength_zscore_30d), ...]"""
    dates = pd.to_datetime([r[0] for r in rows], utc=True)
    df = pd.DataFrame({
        "vix": [r[1] for r in rows],
        "btc_funding_rate": [r[2] for r in rows],
        "alt_strength_zscore_30d": [r[3] for r in rows],
    }, index=dates)
    return df


def test_attribute_trades_empty_inputs():
    assert attribute_trades([], pd.DataFrame())["total_trades"] == 0
    assert attribute_trades([], _macro_df([("2026-01-01", 20, 0, 0)]))["total_trades"] == 0


def test_attribute_trades_overall_win_rate():
    """3 wins, 2 losses → 60% overall win rate."""
    trades = [
        {"open_date": "2026-01-01", "profit_ratio": 0.01},
        {"open_date": "2026-01-02", "profit_ratio": 0.02},
        {"open_date": "2026-01-03", "profit_ratio": -0.01},
        {"open_date": "2026-01-04", "profit_ratio": 0.005},
        {"open_date": "2026-01-05", "profit_ratio": -0.02},
    ]
    macro = _macro_df([
        ("2026-01-01", 15, 0.0001, -0.5),
        ("2026-01-02", 15, 0.0001, -0.5),
        ("2026-01-03", 15, 0.0001, -0.5),
        ("2026-01-04", 15, 0.0001, -0.5),
        ("2026-01-05", 15, 0.0001, -0.5),
    ])
    result = attribute_trades(trades, macro)
    assert result["total_trades"] == 5
    assert result["overall_win_rate"] == pytest.approx(0.6)


def test_attribute_trades_reveals_high_vix_bucket_as_losing():
    """All low-vix trades win; all high-vix trades lose. Attribution must
    surface vix_low as positive-lift and vix_high as negative-lift."""
    trades = [
        # 5 winners in low vix
        *[{"open_date": f"2026-01-{i:02d}", "profit_ratio": 0.02} for i in range(1, 6)],
        # 5 losers in high vix
        *[{"open_date": f"2026-02-{i:02d}", "profit_ratio": -0.02} for i in range(1, 6)],
    ]
    macro = _macro_df([
        *[(f"2026-01-{i:02d}", 12, 0.0001, 0) for i in range(1, 6)],   # vix_low
        *[(f"2026-02-{i:02d}", 30, 0.0001, 0) for i in range(1, 6)],   # vix_high
    ])
    result = attribute_trades(trades, macro)
    assert result["overall_win_rate"] == pytest.approx(0.5)
    assert result["buckets"]["vix_low"]["win_rate"] == pytest.approx(1.0)
    assert result["buckets"]["vix_low"]["lift"] == pytest.approx(0.5)
    assert result["buckets"]["vix_high"]["win_rate"] == pytest.approx(0.0)
    assert result["buckets"]["vix_high"]["lift"] == pytest.approx(-0.5)
    assert "vix_low" in result["top_positive_lift"]
    assert "vix_high" in result["top_negative_lift"]


def test_attribute_trades_top_lists_are_sign_filtered():
    """A bucket with lift = +0.02 must not appear in top_negative_lift even
    if it's the worst of several positive-lift buckets."""
    trades = [
        # All 10 trades in vix_low; 6 wins → 60% win-rate, lift 0 vs overall 60%
        *[{"open_date": f"2026-01-{i:02d}", "profit_ratio": 0.01} for i in range(1, 7)],
        *[{"open_date": f"2026-01-{i:02d}", "profit_ratio": -0.01} for i in range(7, 11)],
        # 5 trades in vix_mid; 4 wins → 80% win-rate, lift +0.20
        *[{"open_date": f"2026-02-{i:02d}", "profit_ratio": 0.01} for i in range(1, 5)],
        {"open_date": "2026-02-05", "profit_ratio": -0.01},
    ]
    macro = _macro_df(
        [(f"2026-01-{i:02d}", 12, 0.0001, 0) for i in range(1, 11)] +
        [(f"2026-02-{i:02d}", 20, 0.0001, 0) for i in range(1, 6)]
    )
    result = attribute_trades(trades, macro)
    # vix_low has lift 0 (or close), vix_mid has +0.20
    # top_negative_lift must NOT include any bucket with lift >= 0
    for b in result["top_negative_lift"]:
        assert result["buckets"][b]["lift"] < 0


def test_attribute_trades_excludes_low_sample_buckets_from_top():
    """A bucket with only 1-2 trades is noise — it shouldn't appear in top-N."""
    # 9 trades all in vix_low (5 win, 4 loss), 1 in vix_high (loss)
    trades = (
        [{"open_date": f"2026-01-{i:02d}", "profit_ratio": 0.01} for i in range(1, 6)] +
        [{"open_date": f"2026-01-{i:02d}", "profit_ratio": -0.01} for i in range(6, 10)] +
        [{"open_date": "2026-02-01", "profit_ratio": -0.01}]
    )
    macro = _macro_df(
        [(f"2026-01-{i:02d}", 12, 0.0001, 0) for i in range(1, 10)] +
        [("2026-02-01", 30, 0.0001, 0)]
    )
    result = attribute_trades(trades, macro)
    # Min sample = max(3, 10//5) = 3. vix_high only has 1 sample → excluded.
    assert "vix_high" not in result["top_negative_lift"]
    assert result["buckets"]["vix_high"]["trades"] == 1


def test_attribute_trades_handles_nan_macro_value():
    """If a macro value is NaN at trade entry time, that signal is just
    skipped for that trade — other signals still count."""
    trades = [
        {"open_date": "2026-01-01", "profit_ratio": 0.01},
        {"open_date": "2026-01-02", "profit_ratio": -0.01},
    ]
    macro = pd.DataFrame({
        "vix": [float("nan"), float("nan")],
        "btc_funding_rate": [0.0001, 0.0001],
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True))
    result = attribute_trades(trades, macro)
    # No vix buckets should be present
    assert not any(b.startswith("vix_") for b in result["buckets"])
    # But funding bucket should be present
    assert any(b.startswith("btc_funding_rate_") for b in result["buckets"])


def test_attribute_trades_uses_latest_macro_before_entry():
    """Trade at 12:00 should see macro snapshot from earlier the same day,
    not from a later snapshot."""
    trades = [{"open_date": "2026-01-15T12:00:00", "profit_ratio": 0.01}]
    macro = _macro_df([
        ("2026-01-15T00:00:00", 15, 0.0001, 0),   # earlier — should be used
        ("2026-01-15T18:00:00", 30, 0.0001, 0),   # later — should NOT be used
    ])
    result = attribute_trades(trades, macro)
    assert "vix_low" in result["buckets"]
    assert "vix_high" not in result["buckets"]


def test_attribute_trades_skips_trade_with_no_prior_macro():
    """If the only macro snapshots are after the trade entry, the trade
    contributes to overall_win_rate but not to any bucket."""
    trades = [{"open_date": "2026-01-01", "profit_ratio": 0.01}]
    macro = _macro_df([("2026-02-01", 15, 0.0001, 0)])
    result = attribute_trades(trades, macro)
    assert result["total_trades"] == 1
    assert result["overall_win_rate"] == pytest.approx(1.0)
    assert result["buckets"] == {}


# ---------------------------------------------------------------------------
# summarize_attribution
# ---------------------------------------------------------------------------

def test_summarize_empty():
    out = summarize_attribution({"total_trades": 0})
    assert "No trades" in out


def _make_attribution_row(name, regime, status, total, wr, pos, neg, buckets):
    """Helper for format-attributions tests — builds the shape returned by
    strategy_registry.get_recent_attributions."""
    return {
        "name": name,
        "target_regime": regime,
        "status": status,
        "total_trades": total,
        "profit_total_pct": 1.5,
        "sharpe": 0.5,
        "attribution": {
            "total_trades": total,
            "overall_win_rate": wr,
            "buckets": buckets,
            "top_positive_lift": pos,
            "top_negative_lift": neg,
        },
    }


def test_format_attributions_empty_returns_empty_string():
    assert format_attributions_for_reflector([]) == ""


def test_format_attributions_renders_each_strategy():
    rows = [
        _make_attribution_row(
            "StratA", "trending", "candidate", 50, 0.60,
            ["fgi_fear"], ["vix_high"],
            {"fgi_fear": {"trades": 12, "wins": 9, "win_rate": 0.75, "lift": 0.15},
             "vix_high": {"trades": 10, "wins": 3, "win_rate": 0.30, "lift": -0.30}},
        ),
        _make_attribution_row(
            "StratB", "ranging", "active", 30, 0.50,
            ["btc_funding_rate_shorts_pay"], [],
            {"btc_funding_rate_shorts_pay": {"trades": 8, "wins": 6, "win_rate": 0.75, "lift": 0.25}},
        ),
    ]
    out = format_attributions_for_reflector(rows)
    assert "StratA" in out
    assert "StratB" in out
    assert "trending" in out
    assert "ranging" in out
    assert "fgi_fear" in out
    assert "vix_high" in out
    assert "btc_funding_rate_shorts_pay" in out
    # Lift formatting includes sign
    assert "+0.15" in out
    assert "-0.30" in out


def test_format_attributions_handles_strategy_with_no_eligible_buckets():
    rows = [
        _make_attribution_row("LowSample", "all", "candidate", 4, 0.25, [], [], {}),
    ]
    out = format_attributions_for_reflector(rows)
    assert "LowSample" in out
    assert "no buckets" in out.lower()


def test_format_attributions_respects_max_chars():
    # Build many rows so the output overflows
    rows = []
    for i in range(50):
        rows.append(_make_attribution_row(
            f"Strat{i}", "all", "candidate", 20, 0.5,
            ["fgi_fear"], [],
            {"fgi_fear": {"trades": 10, "wins": 6, "win_rate": 0.6, "lift": 0.1}},
        ))
    out = format_attributions_for_reflector(rows, max_chars=500)
    assert len(out) <= 500 + len("\n[...truncated]")
    assert "[...truncated]" in out


# ---------------------------------------------------------------------------
# aggregate_attributions_by_bucket — cross-strategy rollup
# ---------------------------------------------------------------------------

def _agg_row(name, regime, pos, neg, buckets):
    """Build the shape get_recent_attributions returns, for aggregation tests."""
    return {
        "name": name, "target_regime": regime, "status": "candidate",
        "total_trades": sum(b["trades"] for b in buckets.values()) or 20,
        "profit_total_pct": 1.0, "sharpe": 0.5,
        "attribution": {
            "total_trades": 20, "overall_win_rate": 0.5,
            "buckets": buckets,
            "top_positive_lift": pos, "top_negative_lift": neg,
        },
    }


def test_aggregate_empty_input():
    out = aggregate_attributions_by_bucket([])
    assert out["n_strategies"] == 0
    assert out["buckets"] == {}
    assert out["top_consistent_winners"] == []


def test_aggregate_counts_appearances_across_strategies():
    rows = [
        _agg_row("A", "trending", ["fgi_fear"], ["vix_low"], {
            "fgi_fear": {"trades": 10, "wins": 8, "win_rate": 0.8, "lift": 0.20},
            "vix_low": {"trades": 10, "wins": 3, "win_rate": 0.3, "lift": -0.20},
        }),
        _agg_row("B", "trending", ["fgi_fear"], [], {
            "fgi_fear": {"trades": 12, "wins": 9, "win_rate": 0.75, "lift": 0.15},
        }),
        _agg_row("C", "trending", [], ["vix_low"], {
            "vix_low": {"trades": 8, "wins": 2, "win_rate": 0.25, "lift": -0.25},
        }),
    ]
    out = aggregate_attributions_by_bucket(rows)
    assert out["n_strategies"] == 3
    assert out["buckets"]["fgi_fear"]["appears_positive"] == 2
    assert out["buckets"]["fgi_fear"]["appears_negative"] == 0
    # avg_lift averaged across the 2 strategies that had fgi_fear data
    assert out["buckets"]["fgi_fear"]["avg_lift"] == pytest.approx((0.20 + 0.15) / 2)
    assert out["buckets"]["vix_low"]["appears_negative"] == 2
    assert "fgi_fear" in out["top_consistent_winners"]
    assert "vix_low" in out["top_consistent_losers"]


def test_aggregate_filters_by_regime():
    rows = [
        _agg_row("A", "trending", ["fgi_fear"], [],
                  {"fgi_fear": {"trades": 10, "wins": 8, "win_rate": 0.8, "lift": 0.2}}),
        _agg_row("B", "ranging", ["fgi_fear"], [],
                  {"fgi_fear": {"trades": 10, "wins": 8, "win_rate": 0.8, "lift": 0.2}}),
        _agg_row("C", "all", ["fgi_fear"], [],
                  {"fgi_fear": {"trades": 10, "wins": 8, "win_rate": 0.8, "lift": 0.2}}),
    ]
    out = aggregate_attributions_by_bucket(rows, regime="trending")
    # Strict match — only A; target_regime='all' is NOT widened in
    assert out["n_strategies"] == 1
    assert out["buckets"]["fgi_fear"]["appears_positive"] == 1


def test_aggregate_excludes_single_strategy_quirks():
    """A bucket appearing in only 1 strategy must not become a 'consistent pattern'."""
    rows = [
        _agg_row("A", "trending", ["fgi_fear"], [],
                  {"fgi_fear": {"trades": 10, "wins": 8, "win_rate": 0.8, "lift": 0.2}}),
        _agg_row("B", "trending", [], [],
                  {"vix_low": {"trades": 10, "wins": 5, "win_rate": 0.5, "lift": 0.0}}),
    ]
    out = aggregate_attributions_by_bucket(rows, min_strategies=2)
    # fgi_fear has appears_positive=1, only 1 strategy — below threshold
    assert "fgi_fear" not in out["top_consistent_winners"]


def test_aggregate_requires_net_positive_AND_positive_avg_lift_for_winners():
    """A bucket positive in 3 (lift +0.01) and negative in 2 (lift -0.10) has
    net +1 but avg lift is negative — must NOT be classified as a winner."""
    rows = [
        _agg_row("A", "all", ["x"], [], {"x": {"trades": 10, "wins": 5, "win_rate": 0.5, "lift": 0.01}}),
        _agg_row("B", "all", ["x"], [], {"x": {"trades": 10, "wins": 5, "win_rate": 0.5, "lift": 0.01}}),
        _agg_row("C", "all", ["x"], [], {"x": {"trades": 10, "wins": 5, "win_rate": 0.5, "lift": 0.01}}),
        _agg_row("D", "all", [], ["x"], {"x": {"trades": 10, "wins": 3, "win_rate": 0.3, "lift": -0.20}}),
        _agg_row("E", "all", [], ["x"], {"x": {"trades": 10, "wins": 3, "win_rate": 0.3, "lift": -0.20}}),
    ]
    out = aggregate_attributions_by_bucket(rows)
    # avg = (3*0.01 + 2*-0.20)/5 = (0.03 - 0.40)/5 = -0.074 → not a winner
    assert out["buckets"]["x"]["avg_lift"] == pytest.approx(-0.074, abs=1e-3)
    assert "x" not in out["top_consistent_winners"]


# ---------------------------------------------------------------------------
# format_aggregate_for_generator
# ---------------------------------------------------------------------------

def test_format_aggregate_empty_returns_empty():
    assert format_aggregate_for_generator({"n_strategies": 0}, "trending") == ""


def test_format_aggregate_with_no_rankable_buckets_returns_empty():
    """0 winners and 0 losers → no useful signal → skip the section."""
    agg = {"n_strategies": 3, "regime": "trending", "buckets": {},
           "top_consistent_winners": [], "top_consistent_losers": []}
    assert format_aggregate_for_generator(agg, "trending") == ""


def test_format_aggregate_renders_winners_and_losers():
    agg = {
        "n_strategies": 4, "regime": "trending",
        "buckets": {
            "fgi_fear": {"appears_positive": 3, "appears_negative": 0,
                          "n_with_data": 3, "avg_lift": 0.08},
            "vix_low": {"appears_positive": 0, "appears_negative": 3,
                         "n_with_data": 3, "avg_lift": -0.06},
        },
        "top_consistent_winners": ["fgi_fear"],
        "top_consistent_losers": ["vix_low"],
    }
    out = format_aggregate_for_generator(agg, "trending")
    assert "fgi_fear" in out
    assert "vix_low" in out
    assert "trending" in out
    assert "WINS" in out and "LOSSES" in out
    # Counts visible
    assert "3/4" in out
    # Lift signs preserved
    assert "+0.08" in out
    assert "-0.06" in out


def test_format_aggregate_labels_pool_wide_when_regime_mismatch():
    """When the aggregate is pool-wide but generator is building for a
    specific regime, the section header must say so."""
    agg = {
        "n_strategies": 6, "regime": None,
        "buckets": {"fgi_fear": {"appears_positive": 4, "appears_negative": 0,
                                  "n_with_data": 4, "avg_lift": 0.07}},
        "top_consistent_winners": ["fgi_fear"], "top_consistent_losers": [],
    }
    out = format_aggregate_for_generator(agg, "breakout")
    assert "pool-wide" in out.lower()
    assert "breakout" in out


def test_format_aggregate_respects_max_chars():
    buckets = {f"bucket_{i}": {"appears_positive": 5, "appears_negative": 0,
                                 "n_with_data": 5, "avg_lift": 0.10} for i in range(20)}
    agg = {
        "n_strategies": 10, "regime": "all",
        "buckets": buckets,
        "top_consistent_winners": [f"bucket_{i}" for i in range(20)],
        "top_consistent_losers": [],
    }
    out = format_aggregate_for_generator(agg, "all", max_chars=400)
    assert len(out) <= 400 + len("\n[...truncated]")


def test_summarize_renders_top_buckets():
    attr = {
        "total_trades": 10,
        "overall_win_rate": 0.5,
        "buckets": {
            "vix_low": {"trades": 5, "wins": 5, "win_rate": 1.0, "lift": 0.5},
            "vix_high": {"trades": 5, "wins": 0, "win_rate": 0.0, "lift": -0.5},
        },
        "top_positive_lift": ["vix_low"],
        "top_negative_lift": ["vix_high"],
    }
    out = summarize_attribution(attr)
    assert "10" in out
    assert "vix_low" in out
    assert "vix_high" in out
    assert "WINS" in out
    assert "LOSSES" in out
