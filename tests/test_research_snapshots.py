"""Synthetic daily candles only; no exchange data or performance claims."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "user_data/scripts"))
import research

PAIRS = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT")


def source_data(root, end="2020-01-12"):
    root.mkdir(parents=True, exist_ok=True)
    for pair in PAIRS:
        pd.DataFrame({
            "date": pd.date_range("2020-01-01", end, tz="UTC"),
            "open": 10., "high": 12., "low": 9., "close": 11., "volume": 100.,
        }).to_feather(root / f"{pair.replace('-', '_')}-1d.feather")
    return root


def test_snapshot_is_canonical_csv_with_pair_composite_hash(tmp_path):
    source = source_data(tmp_path / "source")
    manifest_path = research.freeze_snapshot(source, tmp_path / "S1", "20200102", "20200107")
    manifest = json.loads(manifest_path.read_text())
    assert "pairs" in manifest, "Snapshot must hash canonical per-pair CSV, not Feather bytes"
    prefix = b"ts,open,high,low,close,volume\n1577923200000,10,12,9,11,100\n"
    for pair in PAIRS:
        data = (manifest_path.parent / manifest["pairs"][pair]["file"]).read_bytes()
        assert data.startswith(prefix)
        assert len(data.splitlines()) == 7  # six inclusive days plus header
        assert manifest["pairs"][pair]["sha256"] == hashlib.sha256(data).hexdigest()
    expected = hashlib.sha256("".join(manifest["pairs"][p]["sha256"] for p in sorted(PAIRS)).encode()).hexdigest()
    assert manifest["data_snapshot_hash"] == expected
    assert research.verify_snapshot(manifest_path, expected) == manifest
    (manifest_path.parent / manifest["pairs"]["BTC-USDT"]["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        research.verify_snapshot(manifest_path, expected)


def test_s2_requires_identical_s1_prefix_and_closed_cutoff(tmp_path):
    source = source_data(tmp_path / "source")
    s1 = research.freeze_snapshot(source, tmp_path / "S1", "20200101", "20200107")
    s1_hash = json.loads(s1.read_text())["data_snapshot_hash"]
    assert hasattr(research, "verify_prefix"), "S2 must extend S1 byte for byte"
    s2 = research.freeze_snapshot(source, tmp_path / "S2", "20200101", "20200112",
                                  previous=(s1, s1_hash))
    s2_hash = json.loads(s2.read_text())["data_snapshot_hash"]
    research.verify_prefix(s1, s1_hash, s2, s2_hash)
    path = source / "BTC_USDT-1d.feather"
    frame = pd.read_feather(path)
    frame.loc[0, "close"] = 10.5
    frame.to_feather(path)
    with pytest.raises(ValueError, match="prefix"):
        research.freeze_snapshot(source, tmp_path / "revised", "20200101", "20200112",
                                 previous=(s1, s1_hash))
    with pytest.raises(ValueError, match="closed|cutoff"):
        research.freeze_snapshot(source, tmp_path / "future", "20200101", "29990101",
                                 previous=(s1, s1_hash))


@pytest.mark.parametrize("mutation", ["midday", "duplicate", "nan", "unconfirmed", "invalid_ohlc"])
def test_freeze_rejects_invalid_daily_candles(tmp_path, mutation):
    source = source_data(tmp_path / "source")
    path = source / "BTC_USDT-1d.feather"
    frame = pd.read_feather(path)
    if mutation == "midday":
        frame.loc[0, "date"] += pd.Timedelta(hours=1)
    elif mutation == "duplicate":
        frame.loc[1, "date"] = frame.loc[0, "date"]
    elif mutation == "nan":
        frame.loc[0, "close"] = float("nan")
    elif mutation == "unconfirmed":
        frame["confirm"] = 1
        frame.loc[0, "confirm"] = 0
    else:
        frame.loc[0, "high"] = 5
    frame.to_feather(path)
    with pytest.raises(ValueError):
        research.freeze_snapshot(source, tmp_path / "bad", "20200101", "20200107")
