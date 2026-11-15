"""Keyless, bounded OKX G0 warm-up/dev download. Never fetch a sealed-window bar."""
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode


import pandas as pd

# Direct pagination bounds every response; Freqtrade's bulk downloader can overshoot its end.
starts = {"BTC-USDT": "2018-01-11", "ETH-USDT": "2018-01-11",
          "SOL-USDT": "2020-10-01", "XRP-USDT": "2020-01-01"}
end = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=False)
coverage = {}
for pair, first in starts.items():
    start = int(pd.Timestamp(first, tz="UTC").timestamp() * 1000)
    after, rows, pages = end, [], []
    while after > start:
        query = urlencode(dict(instId=pair, bar="1Dutc", after=after, before=start - 1, limit=100))
        url = "https://www.okx.com/api/v5/market/history-candles?" + query
        payload = json.loads(subprocess.check_output(
            ["curl", "--fail", "--silent", "--show-error", "--max-time", "30", url], text=True))
        assert payload["code"] == "0", payload
        page = payload["data"]
        assert page, (pair, after, "unexpected empty page")
        stamps = [int(row[0]) for row in page]
        assert all(start <= stamp < after <= end for stamp in stamps), (pair, stamps)
        assert all(row[8] == "1" for row in page), "Unconfirmed candle"
        pages.append(dict(url=url, rows=len(page), first=min(stamps), last=max(stamps)))
        rows.extend(page)
        after = min(stamps)
        time.sleep(0.15)
    frame = pd.DataFrame(sorted(rows, key=lambda row: int(row[0])),
                         columns=["ts", "open", "high", "low", "close", "volume", "quote_volume", "quote_volume_ccy", "confirm"])
    frame["date"] = pd.to_datetime(frame.ts.astype("int64"), unit="ms", utc=True)
    expected = pd.date_range(first, "2021-12-31", freq="D", tz="UTC")
    assert list(frame.date) == list(expected), "Duplicate, missing or out-of-window candles"
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column])
    frame[["date", "open", "high", "low", "close", "volume", "confirm"]].to_feather(
        destination / f"{pair.replace('-', '_')}-1d.feather")
    coverage[pair] = dict(first=str(frame.date.min()), last=str(frame.date.max()), rows=len(frame),
                          gaps=0, all_confirmed=True, pages=pages)
    print(pair, first, "2021-12-31", len(frame), flush=True)
(destination / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
