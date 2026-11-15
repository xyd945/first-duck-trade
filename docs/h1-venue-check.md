# H1 venue check: OKX EEA, Bybit EU and the OHLCV source

Card t_85936a03 (parent t_529f2b91). Wandri, 15 November 2026. Research only: read-only, keyless public calls. No dry-run, scheduler, order, API key or repository change was made.

Inputs: `项目/Quant desk/G0 H1 crypto trend.md` §2, §4, §6 and §9 (pairs, bars, history, splits) and `00 Project brief.md`. Local evidence files are named in the last section; numbered citations are listed under Sources.

## Verdict per venue

| Venue and route | Freqtrade `exchange.name` | Verdict |
|---|---|---|
| OKX global host `www.okx.com` | `okx` | **Works**, no extra config. Recommended for H1 data and the dry-run. |
| OKX EEA host `eea.okx.com` | `myokx` | **Works**: same USDT markets and identical candles. Use it if the EEA host is preferred. |
| OKX EEA host by override | `okx` + `ccxt_config.hostname` | **Works with config X** (REST only; the websocket stays on `ws.okx.com`). Not needed; use `myokx`. |
| Bybit EU `api.bybit.eu` | `bybiteu` | **Works with config**: USDC or EUR pairs only. **Does not work for H1 as registered**: no USDT pairs, and history starts on 2025-06-11. |
| Bybit EU host by override | `bybit` + `ccxt_config.hostname` | Works for USDC pairs, same result as `bybiteu`; redundant. |
| Bybit global `api.bybit.com` | `bybit` | **Does not work** for H1: spot history starts on 2021-07-05, and EEA access to Bybit Global is being limited. |
| Binance `api.binance.com` | `binance` | Public data works and goes back further, but no account route exists for an NL resident. Cross-check only. |

**Recommended OHLCV source:** OKX USDT daily candles through `freqtrade download-data` with exchange `okx`. It reproduces G0 §2's history start dates exactly, with no gaps.

**Recommended dry-run exchange block** (keyless, tested with `freqtrade test-pairlist` and the dry-run candle refresh):

```json
"exchange": {
    "name": "okx",
    "key": "",
    "secret": "",
    "password": "",
    "pair_whitelist": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "pair_blacklist": []
}
```

## Versions checked

- first-duck-trade at merged `main` `794fd9196790909940faacab90426aa6418e2fde`. `docker-compose.yml` does not pin a Freqtrade version: lines 11, 55 and 88 use `freqtradeorg/freqtrade:stable`, and line 118 uses `stable_freqai`.
- On 15 November 2026, Docker Hub's `stable` tag was last pushed on 2026-10-30, minutes after the `2026.10` tags.[25] No `2026.11` git tag exists yet (checked with `git ls-remote`), so `stable` is Freqtrade 2026.10 today.
- Freqtrade 2026.10 is commit `4cf0351e08d26e96a6429ef48103b726bb230cf8`. It pins `ccxt==4.5.90` (`requirements.txt` L11).[9] ccxt tag `v4.5.90` is commit `94006a9d74a6eb269d7de631fa28c672afc5cabc`.
- The live tests ran in the isolated engine venv in the H1 workspace: Freqtrade 2026.10 with ccxt 4.5.92 (`v4.5.92` = `c1ad94df715915c9adb481ac2c51199a66539e59`). Every ccxt line cited below was checked in the `v4.5.90` files and is the same there.
- Risk: `stable` will move to 2026.11 if that is released before 19 December. Pinning `freqtradeorg/freqtrade:2026.10` for the dry-run would keep the engine fixed. This is for the dry-run config owner to decide.

## 1. OKX EEA route (`eea.okx.com`)

**Freqtrade support.** Freqtrade 2026.10 lists `myokx` as a supported exchange (`freqtrade/exchange/common.py` L51–67).[1] Its `Myokx` class is the OKX class restricted to spot (`freqtrade/exchange/okx.py` L313–320).[2] The docs say accounts registered on my.okx.com must use `myokx`.[4] OKX's API FAQ says EEA users must send API requests to `eea.okx.com`; a wrong domain gives error 50119 for an API key.[17] Both statements concern keyed accounts, which H1 does not use.

**ccxt classes.** ccxt `myokx` sets `hostname` to `eea.okx.com`, with REST base `https://{hostname}` (`python/ccxt/myokx.py` L18–32).[10] ccxt `okx` sets `hostname` to `www.okx.com` (`python/ccxt/okx.py` L202).[11]

**Hostname override through config: yes.** The mechanism has three steps:
- Freqtrade merges `ccxt_config` and `ccxt_sync_config` into the sync client, and `ccxt_config` and `ccxt_async_config` into the async client (`freqtrade/exchange/exchange.py` L288–301).[3]
- ccxt's constructor deep-merges `describe()` with that config and sets every key as an attribute (`python/ccxt/base/exchange.py` L388–394).[12]
- `okx.sign()` builds each REST URL through `implode_hostname` (`okx.py` L6484–6489).[11]

So `"hostname": "eea.okx.com"` moves all REST calls. The run confirmed it: Freqtrade logged `Applying additional ccxt config: {'hostname': 'eea.okx.com'}`, and the loaded client's REST host was `eea.okx.com`. A `urls` override uses the same mechanism but was not tested, because it is not needed.

The override does not move the websocket. ccxt pro `okx` hard-codes `wss://ws.okx.com:443/ws/v5` (`python/ccxt/pro/okx.py` L64),[13] while pro `myokx` uses `wss://wseea.okx.com:443/ws/v5` (`python/ccxt/pro/myokx.py` L19).[14] Freqtrade enables exchange websockets by default (`exchange.enable_ws`).[5] Use `myokx` rather than the override if the EEA host is wanted.

**Public endpoints on the EEA host work.** Tests ran on 2026-11-15 between 01:43 and 01:50 UTC from the Mac mini on a Dutch IP address:
- `GET /api/v5/public/instruments?instType=SPOT` returned 1,232 spot instruments on both hosts, with the same quote counts (USDT 483, USDC 303, EUR 273). BTC-, ETH-, SOL- and XRP-USDT are `live` on both. The public API does not filter markets by region; tradability is decided per account.[18]
- `GET /api/v5/market/ticker` and `/market/books` for BTC-USDT answered on both hosts.
- `freqtrade test-pairlist` in dry-run with empty keys kept all four USDT pairs on `okx`, `myokx` and `okx` with the override.
- `freqtrade download-data --timeframes 1d --timerange 20171201-` gave identical files from `okx` and `myokx`. `check_downloads.py` found the two DataFrames equal for all four pairs.
- Dry-run candle path: `Exchange.refresh_latest_ohlcv()` with `startup_candle_count = 400` (G0 §9's dry-run value) made 2 calls and returned 599 closed daily candles per pair on all three routes. This is the call the bot's data provider makes each iteration.
- Not tested: websocket streams and a full `freqtrade trade` dry-run, which this card rules out.

**Candle limit.** Freqtrade 2026.10 uses 300 candles per call for OKX spot (`okx.py` L83–84),[2] and allows up to 5 calls per pair at startup (`exchange.py` L962–999).[3] The 2026.10 docs still warn that OKX gives only 100 candles per call,[4] and G0 §9 relied on that warning; the code says 300. With either limit, 400 startup candles fit under the 5-call rule.[6]

**No keys needed.** Freqtrade's docs say API keys are not required in dry-run, may be left empty, and only read-only exchange calls are made.[5]

**Verdict: works.** Recommended config, the global host:

```json
"exchange": {
    "name": "okx",
    "key": "",
    "secret": "",
    "password": "",
    "pair_whitelist": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "pair_blacklist": []
}
```

EEA host alternative: the same block with `"name": "myokx"`.

Override form (works for REST, not recommended):

```json
"exchange": {
    "name": "okx",
    "key": "",
    "secret": "",
    "password": "",
    "ccxt_config": {"hostname": "eea.okx.com"},
    "ccxt_async_config": {"hostname": "eea.okx.com"},
    "pair_whitelist": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "pair_blacklist": []
}
```

Why `okx` rather than `myokx`: both returned identical data. `okx` is what G0 §9 registers for the dry-run and what the repository templates use, and G0's data checks were made against `www.okx.com`. The EEA host matters only for a keyed EEA account.[4][17] Switching later is a one-line change that does not change the data.

## 2. Bybit EU spot

**Freqtrade support.** Freqtrade 2026.10 lists `bybiteu` as supported (`common.py` L51–67),[1] and its `BybitEU` class allows spot only (`freqtrade/exchange/bybit.py` L337–345).[7] ccxt `bybiteu` sets `hostname` to `bybit.eu`, uses `https://api.{hostname}`, and loads spot markets only (`python/ccxt/bybiteu.py` L19–66).[15] Bybit's API docs tell EEA users to use `https://api.bybit.eu`.[19] ccxt `bybit` defaults to `bybit.com` (`python/ccxt/bybit.py` L43).[16] Overriding it to `bybit.eu` through `ccxt_config` gave the same result as `bybiteu` in testing.

**Public data on `api.bybit.eu`:**
- `GET /v5/market/instruments-info?category=spot` returned 129 spot instruments: USDC 108, EUR 17, PLN 4, and **no USDT**. BTC, ETH, SOL and XRP trade against USDC and EUR.
- `freqtrade test-pairlist` with the USDT whitelist removed all four pairs ("not compatible with exchange Bybit EU"), leaving an empty whitelist. With USDC pairs, all four were kept.
- `freqtrade download-data` for 1d USDC pairs from 2017-12-01 starts on **2025-06-11** (522 candles to 2026-11-14), with no gaps.
- Oddity: the kline and ticker endpoints on `api.bybit.eu` still answer for `BTCUSDT` (klines from 2025-08-28), although no USDT market is listed. Freqtrade does not use these, because the market is not loaded.

**Bybit global (`api.bybit.com`).** Daily USDT spot klines start on 2021-07-05 for BTC and ETH, 2021-07-20 for XRP and 2021-10-21 for SOL (REST probe). That is too short for G0's 2018 warm-up. Bybit is also progressively limiting EEA residents' access to Bybit Global.[20]

**Verdict: works with config (USDC or EUR pairs); does not work for H1 as registered.** Exact config, tested with `test-pairlist` and `download-data`:

```json
"stake_currency": "USDC",
"exchange": {
    "name": "bybiteu",
    "key": "",
    "secret": "",
    "pair_whitelist": ["BTC/USDC", "ETH/USDC", "SOL/USDC", "XRP/USDC"],
    "pair_blacklist": []
}
```

G0 §4's Bybit check is a cost sensitivity on the selected configuration: 0.30% base and 0.60% stressed, from the 0.25% VIP 0 fee.[21] It can run on the OKX data with `--fee` and needs no Bybit data or venue [inference].

## 3. Free public OHLCV source for G0's history

G0 needs daily (`1Dutc`, 00:00 UTC) USDT bars for BTC, ETH, SOL and XRP. BTC and ETH run from 2018-01-11 (warm-up start, §6), XRP from 2020-01-01 and SOL from 2020-10-01 (§2). The development window starts on 2019-01-14.

| Source (how fetched) | BTC | ETH | XRP | SOL | Meets G0? |
|---|---|---|---|---|---|
| OKX `okx` or `myokx`, `freqtrade download-data` | 2018-01-11 | 2018-01-11 | 2020-01-01 | 2020-10-01 | **Yes, exactly**: 0 gaps, every bar at 00:00 UTC |
| Binance `binance`, `freqtrade download-data` (REST) | 2017-08-17 | 2017-08-17 | 2018-05-04 | 2020-08-11 | More history, but a different venue from the registered one |
| Binance dumps, `data.binance.vision` | monthly 1d archives exist | | | | Not used by Freqtrade for 1d |
| Bybit EU `bybiteu` (USDC) | 2025-06-11 | 2025-06-11 | 2025-06-11 | 2025-06-11 | No |
| Bybit global `bybit` (USDT, REST probe) | 2021-07-05 | 2021-07-05 | 2021-07-20 | 2021-10-21 | No |

Notes:
- The OKX download had 3,230 BTC and ETH bars to 2026-11-14, 2,510 XRP bars and 2,236 SOL bars. This matches G0 §9's counts to 2026-11-13 plus one day.
- ccxt `okx` turns `1d` into `1Dutc` (default timezone UTC, timeframes of 6h or more). It switches to the `history-candles` endpoint when `since` is older than 1,439 bars (`okx.py` L2684–2728).[11] The downloaded timestamps confirm the 00:00 UTC alignment.
- The Binance REST rows for BTC and ETH start at the 2017-12-01 timerange I requested. The listing dates 2017-08-17, 2018-05-04 and 2020-08-11 come from a REST probe with `startTime` in 2017.
- Binance dumps: HEAD requests for `BTCUSDT-1d-2018-01.zip` and `SOLUSDT-1d-2020-08.zip` returned 200. Freqtrade 2026.10 uses the dumps only for spot 1s, 1m, 3m and 5m; daily data goes through REST (`freqtrade/exchange/binance.py` L194–215).[8]
- Switching H1 to Binance data would change the registered source and the eligibility dates. That would be a G0 change, so keep Binance as a cross-check.

Command used (keyless; the config is the OKX block above plus `"dry_run": true`, `"stake_currency": "USDT"` and `"timeframe": "1d"`):

```bash
freqtrade download-data -c cfg-okx.json --userdir ud --datadir data-okx --timeframes 1d --timerange 20171201-
```

The still-open bar is dropped, so the last bar fetched at 01:44 UTC on 15 November was 2026-11-14. G0 §2's S2 freeze after 00:10 UTC on 2026-12-19 fits this behaviour. I did not check whether OKX ever revises historical candles.

## 4. Regional and regulatory caveats for an NL resident

1. **USDT.** OKX does not let EEA users trade USDT, because USDT is not MiCA-authorised.[18] Bybit EU lists no USDT spot market (instruments probe above). ESMA expected CASPs to stop offering non-MiCA-compliant stablecoins by the end of Q1 2025.[22] Its opinion of 8 October 2026 says MiCA-authorised CASPs should cease all services on them, with legacy exposures remediated within three months of publication.[23] By my arithmetic, that is about 8 January 2027.
   - Effect on H1 paper trading: none found. A keyless dry-run reads public market data and makes only read-only calls,[5] so it uses no trading service.
   - Effect on a live step: it would need USDC or EUR pairs, as G0 §2 already records. Their history is short: OKX BTC- and XRP-USDC from 2025-07-01, OKX EUR pairs from late 2023 or 2024 (G0 §9), and Bybit EU USDC from 2025-06-11 (this check).
2. **Which OKX host.** Public data was identical on `www.okx.com` and `eea.okx.com`, so NL residence does not force `myokx` for a keyless dry-run. The host matters only once keys are used: an EEA account must use `eea.okx.com`.[17][4]
3. **Bybit.** Bybit EU is the EEA route and offers spot only,[7][15] at a 0.25% VIP 0 spot fee.[21] EEA residents' access to Bybit Global is being progressively limited.[20]
4. **Binance.** Binance left the Netherlands. Since 2023-07-17, Dutch residents can only withdraw.[24] Its public API was reachable from the Dutch IP address (HTTP 200), but there is no account route. Use it only as a data cross-check.
5. **Not checked; for Oliver:** the exchanges' API and market-data terms. This matters if frozen OHLCV files are committed to the public `first-duck-trade` repository. Tax is not checked either (Claire, before any live order, as G0 already notes).

## Evidence and reproduction

The evidence bundle `h1-venue-evidence.tar.gz` is saved in the team Obsidian vault at `项目/Quant desk/H1 venue check evidence/`. It holds scripts, configs, JSON results and logs; the downloaded candle files are not included. Everything ran from `/Users/tkj945/.hermes/profiles/wandri/cache/scratch/h1-venue/` with the engine venv `.engine-venv` in the t_529f2b91 workspace.

| File | What it shows |
|---|---|
| `probe_venues.py` / `.json` | Raw public REST: OKX instruments on both hosts, Bybit EU and global instruments and earliest klines, Binance earliest klines, and `data.binance.vision` HEAD checks |
| `cfg-*.json`, `tp-*.log` | `freqtrade test-pairlist` per route (dry-run, empty keys) |
| `dl-*.log` | `freqtrade download-data` logs for `okx`, `myokx`, `bybiteu` (USDC) and `binance` |
| `check_downloads.py` / `.json` | Start dates, gaps, 00:00 UTC alignment, `okx` against `myokx` equality, and asserts on G0 §2's start dates |
| `check_dryrun_candles.py` / `.json`, `dryrun-candles.log` | `refresh_latest_ohlcv()` with `startup_candle_count = 400` on `okx`, `myokx` and the override |

Re-run with `<engine-venv>/bin/python check_downloads.py` and `<engine-venv>/bin/python check_dryrun_candles.py cfg-okx.json cfg-myokx.json cfg-okx-eea-host.json`. Both exit 0 on the evidence above.

Code links point to the pinned Freqtrade commit or ccxt tag. The exact line ranges are cited in the text, because the source list keeps one entry per file.

## Sources

[1] https://github.com/freqtrade/freqtrade/blob/4cf0351e08d26e96a6429ef48103b726bb230cf8/freqtrade/exchange/common.py — Freqtrade 2026.10 freqtrade/exchange/common.py SUPPORTED_EXCHANGES (L51-67)
[2] https://github.com/freqtrade/freqtrade/blob/4cf0351e08d26e96a6429ef48103b726bb230cf8/freqtrade/exchange/okx.py — Freqtrade 2026.10 freqtrade/exchange/okx.py class Myokx (L313-320)
[3] https://github.com/freqtrade/freqtrade/blob/4cf0351e08d26e96a6429ef48103b726bb230cf8/freqtrade/exchange/exchange.py — Freqtrade 2026.10 freqtrade/exchange/exchange.py ccxt_config / ccxt_async_config merge (L288-301)
[4] https://www.freqtrade.io/en/2026.10/exchanges — Freqtrade 2026.10 docs: Exchange-specific Notes (OKX, Bybit, exchange configuration)
[5] https://www.freqtrade.io/en/2026.10/configuration — Freqtrade 2026.10 docs: Configuration (dry-run API keys; exchange.enable_ws)
[6] https://www.freqtrade.io/en/2026.10/strategy-customization — Freqtrade 2026.10 docs: Strategy Customization (startup_candle_count vs ohlcv_candle_limit x 5)
[7] https://github.com/freqtrade/freqtrade/blob/4cf0351e08d26e96a6429ef48103b726bb230cf8/freqtrade/exchange/bybit.py — Freqtrade 2026.10 freqtrade/exchange/bybit.py class BybitEU (L337-345)
[8] https://github.com/freqtrade/freqtrade/blob/4cf0351e08d26e96a6429ef48103b726bb230cf8/freqtrade/exchange/binance.py — Freqtrade 2026.10 freqtrade/exchange/binance.py fast download only for spot 1s/1m/3m/5m (L194-215)
[9] https://github.com/freqtrade/freqtrade/blob/4cf0351e08d26e96a6429ef48103b726bb230cf8/requirements.txt — Freqtrade 2026.10 requirements.txt ccxt==4.5.90 (L11)
[10] https://github.com/ccxt/ccxt/blob/v4.5.90/python/ccxt/myokx.py — ccxt v4.5.90 python/ccxt/myokx.py hostname eea.okx.com (L18-32)
[11] https://github.com/ccxt/ccxt/blob/v4.5.90/python/ccxt/okx.py — ccxt v4.5.90 python/ccxt/okx.py hostname www.okx.com (L202)
[12] https://github.com/ccxt/ccxt/blob/v4.5.90/python/ccxt/base/exchange.py — ccxt v4.5.90 python/ccxt/base/exchange.py __init__ applies constructor config to attributes (L388-394)
[13] https://github.com/ccxt/ccxt/blob/v4.5.90/python/ccxt/pro/okx.py — ccxt v4.5.90 python/ccxt/pro/okx.py websocket URL ws.okx.com (L64)
[14] https://github.com/ccxt/ccxt/blob/v4.5.90/python/ccxt/pro/myokx.py — ccxt v4.5.90 python/ccxt/pro/myokx.py websocket URL wseea.okx.com (L19)
[15] https://github.com/ccxt/ccxt/blob/v4.5.90/python/ccxt/bybiteu.py — ccxt v4.5.90 python/ccxt/bybiteu.py hostname bybit.eu, spot-only markets (L19-66)
[16] https://github.com/ccxt/ccxt/blob/v4.5.90/python/ccxt/bybit.py — ccxt v4.5.90 python/ccxt/bybit.py hostname bybit.com (L43)
[17] https://www.okx.com/en-us/help/api-faq-eea — OKX API FAQ (EEA users must use eea.okx.com), modified 2026-08-27
[18] https://www.okx.com/learn/usdt-not-tradeable-okx-europe-stablecoin-guide — OKX Learn: Why USDT Is Not Tradeable on OKX in Europe (P003)
[19] https://bybit-exchange.github.io/docs/v5/guide — Bybit API docs v5: Integration Guidance (EEA users use api.bybit.eu)
[20] https://announcements.bybit.com/en/article/important-notice-for-users-in-the-european-economic-area-eea--blt4135ab861456d7bf — Bybit announcement 2026-06-28: Important Notice for Users in the EEA
[21] https://www.bybit.eu/en-EU/help-center/article/Bybit-Spot-Fees-Explained — Bybit EU Spot: Fees Explained (P004)
[22] https://www.esma.europa.eu/press-news/esma-news/esma-and-european-commission-publish-guidance-non-mica-compliant-arts-and-emts — ESMA 2025-01-17: guidance on non-MiCA compliant ARTs and EMTs (search snippet)
[23] https://www.esma.europa.eu/press-news/esma-news/esma-sets-out-supervisory-expectations-services-related-unauthorised — ESMA 2026-10-08: supervisory expectations on services related to unauthorised stablecoins
[24] https://www.binance.com/en/support/announcement/detail/b5a647be31cf469b87fc3337fd461ced — Binance 2023-06-16: Notice on Changes of Services in the Netherlands (search snippet)
[25] https://hub.docker.com/v2/repositories/freqtradeorg/freqtrade/tags/stable — Docker Hub API: freqtradeorg/freqtrade:stable tag metadata (last pushed 2026-10-30)
