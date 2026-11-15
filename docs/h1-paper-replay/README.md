# H1 paper adapter development comparison

`comparison.json` is the derived, per-day/per-pair comparison from the offline execution replay. `manifest.json` binds it to the clean source commit, source-file SHA-256 hashes, installed engine versions and the canonical input snapshot hash. It also records the two local suite results.

The 30-day window is **2–31 December 2021**. The measured result is **120/120 targets matched**, **100% raw and risk-adjusted agreement**, and **32 real Freqtrade dry-run order records**. All 120 opportunities remain in the denominator.

This replays archived development candles with flat offline books at daily opens. It does not measure historical spread, public-data execution slippage, forward returns or a G2 pass. The service lifecycle and weekly writer have separate synthetic offline subprocess coverage in the engine suite.

Raw candles, journals, trades databases and local logs are not published here. The manifest's source commit precedes this evidence-only commit; final-head reruns are recorded in PR #56 and its Kanban handoff.

Reproduce with an isolated environment installed from `docker/requirements-h1-engine.txt`:

```sh
TRIAL_LEDGER_PATH=/new/path/replay-ledger.sqlite .venv-h1/bin/python user_data/scripts/h1_replay.py --data /path/to/h1-dev-bounded --output /new/path/replay
```

The output directory must not already exist. The input's canonical snapshot hash must equal the manifest's `input_snapshot_hash` for an exact reproduction. Trial identifiers and creation times differ between runs; the day/pair details, counts and agreement must match.
