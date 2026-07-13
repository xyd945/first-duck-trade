"""
FreqaiBaselineLGBM — the hand-written FreqAI baseline (issue #47, Phase 1).

LightGBMRegressor predicting 24-candle (1 day on 1h) forward return from
the full whitelisted feature set: price/volume technicals plus the external
context the rule-based factory already trusts (funding, OI, ETH/BTC
strength, FGI, VIX).

This file is the reference implementation and Docker smoke-test vehicle —
it is NOT auto-registered as a candidate. The factory path renders
equivalent (declarative) subclasses from specs via freqai_spec.py; keep the
two in sync when the base-class contract changes.

Run it manually:

  docker compose --profile backtest run --rm freqtrade-freqai backtesting \
    --config /freqtrade/user_data/configs/config-freqai-base.json \
    --strategy FreqaiBaselineLGBM --freqaimodel LightGBMRegressor \
    --timerange 20260501-20260701 --timeframe 1h
"""

from base_freqai import BaseFreqaiStrategy


class FreqaiBaselineLGBM(BaseFreqaiStrategy):
    STRATEGY_THESIS = (
        "A gradient-boosted regressor over momentum, volatility, volume and "
        "macro-context features can predict next-day BTC/ETH/SOL returns "
        "well enough to beat threshold-gated long entries against HODL."
    )
    STRATEGY_ARCHETYPE = "ml_regressor"
    TARGET_REGIME = "all"
    GENERATION_ID = "freqai-baseline-manual"

    FREQAI_FEATURES = [
        "rsi", "ema_dist", "natr", "adx", "bb_width", "roc", "volume_z",
        "pct_change", "hl_range", "time_cycle",
        "funding", "oi_change", "eth_btc", "alt_strength",
        "macro_fgi", "macro_vix",
    ]

    ENTRY_THRESHOLD = 0.005
    EXIT_THRESHOLD = 0.0

    stoploss = -0.06
    minimal_roi = {"0": 0.15, "60": 0.08, "120": 0.04, "240": 0.02}
