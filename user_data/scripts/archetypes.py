"""
Strategy archetypes — single source of truth (Phase 6).

The LLM, left to its own devices, converges on ~6 textbook patterns
(Donchian breakouts, Keltner channels, EMA crosses, BB-RSI bounces,
volume breakouts, MACD divergence) regardless of how the prompt is
phrased. Diversity-by-suggestion fails because reasoning models
rationalize back to comfortable patterns.

Phase 6 enforces diversity STRUCTURALLY: every generation call takes
an explicit `archetype` parameter and the spec validator rejects
non-matching code. This module is the single source of truth for:

  - the 10 archetypes (enum values)
  - per-archetype thesis + indicator suggestions + threshold guidance
    (injected verbatim into the generation prompt)
  - the coherence matrix — which (archetype, regime) cells make
    sense to generate. Mean-reversion in trending markets, momentum-
    continuation in ranging markets, funding-contrarian in ranging
    markets — all incoherent combinations are pre-filtered here so
    we don't waste LLM calls on category errors.

The matrix has 20 coherent cells. Each weekly cycle generates ONE
strategy per cell, so diversity is by construction rather than by
hope.
"""

from __future__ import annotations

# Valid target regimes (must match validate_spec in strategy_spec.py)
VALID_REGIMES = ("trending", "ranging", "breakout", "all")


# ---------------------------------------------------------------------------
# Archetype catalog
# ---------------------------------------------------------------------------
# Each archetype:
#   thesis            one-sentence pitch (used in the prompt header)
#   blurb             multi-line LLM guidance: setups, indicators, threshold
#                     conventions, what success looks like
#   coherent_regimes  list of target_regime values this archetype can credibly
#                     claim. Generation skips (archetype, regime) cells not
#                     in this list — those are category errors (e.g.
#                     mean-reversion in a strong trend = catching falling knives).

ARCHETYPES: dict[str, dict] = {
    "momentum_continuation": {
        "thesis": "Enter on existing trend strength; ride continuation moves.",
        "coherent_regimes": ["trending", "breakout"],
        "blurb": """\
Thesis: trend-following continuation. Enter LONG when an established uptrend
is mid-move (not exhausted, not yet reversing). Exit on momentum decay.

Typical setups:
  - Higher highs + higher lows with EMA fast > EMA slow
  - ADX > 25 confirming directional strength
  - MACD histogram positive and rising
  - Donchian-channel high break with prior trend established

Indicators to consider: EMA (fast, slow), ADX, MACD, Donchian channel, ATR.
Macro filter: regime must actually be trending (target_regime='trending') or
in a confirmed breakout. AVOID this archetype in ranging markets — every
'pullback entry' becomes a whipsaw.

Threshold conventions: ADX > 25 is meaningful; > 40 is exhaustion. Use ATR
for stoploss sizing, not fixed % stops.""",
    },

    "mean_reversion": {
        "thesis": "Fade extreme oscillator readings back to the mean.",
        "coherent_regimes": ["ranging"],
        "blurb": """\
Thesis: in a sideways/choppy market, oscillator extremes revert. Enter LONG
when oversold; exit when neutral or overbought.

Typical setups:
  - RSI < 30 with price at lower Bollinger Band
  - Stochastic %K < 20 with bullish divergence
  - Price touching prior support after consolidation
  - Williams %R < -80

Indicators to consider: RSI, Bollinger Bands, Stochastic, Williams %R.
Macro filter: REQUIRES ranging regime (target_regime='ranging'). Mean
reversion in trending markets = catching falling knives. The R7 regime
gate will reject your strategy if it claims target='trending' but only
fires on RSI < 30.

Threshold conventions: RSI 30/70 is textbook, 20/80 is aggressive.
Bollinger 2.0σ is standard. Avoid: tight 1.5σ in wide-range markets,
loose 3σ that only fires on crashes.""",
    },

    "breakout_volume": {
        "thesis": "Enter on confirmed breakouts with volume backing.",
        "coherent_regimes": ["breakout"],
        "blurb": """\
Thesis: range expansion with volume confirmation. Detect a structural
breakout (price clearing prior resistance) accompanied by volume spike.

Typical setups:
  - Donchian/horizontal level break + volume > 1.5x SMA20 volume
  - 20-day high break + ATR expansion (volatility regime change)
  - Volume oscillator (e.g. PVO) confirming the break

Indicators to consider: Donchian channel, volume + volume SMA, ATR, PVO.
Macro filter: REQUIRES breakout regime. In ranging markets, breakouts fake
out; in trending markets you're already in the move and the break is just
continuation.

Threshold conventions: volume > 1.5x is the standard "confirmation"
threshold. Don't gate on volume > 3x — that's an outlier filter that fires
once a year.""",
    },

    "vol_squeeze": {
        "thesis": "Trade volatility regime change (compression → expansion).",
        "coherent_regimes": ["breakout"],
        "blurb": """\
Thesis: when implied volatility (BB width / ATR) compresses below historical
norms, the subsequent expansion is large and directional. Enter on the
first confirmed expansion.

Typical setups:
  - Bollinger Band width at 6-month low (squeeze)
  - + ATR rising from a multi-week trough
  - + price breaking BB upper band

Indicators to consider: Bollinger Band width (BBB column), ATR percentile,
Keltner channels (BB inside Keltner = squeeze condition).

Macro filter: target_regime='breakout' is the natural fit. Squeeze in a
range often fails BACK into the range — wrong call.

Threshold conventions: BB width below 30th percentile of 90-day history
qualifies as 'squeezed'. ATR rising for 2+ bars from a trough confirms
expansion. Don't use static thresholds — they'll fire constantly in
quiet periods and never in volatile ones.""",
    },

    "vol_compression_mean_reversion": {
        "thesis": "When ATR collapses inside a range, fade the edges harder.",
        "coherent_regimes": ["ranging"],
        "blurb": """\
Thesis: low volatility + ranging market = tight, predictable oscillations.
Fade the edges of the range with more confidence than a generic mean-
reversion strategy because the regime is doubly confirmed.

Typical setups:
  - ATR at 60-day low AND price near range boundary
  - BB width < 30th percentile AND %B < 0.05 (lower) or > 0.95 (upper)
  - Keltner channel mid line as exit target

Indicators to consider: ATR, BB width, %B, Keltner channel.

Macro filter: REQUIRES ranging regime. This is mean reversion's stronger
variant — narrower stops, tighter targets, more confidence.

Threshold conventions: %B < 0.05 is aggressive entry; combine with RSI
oversold for confirmation. Stops should be tighter than vanilla mean
reversion (smaller ATR multiplier) because the regime predicts low
movement.""",
    },

    "funding_contrarian": {
        "thesis": "Fade overheated funding; buy when shorts pay (squeeze fuel).",
        "coherent_regimes": ["trending", "breakout", "all"],
        "blurb": """\
Thesis: BTC perpetual funding rate captures CROWD POSITIONING. When funding
is sustainedly positive (longs paying shorts), the market is long-loaded
and prone to long squeezes. When funding goes negative (shorts paying),
the crowd is short — squeeze fuel.

Typical setups:
  - btc_funding_rate > 0.0005 for 3+ consecutive 8h periods → FADE (look
    for short entry signal in TA)... since we're spot-only and long-only,
    instead WAIT for funding to normalize, THEN enter long on the dip
  - btc_funding_rate < 0 + price reclaims a key level → LONG (short squeeze)
  - btc_funding_rate < 0 + RSI bullish divergence → strong contrarian long

Indicators to consider: btc_funding_rate (the primary trigger),
btc_oi_pct_change_24h (secondary confirmation), RSI for timing.

This is one of the few archetypes that has a 'all' regime variant — the
positioning signal is regime-independent. Funding extremes mean something
in trending markets (exhaustion) AND in breakouts (over-leveraged
participants).

DO NOT use this in ranging markets — funding stays near zero when nobody's
leveraged into directional bets, and your trigger never fires.

Threshold conventions:
  - frothy: funding > 0.0003 (3bp/8h)
  - extreme frothy: funding > 0.0005 (5bp/8h, sustained 3+ periods)
  - shorts paying: funding < 0 (rare, valuable)
  - extreme: funding < -0.0002 (squeeze fuel)""",
    },

    "oi_cascade_followthrough": {
        "thesis": "Buy after forced de-leveraging cascades (sharp negative OI).",
        "coherent_regimes": ["breakout", "all"],
        "blurb": """\
Thesis: a sharp drop in BTC open interest (btc_oi_pct_change_24h < -10%
or worse) indicates forced de-leveraging — liquidations are flushing
positions out. Historically these cascades mark short-term bottoms
because the marginal seller is GONE.

Typical setups:
  - btc_oi_pct_change_24h < -10% (significant de-lev) + price stabilization
  - btc_oi sharp drop + price reclaim of pre-cascade level → high-confidence long
  - + btc_funding_rate flipping negative (the longs got flushed) → even stronger

Indicators to consider: btc_oi_pct_change_24h (primary), btc_oi (raw),
btc_funding_rate (confirmation that flush happened).

Coherent regimes: 'breakout' fits because cascades ARE volatility events.
'all' fits because the cascade-bottom signal works across regime contexts.
NOT for ranging (no leveraged crowd to flush in sideways markets).

Threshold conventions: -5% daily OI change is noteworthy, -10% is a real
cascade, -20% is rare and historically very high signal. Pair with TA
confirmation (price reclaim) — don't catch the falling knife pre-flush.""",
    },

    "alt_strength_divergence": {
        "thesis": "Long BTC when alt-strength z-score is extreme (capitulation).",
        "coherent_regimes": ["trending", "breakout", "all"],
        "blurb": """\
Thesis: when ETH/BTC ratio's 30-day z-score is extreme NEGATIVE (alt
capitulation INTO BTC), historically BTC is approaching a local top
because the rotation has run its course. Conversely, extreme positive
z-score (alt-season) often precedes broader market exhaustion.

For long-only BTC trading the dominant signal: long BTC when alt-strength
z-score is moderately negative (-0.5 to -1.5) AND BTC technicals are
constructive. Extreme negative (z < -1.5) is late-cycle BTC strength —
trade carefully.

Typical setups:
  - alt_strength_zscore_30d in [-1.5, -0.5] + BTC EMA uptrend → continuation long
  - alt_strength_zscore_30d < -2.0 + extreme funding-positive → CAUTION,
    likely late-cycle top forming
  - alt_strength_zscore_30d crossing from > 0 to < -0.5 → rotation INTO BTC
    starting, early entry

Indicators to consider: alt_strength_zscore_30d (primary),
eth_btc_change_7d (rate of change confirms direction),
btc_funding_rate (helps separate continuation from exhaustion).

This is a cross-asset POSITIONING signal — works across regimes.
NOT well-suited to ranging because the cross-asset move tends to coincide
with directional BTC moves.

Threshold conventions: z-scores beyond ±2.0 are extreme (~5% of days);
±1.0 is meaningful (~16% of days); ±0.5 is noise. Don't gate on
|z| > 0.3 — that fires constantly.""",
    },

    "macro_led_risk_on": {
        "thesis": "Macro thesis primary (DXY/VIX/FGI); TA only confirms timing.",
        "coherent_regimes": ["trending", "breakout", "all"],
        "blurb": """\
Thesis: macro conditions DRIVE the trade thesis. TA exists only to time
the entry. Most current strategies are TA-primary with macro as filter;
this archetype flips that hierarchy.

Typical setups:
  - DXY weakening (dxy < dxy_sma20) + VIX falling (vix < 20) → risk-on
    backdrop. Enter long BTC on any TA-confirmed pullback.
  - Gold flat + SPX rising + FGI moderate (between -10 and +20) → durable
    risk-on. Enter on EMA cross or RSI mid-range bounce.
  - Macro deterioration (DXY rising sharply, VIX > 25) → AVOID even strong
    TA setups; macro headwind dominates.

Indicators to consider: dxy, vix, gold, spx (macro context columns);
fgi (composite of internal signals); minimal TA — EMA cross or RSI > 50
for timing only.

Coherent regimes: trending, breakout, all. Macro is a regime-independent
frame. NOT for ranging — macro thesis is directional by design.

Threshold conventions:
  - vix < 18: complacent, risk-on bias confirmed
  - vix 18-25: neutral; macro filter inconclusive
  - vix > 25: defensive; macro headwind
  - dxy: use SMA20 cross as direction; absolute level matters less
  - fgi: range is -22 to +45 (NOT 0-100, see EXTERNAL DATA section)""",
    },

    "liquidity_sweep_followthrough": {
        "thesis": "Enter on stop-runs + reclaim of swing levels.",
        "coherent_regimes": ["trending", "ranging", "breakout"],
        "blurb": """\
Thesis: markets routinely sweep liquidity below swing lows (or above swing
highs) before continuing in the original direction. Entry on a confirmed
sweep + reclaim is one of the highest-edge setups in crypto.

Typical setups:
  - Price prints new 20-bar low BELOW prior swing low (the 'sweep')
  - Then reclaims the prior swing low within N bars (the 'reclaim')
  - Confirm with rising volume on the reclaim bar
  - Enter long on the close of the reclaim bar; stop below the sweep low

Indicators to consider: rolling lookback low (20-bar / 50-bar), volume,
ATR for stop sizing. Optional: btc_oi_pct_change_24h spiking negative
during the sweep = forced liquidation, extra confirmation.

Coherent regimes: works in all three directional regimes. In ranging,
sweeps mark range-low bounces (LiquiditySweepStrategy already exploits
this). In trending, sweeps mark continuation entries. In breakout, sweeps
often precede the genuine breakout.

NOT a generic 'all' regime — this is specifically a price-action setup
that needs a directional context to work. If you want regime-agnostic,
use funding_contrarian or alt_strength_divergence instead.

Threshold conventions: sweep depth = (sweep_low - prior_swing_low) / ATR.
A meaningful sweep is > 0.5 ATR. Reclaim window: 1-5 bars typically.""",
    },
}


# ---------------------------------------------------------------------------
# Coherence matrix
# ---------------------------------------------------------------------------

def coherence_matrix() -> list[tuple[str, str]]:
    """The full list of (archetype, regime) cells the generator should iterate.

    Returns a list of tuples; each tuple is one cell the generator will
    produce a strategy for. Excludes incoherent combinations like
    `mean_reversion` in `trending` markets.

    Currently 20 cells. Schedule and pool capacity in orchestrator.py /
    strategy_registry.py are sized for this number; if the matrix grows,
    bump those too.
    """
    cells = []
    for archetype, spec in ARCHETYPES.items():
        for regime in spec["coherent_regimes"]:
            cells.append((archetype, regime))
    return cells


def archetype_names() -> list[str]:
    """Just the archetype enum values, in definition order."""
    return list(ARCHETYPES.keys())


def prompt_blurb_for(archetype: str) -> str:
    """Return the verbose LLM guidance block for an archetype.

    Raises KeyError if the archetype isn't in our enum — caller should
    have validated the archetype before getting here. Letting it raise is
    deliberate: a silent fallback would mask the bug.
    """
    return ARCHETYPES[archetype]["blurb"]


def thesis_for(archetype: str) -> str:
    """One-sentence pitch — used in the prompt header line."""
    return ARCHETYPES[archetype]["thesis"]


def coherent_regimes_for(archetype: str) -> list[str]:
    """List of regimes this archetype can credibly target."""
    return list(ARCHETYPES[archetype]["coherent_regimes"])


def is_coherent(archetype: str, regime: str) -> bool:
    """Quick check: can this archetype legitimately target this regime?"""
    spec = ARCHETYPES.get(archetype)
    if not spec:
        return False
    return regime in spec["coherent_regimes"]
