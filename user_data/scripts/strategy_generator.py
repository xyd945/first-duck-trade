"""
Strategy Generator — LLM-powered strategy creation.

Uses Claude API to generate Freqtrade strategy code based on:
  - Target regime (trending, ranging, breakout, all)
  - Available indicators and their documentation
  - Backtest results of existing strategies (what's working, what's not)
  - Market context (optional, from regime classifier)

The generator:
  1. Constructs a prompt with constraints and context
  2. Calls Claude API
  3. Extracts Python code from the response
  4. Saves to user_data/strategies/candidates/
  5. Runs validation pipeline
  6. Returns the result (pass/fail + file path)
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("strategy_generator")

BASE_DIR = Path(__file__).resolve().parent.parent  # user_data/
CANDIDATES_DIR = BASE_DIR / "strategies" / "candidates"
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert algorithmic trading strategy developer for the Freqtrade framework.
You write Python strategies that extend BaseGeneratedStrategy for SPOT crypto trading (BTC/ETH/SOL/XRP on USDT).

CRITICAL RULES:
1. The strategy MUST extend BaseGeneratedStrategy (import with: from base_generated import BaseGeneratedStrategy)
2. You MUST implement: populate_indicators, populate_entry_trend, populate_exit_trend
3. You MUST set class attributes: STRATEGY_THESIS, TARGET_REGIME, GENERATION_ID
4. Allowed imports: freqtrade.strategy, pandas, pandas_ta (as ta), numpy (as np),
   and `from indicators.external_data import add_external_data` (see EXTERNAL DATA below).
5. NO file I/O, NO network calls, NO exec/eval, NO os/sys/subprocess
6. NO .shift(-N) — that's look-ahead bias (accessing future data)
7. NO .rolling(center=True) — that's also look-ahead bias
8. NO ta.vwap() — it requires DatetimeIndex which breaks in Freqtrade backtesting
9. Always use .shift(1) or more to reference past data for signals
10. Use vectorized pandas operations, NO for loops over rows
11. Timeframe is 1h. startup_candle_count should be >= 200.
12. SPOT TRADING ONLY — LONG entries only. Do NOT set can_short = True. Do NOT generate short signals.
13. Entry signals use 'enter_long' column. Exit signals use 'exit_long' column.
14. The FIRST line of populate_indicators MUST be: `dataframe = add_external_data(dataframe)`.
    This injects macro context the strategy is encouraged (not required) to use.

EXTERNAL DATA (available via add_external_data — already shifted +1 day to avoid look-ahead):
  dataframe['fgi']    PROJECT-SPECIFIC composite (NOT the public 0-100
                      Alternative.me Fear & Greed Index). Empirical range on
                      our data is roughly -22 to +45, median ~5, std ~12.
                      Negative = Fear (oversold macro, contrarian-long signal),
                      Positive = Greed.
                      Useful thresholds (empirical, ~10% of days fall here):
                        fgi < -10   strong fear  (good contrarian long entry)
                        fgi > +20   strong greed (caution on momentum entries)
                      DO NOT compare against 50, 70, or any 0-100 scale — those
                      thresholds will silently never (or always) fire and ruin
                      the strategy. fgi < 0 fires roughly 45% of days.
  dataframe['vix']    CBOE Volatility Index daily close. High vix = panic.
                      Low vix (under ~18) historically favors trend continuation.
  dataframe['gold']   Gold futures close. Rising gold often means risk-off.
                      A falling gold + rising spx pair tends to be a risk-on signal.
  dataframe['dxy']    US Dollar Index. Strong dollar (rising dxy) is generally
                      a headwind for risk assets including crypto.
  dataframe['spx']    S&P 500 close. Crypto correlates with US equities most of
                      the time; spx weakness is often a leading signal for BTC.

CRYPTO POSITIONING (from BTC perpetuals on Binance Futures):
  dataframe['btc_funding_rate']        Last published 8h funding rate (decimal).
                                       Positive = longs paying shorts = market
                                       is long-loaded (exhaustion risk).
                                       Sustained > 0.0005 (5bp / 8h) is a frothy
                                       leverage signal; < 0 means shorts are
                                       paying (squeeze fuel).
  dataframe['btc_oi']                  BTC futures open interest in USD.
                                       Absolute scale — useful in ratios, not
                                       on its own.
  dataframe['btc_oi_pct_change_24h']   1-day % change in OI. Positive = new
                                       positions building (often trend continuation);
                                       sharply negative = forced de-leveraging
                                       (often marks short-term bottoms).

ALT STRENGTH / BTC DOMINANCE PROXY (ETH/BTC ratio from Binance spot daily):
  dataframe['eth_btc_ratio']           ETH/USDT ÷ BTC/USDT. Rising = alts
                                       outperforming BTC = BTC dominance
                                       falling. Falling = capital flight INTO
                                       BTC = BTC dominance rising.
  dataframe['eth_btc_change_7d']       7-day % change in the ratio. Captures
                                       the alt-strength momentum without the
                                       noise of a daily diff.
  dataframe['alt_strength_zscore_30d'] z-score of eth_btc_ratio over 30 days.
                                       > +1.5 = alt-season extreme; < -1.5 =
                                       capitulation into BTC (crisis-adjacent).
                                       Use as a regime filter: BTC-following
                                       strategies often work best when alts
                                       are weak (z < 0); reversion / breakout
                                       BTC plays often coincide with extreme
                                       capital flows in either direction.

These columns are 1d/8h data forward-filled to the strategy's 1h timeframe. They
may be NaN if the macro/perp feed hasn't run yet — wrap conditions in a NaN guard,
e.g. `dataframe['vix'].notna() & (dataframe['vix'] < 20)`.

PANDAS_TA COLUMN NAMING — THIS IS CRITICAL, get it right:
pandas_ta encodes parameters into column names. You MUST use the exact column names.
We use pandas_ta 0.3.16 (stable). The column naming is specific to this version.

IMPORTANT: To avoid column name mismatches with hyperopt parameters, ALWAYS use
HARDCODED literal values for indicator lengths/periods, NOT hyperopt parameter values.
Use hyperopt parameters only for thresholds and signal conditions, NOT for indicator computation.

ta.donchian(high, low, lower_length=N, upper_length=N):
  Columns: 'DCL_N_N', 'DCM_N_N', 'DCU_N_N'
  Example: ta.donchian(df['high'], df['low'], lower_length=20, upper_length=20)
    -> 'DCL_20_20', 'DCM_20_20', 'DCU_20_20'

ta.bbands(close, length=N, std=S):
  Columns: 'BBL_N_S', 'BBM_N_S', 'BBU_N_S', 'BBB_N_S', 'BBP_N_S'
  Example: ta.bbands(df['close'], length=20, std=2.0)
    -> 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'

ta.macd(close, fast=F, slow=S, signal=SIG):
  Columns: 'MACD_F_S_SIG', 'MACDh_F_S_SIG', 'MACDs_F_S_SIG'
  Example: ta.macd(df['close']) -> 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'

ta.stoch(high, low, close, k=K, d=D, smooth_k=SK):
  Columns: 'STOCHk_K_D_SK', 'STOCHd_K_D_SK'
  Example: ta.stoch(df['high'], df['low'], df['close']) -> 'STOCHk_14_3_3', 'STOCHd_14_3_3'

ta.kc(high, low, close, length=N, scalar=S):
  Columns: 'KCLe_N_S', 'KCBe_N_S', 'KCUe_N_S'  (note: S is float in column name)
  Example: ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
    -> 'KCLe_20_2.0', 'KCBe_20_2.0', 'KCUe_20_2.0'

ta.adx(high, low, close, length=N):
  Columns: 'ADX_N', 'DMP_N', 'DMN_N'
  Example: ta.adx(df['high'], df['low'], df['close'], length=14) -> 'ADX_14', 'DMP_14', 'DMN_14'

Simple indicators (return a single Series, assign directly):
  ta.ema(close, length=N), ta.sma(close, length=N), ta.rsi(close, length=N),
  ta.atr(high, low, close, length=N), ta.cci(high, low, close, length=N),
  ta.willr(high, low, close, length=N), ta.mfi(high, low, close, volume, length=N)

CORRECT PATTERN — hardcode indicator params, use hyperopt for thresholds:
  from indicators.external_data import add_external_data  # at top of file
  ...

  def populate_indicators(self, dataframe, metadata):
      # FIRST line — injects fgi, vix, gold, dxy, spx (see EXTERNAL DATA above)
      dataframe = add_external_data(dataframe)

      # In populate_indicators — use LITERAL values:
      bb = ta.bbands(dataframe['close'], length=20, std=2.0)
      dataframe['bb_upper'] = bb['BBU_20_2.0']
      dataframe['bb_lower'] = bb['BBL_20_2.0']
      dataframe['bb_mid'] = bb['BBM_20_2.0']
      dataframe['bb_pct'] = bb['BBP_20_2.0']

      donchian = ta.donchian(dataframe['high'], dataframe['low'], lower_length=20, upper_length=20)
      dataframe['dc_upper'] = donchian['DCU_20_20']
      dataframe['dc_lower'] = donchian['DCL_20_20']
      return dataframe

  # In populate_entry_trend — use hyperopt params for THRESHOLDS:
  rsi_oversold = IntParameter(20, 40, default=30, space="buy")
  # ... (dataframe['rsi'] < self.rsi_oversold.value) ...

STRONG ENCOURAGEMENT — these columns exist because they're alpha:
  Macro filter:    (dataframe['fgi'].fillna(0) < -10) & (dataframe['vix'].fillna(20) < 25)
  Positioning:     (dataframe['btc_funding_rate'].fillna(0) < 0.0003)   # not frothy
                   & (dataframe['btc_oi_pct_change_24h'].fillna(0) > -5) # not crashing

  Past LLM strategies that ignored funding/OI lost money because they
  entered into over-leveraged tops. STRONGLY consider gating entries on
  btc_funding_rate or btc_oi_pct_change_24h — they're leading signals
  the pure-TA columns can't see.

  Alt-strength regime: (dataframe['alt_strength_zscore_30d'].fillna(0) < 0.5)
    means BTC is taking share from alts — a useful environment for
    BTC trend strategies. Conversely, |zscore| > 1.5 marks regime extremes
    where mean reversion of the ratio itself often follows.

OUTPUT: Return ONLY the Python code. No explanations, no markdown fences, just the .py file content.
"""


def dedupe_class_name(filepath, class_name: str, name_exists) -> str:
    """If class_name collides with an already-registered strategy, rename the
    class inside the .py file to a unique variant and return the new name.

    The registry's `name` column is UNIQUE and Freqtrade loads strategies by
    class name — so a genuine collision means both the DB insert fails AND
    Freqtrade can't distinguish two classes with the same name. The LLM
    occasionally regenerates a class name that matches a retired strategy;
    without this helper that candidate was silently orphaned.

    Args:
        filepath: Path to the generated .py file.
        class_name: The class name extracted from the file.
        name_exists: Callable(name) -> bool. Typically
            `lambda n: get_strategy_by_name(n) is not None`.

    Returns:
        A class name guaranteed not to collide in the registry. If renaming
        was needed, the .py file on disk is rewritten in place.
    """
    from pathlib import Path

    if not name_exists(class_name):
        return class_name

    # Cap iterations defensively: if a pathological name_exists callback
    # claims every variant is taken, fail loudly instead of hanging.
    candidate = class_name
    for i in range(2, 1002):
        candidate = f"{class_name}_v{i}"
        if not name_exists(candidate):
            break
    else:
        raise RuntimeError(
            f"dedupe_class_name: could not find a free variant of {class_name} "
            f"after 1000 attempts"
        )

    fp = Path(filepath)
    source = fp.read_text()
    pattern = re.compile(rf'\bclass\s+{re.escape(class_name)}\b')
    new_source, n_replaced = pattern.subn(f"class {candidate}", source, count=1)
    if n_replaced == 0:
        raise ValueError(
            f"dedupe_class_name: could not find 'class {class_name}' in {fp}"
        )
    fp.write_text(new_source)
    log.info(f"Renamed class {class_name} -> {candidate} in {fp.name} (collision)")
    return candidate


def _format_failure_examples(failures: list) -> str:
    """Render failure rows from registry.get_recent_failures into a compact block."""
    if not failures:
        return ""
    parts = []
    for i, f in enumerate(failures, 1):
        thesis = (f.get("thesis") or "").strip() or "(no thesis recorded)"
        reason = (f.get("failure_reason") or "").strip() or "(no reason recorded)"
        verdict = f.get("failure_verdict") or "UNKNOWN"
        regime = f.get("target_regime") or "all"
        trades = f.get("total_trades")
        profit = f.get("profit_total_pct")
        sharpe = f.get("sharpe")
        metrics = []
        if trades is not None:
            metrics.append(f"trades={trades}")
        if profit is not None:
            metrics.append(f"profit={profit}%")
        if sharpe is not None:
            metrics.append(f"sharpe={sharpe}")
        metric_str = f" [{', '.join(metrics)}]" if metrics else ""
        block = (
            f"#{i} [{verdict}] regime={regime}{metric_str}\n"
            f"   thesis: {thesis}\n"
            f"   why it failed: {reason}"
        )
        excerpt = (f.get("code_excerpt") or "").strip()
        if excerpt:
            block += f"\n   entry logic:\n{_indent(excerpt, 6)}"
        parts.append(block)
    return "\n\n".join(parts)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in text.splitlines())


def build_generation_prompt(
    target_regime: str = "all",
    context: str = "",
    existing_results: str = "",
    generation_id: str = "",
    reflector_insights: str = "",
    failure_examples: str = "",
    attribution_patterns: str = "",
    archetype: str | None = None,
) -> str:
    """Build the user prompt for strategy generation.

    archetype: Phase 6 — if provided, the strategy MUST be of this archetype.
    The archetype's thesis + indicator/threshold guidance is injected at the
    TOP of the prompt (highest priority instruction) and the LLM is told the
    spec validator will reject a non-matching spec. Values must match the
    enum in archetypes.py.

    reflector_insights: strategic lessons the reflector agent wrote after
    reviewing live trades. Pre-rendered markdown from
    `registry.load_recent_reflections()`.

    failure_examples: pre-rendered block of prior strategies that failed
    backtest, with their failure reason + entry logic. From
    `_format_failure_examples(registry.get_recent_failures(...))`.

    attribution_patterns: pre-rendered macro-bucket attribution rollup —
    which conditions consistently favored wins vs losses in recent
    backtests of comparable strategies. From
    `trade_attribution.format_aggregate_for_generator(
        aggregate_attributions_by_bucket(rows, regime), target_regime)`.
    Sits ABOVE failure_examples because it's prescriptive ("aim here")
    vs prohibitive ("don't do that") — the LLM gets a positive target
    before the don't-list.
    """

    prompt = f"""Generate a new Freqtrade trading strategy for SPOT crypto trading (LONG only, no shorting).

TARGET REGIME: {target_regime}
GENERATION ID: {generation_id}

"""

    # Phase 6: archetype is the single most important constraint — put it
    # at the TOP of the prompt. The spec validator enforces this; a wrong
    # archetype = guaranteed rejection.
    if archetype:
        from archetypes import thesis_for, prompt_blurb_for
        prompt += f"""ARCHETYPE: {archetype}
{thesis_for(archetype)}

The spec you emit MUST set "archetype": "{archetype}" and the strategy logic
MUST follow this archetype's thesis. The spec validator will REJECT specs
whose archetype field doesn't match this instruction.

ARCHETYPE GUIDANCE:
{prompt_blurb_for(archetype)}

"""

    if reflector_insights:
        prompt += f"""LESSONS FROM RECENT REFLECTIONS (trade review agent):
These are observations from live paper-trading. Apply the takeaways where they fit.
{reflector_insights}

"""

    if attribution_patterns:
        prompt += f"""{attribution_patterns}

"""

    if failure_examples:
        prompt += f"""RECENT FAILURES TO AVOID (do NOT repeat these approaches):
Each entry is a prior LLM-generated strategy that failed. Learn from the failure mode:
if a thesis keeps losing money, it's not alpha — try a different setup, indicator family,
or regime. Do not re-propose anything with a near-identical entry logic.

{failure_examples}

"""

    if existing_results:
        prompt += f"""EXISTING STRATEGY RESULTS (learn from these):
{existing_results}

Based on these results, try a DIFFERENT approach. If existing strategies use EMA crossovers,
try Bollinger Bands or RSI mean-reversion. If they use momentum, try breakout or range strategies.
Avoid repeating approaches that already failed.

"""

    if context:
        prompt += f"""MARKET CONTEXT:
{context}

"""

    prompt += """Generate a complete strategy. Be creative but realistic.
The strategy should have clear entry/exit logic with at least 3 conditions each.
Include hyperopt parameters (IntParameter, DecimalParameter) for key values.
Set appropriate stoploss (-3% to -8%) and minimal_roi.
"""

    return prompt


# ---------------------------------------------------------------------------
# Code Extraction
# ---------------------------------------------------------------------------
def extract_python_code(response_text: str) -> str:
    """Extract Python code from LLM response, handling markdown fences."""
    # Try to find code in ```python ... ``` blocks
    match = re.search(r'```python\s*\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ``` blocks
    match = re.search(r'```\s*\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no fences, check if the whole response looks like Python
    if 'class ' in response_text and 'def populate_' in response_text:
        return response_text.strip()

    return response_text.strip()


# ---------------------------------------------------------------------------
# R3 — JSON-spec generator (replaces free-form Python output)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_SPEC = """You are an expert algorithmic trading strategy developer for Freqtrade.
You design strategies as a JSON SPEC. A code generator translates your spec into a
Python class — you NEVER write Python directly. The structural separation between
`core` (must-be-true thesis conditions) and `macro_confidence` (mean must clear
a threshold) is what makes this work: it lets you use rich macro context without
over-constraining the entry into never-firing AND-of-everything logic.

OUTPUT FORMAT — return ONE JSON object, no prose, no markdown fences:

{
  "name": "PascalCaseClassName",
  "thesis": "One sentence describing why this should produce alpha.",
  "archetype": "momentum_continuation" | "mean_reversion" | "breakout_volume" |
               "vol_squeeze" | "vol_compression_mean_reversion" |
               "funding_contrarian" | "oi_cascade_followthrough" |
               "alt_strength_divergence" | "macro_led_risk_on" |
               "liquidity_sweep_followthrough",
  "target_regime": "trending" | "ranging" | "breakout" | "all",
  "timeframe": "1h",
  "indicators": [
    {"compute": "bb = ta.bbands(dataframe['close'], length=20, std=2.0)",
     "columns": [
       {"name": "bb_lower", "source": "bb['BBL_20_2.0']"},
       {"name": "bb_upper", "source": "bb['BBU_20_2.0']"}
     ]},
    {"compute": "dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)"},
    {"compute": "dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=14)"}
  ],
  "params": [
    {"name": "rsi_oversold", "type": "int",     "low": 20,    "high": 40,    "default": 30,   "space": "buy"},
    {"name": "bb_pos",       "type": "decimal", "low": 0.0,   "high": 0.3,   "default": 0.1,  "space": "buy"},
    {"name": "rsi_exit",     "type": "int",     "low": 60,    "high": 80,    "default": 70,   "space": "sell"}
  ],
  "entry": {
    "core": [
      "dataframe['rsi'].shift(1) < self.rsi_oversold.value",
      "dataframe['close'].shift(1) < dataframe['bb_lower'].shift(1) * (1 + self.bb_pos.value)"
    ],
    "macro_confidence": [
      "dataframe['fgi'] < 0",
      "dataframe['btc_funding_rate'] < 0.0003",
      "dataframe['vix'] < 25"
    ],
    "macro_min_confidence": 0.5
  },
  "exit": {
    "core": [
      "dataframe['rsi'] > self.rsi_exit.value"
    ]
  },
  "risk": {
    "stoploss": -0.05,
    "minimal_roi": {"0": 0.10, "60": 0.05, "240": 0.02},
    "max_open_trades": 3
  }
}

HARD RULES — your spec will be rejected if any of these are violated:

1. `entry.core` MUST be 2-4 conditions. These are your thesis. All must AND-true.
2. `entry.macro_confidence` is 0-N conditions. The renderer averages them as
   0/1 and requires the mean to clear `macro_min_confidence` (0.3-0.7 typical).
   You can list 5 macro signals — they DON'T all need to fire. This is the
   point of having two buckets. Use it.
3. Every `dataframe['x']` reference in `entry.core`, `entry.macro_confidence`,
   or `exit.core` MUST trace back to a declared column: base OHLCV
   (open/high/low/close/volume), an `indicators[]` declaration (Form A name
   or Form B columns[].name), or the external-data list below. The validator
   cross-checks this — referencing an undeclared column (e.g. using
   `dataframe['rsi']` in exit but not adding RSI to indicators) is
   auto-rejected with the list of columns you actually declared.
4. Use `.shift(1)` on every column reference in `entry.core` that compares
   "what just happened" — never compare today's close to today's indicator,
   that's same-bar look-ahead.
5. Use HARDCODED literal lengths/periods in indicator `compute` strings.
   Hyperopt params are for THRESHOLDS in entry/exit, never for indicator
   lengths — pandas_ta encodes length into column names so changing length
   breaks the column lookup.
6. SPOT, LONG-ONLY. No `can_short`. No short conditions.
7. risk.stoploss must be negative. Suggested range -0.03 to -0.10.
8. Each `indicators[]` entry uses EXACTLY ONE of two forms. Mixing them is the
   #1 LLM-produced bug in this pipeline and is auto-rejected by the validator.

INDICATOR FORMS — pick exactly one per entry:

  Form A (single column, inline): assign directly to dataframe in `compute`,
  NO `columns` block.
    {"compute": "dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)"}

  Form B (multi-column, local var): `compute` introduces a local variable,
  `columns` extracts named values from it. The renderer turns each entry
  into `dataframe['<name>'] = <source>`.
    {"compute": "bb = ta.bbands(dataframe['close'], length=20, std=2.0)",
     "columns": [
       {"name": "bb_lower", "source": "bb['BBL_20_2.0']"},
       {"name": "bb_upper", "source": "bb['BBU_20_2.0']"}
     ]}

  WRONG — DO NOT WRITE THIS. The renderer would emit two assignments; the
  second references an undefined local. The validator rejects it.
    {"compute": "dataframe['ema_20'] = ta.ema(dataframe['close'], length=20)",
     "columns": [{"name": "ema_20", "source": "dataframe['ema_20']"}]}
    Reason: `compute` already assigned to dataframe. The `columns` block
    must NOT be present. Drop it, or switch to Form B with a local var.

EXTERNAL DATA COLUMNS (always available, the renderer wires them in):
  dataframe['fgi']                     Fear & Greed composite. Negative = fear (contrarian-long signal).
  dataframe['vix']                     CBOE Volatility Index. <18 favors trend continuation; >30 = panic.
  dataframe['gold']                    Gold futures close. Rising gold = risk-off.
  dataframe['dxy']                     US Dollar Index. Strong dollar = headwind for crypto.
  dataframe['spx']                     S&P 500 close. Crypto correlates with US equities most days.
  dataframe['btc_funding_rate']        Last 8h funding rate (decimal). > 0.0005 = frothy long-loaded.
  dataframe['btc_oi']                  BTC futures open interest in USD.
  dataframe['btc_oi_pct_change_24h']   24h % change in OI. Positive = positions building.

These may be NaN early-history; the renderer wraps each macro_confidence
condition with `.fillna(False)` automatically so you don't have to.

PANDAS_TA COLUMN NAMING — these are encoded in the column name:
  ta.bbands(close, length=20, std=2.0)         -> BBL_20_2.0 / BBM_20_2.0 / BBU_20_2.0 / BBB_20_2.0 / BBP_20_2.0
  ta.donchian(high, low, lower_length=20, upper_length=20) -> DCL_20_20 / DCM_20_20 / DCU_20_20
  ta.macd(close)                               -> MACD_12_26_9 / MACDh_12_26_9 / MACDs_12_26_9
  ta.adx(high, low, close, length=14)          -> ADX_14 / DMP_14 / DMN_14
  ta.stoch(high, low, close)                   -> STOCHk_14_3_3 / STOCHd_14_3_3
  ta.kc(high, low, close, length=20, scalar=2) -> KCLe_20_2.0 / KCBe_20_2.0 / KCUe_20_2.0

Single-Series indicators (assign directly via `compute`):
  ta.ema, ta.sma, ta.rsi, ta.atr, ta.cci, ta.willr, ta.mfi, ta.obv

OUTPUT: a single JSON object. No prose. No fences. Just JSON.
"""


def _extract_spec_json(text: str) -> dict | None:
    """Pull a JSON object out of the LLM's response. Tolerates accidental
    markdown fences and surrounding prose. Returns None if no parseable object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate_strategy(
    target_regime: str = "all",
    context: str = "",
    existing_results: str = "",
    model: str | None = None,
    max_retries: int = 1,
    reflector_insights: str = "",
    failure_examples: str = "",
    attribution_patterns: str = "",
    provider: str | None = None,
    archetype: str | None = None,
) -> dict:
    """
    Generate a new strategy via the LLM.

    `model` / `provider` are forwarded to llm_client.chat_completion. When
    None, the env-driven defaults apply (LLM_PROVIDER, then that provider's
    default model — see PROVIDER_DEFAULTS in llm_client.py).

    Returns dict with:
      - success: bool
      - filepath: Path (if successful)
      - validation: ValidationResult
      - generation_id: str
      - error: str (if failed)
    """
    # Import validation pipeline + the LLM wrapper. llm_client lazily imports
    # the SDKs it needs, so a missing anthropic install only matters if the
    # caller actually selects the anthropic provider.
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from validation_pipeline import validate_strategy_file
    from llm_client import chat_completion

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    generation_id = f"gen-{timestamp}"

    for attempt in range(max_retries + 1):
        attempt_id = f"{generation_id}-v{attempt}"
        log.info(f"Generating strategy: {attempt_id} (regime={target_regime})")

        prompt = build_generation_prompt(
            target_regime=target_regime,
            context=context,
            existing_results=existing_results,
            generation_id=attempt_id,
            reflector_insights=reflector_insights,
            failure_examples=failure_examples,
            attribution_patterns=attribution_patterns,
            archetype=archetype,
        )

        try:
            raw_text = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system=SYSTEM_PROMPT_SPEC,
                model=model,
                max_tokens=4096,
                provider=provider,
            )
            spec = _extract_spec_json(raw_text)

            if spec is None:
                if attempt < max_retries:
                    log.warning(f"Spec parse failed (attempt {attempt}), retrying.")
                    existing_results += (
                        "\n\nPREVIOUS ATTEMPT RETURNED INVALID JSON — emit a single "
                        "JSON object only, no prose, no markdown fences."
                    )
                    continue
                return {"success": False, "generation_id": attempt_id,
                        "error": "spec parse failed after retries"}

            spec["generation_id"] = attempt_id

            from strategy_spec import render_strategy, validate_spec, SpecError
            try:
                validate_spec(spec)
            except SpecError as e:
                if attempt < max_retries:
                    log.warning(f"Spec invalid (attempt {attempt}): {e}. Retrying.")
                    existing_results += (
                        f"\n\nPREVIOUS SPEC WAS STRUCTURALLY INVALID:\n  {e}\nFix the spec."
                    )
                    continue
                return {"success": False, "generation_id": attempt_id,
                        "error": f"spec invalid after retries: {e}"}

            code = render_strategy(spec)

            safe_regime = target_regime.replace("/", "_")
            filename = f"Strategy_{safe_regime}_{timestamp}_v{attempt}.py"
            filepath = CANDIDATES_DIR / filename

            filepath.write_text(code)
            log.info(f"Strategy written to: {filepath} (rendered from spec, class={spec.get('name')})")

            # Validate the rendered code — should always pass by construction;
            # safety net catches renderer bugs.
            result = validate_strategy_file(filepath)
            log.info(f"Validation: {result}")

            if not result.passed:
                # Validation failure — feed back and retry (or give up)
                if attempt < max_retries:
                    log.warning(
                        f"Validation failed (attempt {attempt}), retrying with error feedback..."
                    )
                    existing_results += (
                        f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n"
                        f"{result}\n"
                        f"Fix these issues in the next attempt."
                    )
                    continue
                return {
                    "success": False,
                    "filepath": filepath,
                    "validation": result,
                    "generation_id": attempt_id,
                    "error": f"Validation failed after {max_retries + 1} attempts",
                }

            # Critic pass (judgment review: over-constrained logic, NaN guards,
            # regime/logic mismatch). Always non-blocking on its own errors.
            # Critic uses the same provider/model as the generator by default —
            # passing model=None lets llm_client pick the provider default.
            from strategy_critic import critic_review, format_critic_feedback
            critic = critic_review(code, model=model, provider=provider)
            verdict = critic.get("verdict", "PASS")
            log.info(f"Critic verdict: {verdict} — {critic.get('summary', '')}")

            if verdict == "REJECT" and attempt < max_retries:
                feedback = format_critic_feedback(critic)
                log.warning(f"Critic REJECTED (attempt {attempt}), retrying with feedback:\n{feedback}")
                existing_results += (
                    f"\n\nPREVIOUS ATTEMPT REJECTED BY CRITIC:\n{feedback}\n"
                    f"Address these issues in the next attempt."
                )
                continue

            # PASS, WARN, or REJECT with no retries left — return the strategy.
            # REJECT-no-retries proceeds anyway: the strategy will likely fail
            # backtest, which is the next gate. WARN proceeds with note logged.
            if verdict == "WARN":
                for issue in critic.get("issues", []):
                    log.warning(f"  Critic WARN [{issue.get('severity','?')}/{issue.get('category','?')}]: {issue.get('description','')}")
            elif verdict == "REJECT":
                log.warning(f"Critic REJECTED but no retries left — proceeding to backtest anyway")
            return {
                "success": True,
                "filepath": filepath,
                "validation": result,
                "critic": critic,
                "generation_id": attempt_id,
                "class_name": spec.get("name"),
            }

        except Exception as e:
            log.error(f"Generation attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                return {
                    "success": False,
                    "generation_id": attempt_id,
                    "error": str(e),
                }

    return {"success": False, "error": "Exhausted all retries"}


def _summarize_backtest_for_llm(bt: dict) -> str:
    """Render a mini-backtest result into a short, LLM-friendly diagnostic.

    Different result shapes get different framings because what the LLM needs
    to fix is very different in each case.
    """
    if not bt.get("success"):
        return (
            f"The strategy was generated but FAILED to backtest: {bt.get('error', 'unknown')}. "
            "Check for import errors, missing columns referenced in entry/exit logic, "
            "or syntax issues in the spec's expressions."
        )
    trades = bt.get("total_trades", 0)
    profit = bt.get("profit_total_pct", 0.0)
    sharpe = bt.get("sharpe", 0.0)
    drawdown = bt.get("max_drawdown_pct", 0.0)

    if trades == 0:
        return (
            "Your previous strategy produced ZERO TRADES on a 30-day backtest. "
            "The core conditions never simultaneously evaluated True. "
            "Diagnose: either tighten the number of `entry.core` conditions (3 max), "
            "loosen the thresholds in your IntParameter/DecimalParameter defaults, "
            "or move some core conditions into `macro_confidence` so they don't all need to fire."
        )
    if trades < 5:
        return (
            f"Your previous strategy produced only {trades} trades over 30 days — "
            f"far too few to evaluate. Loosen entry thresholds or relax the macro_min_confidence "
            f"(try 0.3 instead of 0.5)."
        )
    if profit <= 0 and sharpe <= 0:
        return (
            f"Your previous strategy traded {trades} times over 30 days but lost money "
            f"({profit:.2f}% return, sharpe {sharpe:.2f}, drawdown {drawdown:.2f}%). "
            f"The entry timing or exit conditions are wrong — reconsider the thesis. "
            f"Try a different indicator family, invert the entry direction, or tighten exits."
        )
    return (
        f"Your previous strategy was passable: {trades} trades, {profit:.2f}% profit, "
        f"sharpe {sharpe:.2f}. See if you can improve it further."
    )


def generate_and_iterate(
    target_regime: str = "all",
    context: str = "",
    existing_results: str = "",
    reflector_insights: str = "",
    failure_examples: str = "",
    attribution_patterns: str = "",
    model: str | None = None,
    provider: str | None = None,
    archetype: str | None = None,
    max_turns: int = 3,
    accept_min_trades: int = 10,
    backtest_fn=None,
) -> dict:
    """Generate → mini-backtest → refine, up to max_turns times.

    On each turn:
      1. Call generate_strategy (which already does validation + critic with
         their own internal retries).
      2. Run an out-of-sample 90-day mini-backtest on a slice that ends
         30 days before today — the orchestrator's full backtest will
         re-evaluate on all data including those 30 most-recent days.
      3. Score the result. ACCEPT if trades >= accept_min_trades AND
         profit > 0 AND sharpe > 0. Looser-than-full thresholds (full
         requires trades >= 20), but tightened from the original
         "profit > 0 OR sharpe > 0" because trials #5/#6 showed that
         OR-acceptance let near-zero-edge strategies through, then they
         consumed full-backtest slots and got retired anyway.
      4. If not accepted and turns remain, append a structured backtest
         diagnostic to existing_results and loop.
      5. After max_turns, return the BEST attempt seen (most trades + best
         profit) with `iterated=True` and `turns_used=N` flags.

    backtest_fn is injected for testability. Defaults to backtest_runner.run_mini_backtest
    with its out-of-sample window defaults.
    """
    if backtest_fn is None:
        # Lazy import — avoids hard dep when only the generator is exercised in tests.
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from backtest_runner import run_mini_backtest

        def backtest_fn(strategy_name: str) -> dict:
            return run_mini_backtest(strategy_name)

    best: dict | None = None
    best_score = -float("inf")
    accumulated_feedback = existing_results

    for turn in range(max_turns):
        log.info(f"=== Iterative generation, turn {turn + 1}/{max_turns} for regime={target_regime} ===")
        gen = generate_strategy(
            target_regime=target_regime,
            context=context,
            existing_results=accumulated_feedback,
            reflector_insights=reflector_insights,
            failure_examples=failure_examples,
            attribution_patterns=attribution_patterns,
            model=model,
            provider=provider,
            archetype=archetype,
        )
        if not gen.get("success"):
            # generate_strategy itself failed (spec parse, validation, etc.). Give up.
            log.warning(f"  Generation itself failed on turn {turn + 1}: {gen.get('error')}")
            return gen

        strategy_class = gen.get("class_name") or "unknown"
        log.info(f"  Generation succeeded ({strategy_class}). Running mini-backtest...")

        bt = backtest_fn(strategy_class)
        log.info(
            f"  Mini-backtest: success={bt.get('success')}, "
            f"trades={bt.get('total_trades', 0)}, "
            f"profit={bt.get('profit_total_pct', 0)}%, "
            f"sharpe={bt.get('sharpe', 0)}"
        )

        # Score the attempt — trade count dominates (a strategy that fires
        # can be rescued by hyperopt; a 0-trade strategy cannot). Profit and
        # sharpe break ties when trade counts are similar.
        trades = bt.get("total_trades", 0)
        profit = bt.get("profit_total_pct", 0.0) or 0.0
        sharpe = bt.get("sharpe", 0.0) or 0.0
        score = trades * 100 + profit * 10 + sharpe
        if score > best_score:
            best_score = score
            best = {**gen, "mini_backtest": bt, "turn": turn + 1}

        # Acceptance check. Both profit AND sharpe required — see docstring
        # for the trial-#5/#6 false-positive rationale.
        accepted = (
            bt.get("success")
            and trades >= accept_min_trades
            and profit > 0
            and sharpe > 0
        )
        if accepted:
            log.info(f"  ACCEPTED on turn {turn + 1}")
            return {**gen, "mini_backtest": bt, "iterated": True, "turns_used": turn + 1, "accepted": True}

        if turn + 1 < max_turns:
            diag = _summarize_backtest_for_llm(bt)
            log.info(f"  Not accepted — feeding back to LLM for refinement:\n    {diag}")
            accumulated_feedback += f"\n\nPREVIOUS ATTEMPT BACKTEST RESULT:\n{diag}"

    # Exhausted turns — return the best attempt we saw, marked as not accepted
    log.warning(f"  Iterative generation exhausted {max_turns} turns without acceptance")
    if best is None:
        return {"success": False, "error": "no attempts succeeded"}
    return {**best, "iterated": True, "turns_used": max_turns, "accepted": False}


def generate_batch(
    count: int = 5,
    regimes: list = None,
    cells: list[tuple[str, str]] = None,
    context: str = "",
    existing_results: str = "",
    reflector_insights: str = "",
    get_failures_for_regime=None,
    get_attribution_for_regime=None,
    iterative: bool = False,
    max_turns: int = 3,
) -> list:
    """Generate a batch of strategies.

    Two modes:
      (Phase 6, preferred) cells: list of (archetype, regime) tuples
          Iterates the coherence matrix — one strategy per cell, each
          baked to a specific archetype × regime. The archetype is
          enforced by the spec validator. Use archetypes.coherence_matrix()
          to get the full 20-cell list.

      (legacy) regimes + count
          Cycles through regimes for `count` iterations with no
          archetype constraint. Kept for backward compat — old tests,
          ad-hoc CLI runs — but the orchestrator should use cells.

    get_failures_for_regime: optional callable(regime: str) -> str
        Returns a pre-formatted failure_examples block for the given regime.
        Lets the caller inject per-regime failure memory without re-querying
        the registry here.

    get_attribution_for_regime: optional callable(regime: str) -> str
        Returns a pre-formatted historical attribution patterns block for the
        given regime. Same callable pattern as get_failures_for_regime so the
        orchestrator can fan registry queries out cleanly.
    """
    # Build the list of (archetype, regime) jobs to run, where archetype
    # may be None for legacy mode.
    if cells is not None:
        jobs = list(cells)
        log.info(f"Batch: iterating {len(jobs)} (archetype, regime) cells")
    else:
        if regimes is None:
            regimes = ["trending", "ranging", "breakout", "all"]
        jobs = [(None, regimes[i % len(regimes)]) for i in range(count)]
        log.info(f"Batch: legacy mode, {len(jobs)} regime-only generations")

    results = []
    for i, (archetype, regime) in enumerate(jobs):
        tag = f"{archetype or 'no-archetype'} × {regime}"
        log.info(f"=== Generating strategy {i+1}/{len(jobs)}: {tag} ===")
        failures = get_failures_for_regime(regime) if get_failures_for_regime else ""
        attribution = get_attribution_for_regime(regime) if get_attribution_for_regime else ""
        if iterative:
            result = generate_and_iterate(
                target_regime=regime,
                context=context,
                existing_results=existing_results,
                reflector_insights=reflector_insights,
                failure_examples=failures,
                attribution_patterns=attribution,
                archetype=archetype,
                max_turns=max_turns,
            )
        else:
            result = generate_strategy(
                target_regime=regime,
                context=context,
                existing_results=existing_results,
                reflector_insights=reflector_insights,
                failure_examples=failures,
                attribution_patterns=attribution,
                archetype=archetype,
            )
        # Stamp the requested archetype + regime onto the result so the
        # caller can register it without re-parsing the file. (The file
        # also stores STRATEGY_ARCHETYPE as a class attr — see strategy_spec.)
        result["archetype"] = archetype
        result["target_regime"] = regime
        results.append(result)

        if result.get("success"):
            log.info(f"  SUCCESS: {result['filepath']}")
        else:
            log.warning(f"  FAILED: {result.get('error', 'unknown')}")

    passed = sum(1 for r in results if r.get("success"))
    log.info(f"Batch complete: {passed}/{len(jobs)} strategies passed validation")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Generate trading strategies using LLM")
    parser.add_argument("--regime", default="all", help="Target regime")
    parser.add_argument("--count", type=int, default=1, help="Number of strategies to generate")
    parser.add_argument("--model", default=None,
                        help="Override model ID; default per LLM provider in llm_client.py")
    parser.add_argument("--provider", default=None,
                        help="Override LLM provider (anthropic / deepseek); default from LLM_PROVIDER env")
    parser.add_argument(
        "--dry-prompt",
        action="store_true",
        help="Print the assembled prompt using real registry data, without calling the LLM",
    )
    parser.add_argument(
        "--failure-k", type=int, default=8,
        help="How many recent failures to inject (for --dry-prompt)",
    )
    args = parser.parse_args()

    if args.dry_prompt:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_registry import get_recent_failures, load_recent_reflections

        failures = get_recent_failures(k=args.failure_k, regime=args.regime)
        failure_block = _format_failure_examples(failures)
        reflections = load_recent_reflections(n=2)

        prompt = build_generation_prompt(
            target_regime=args.regime,
            context="(dry-run — no live regime state injected)",
            existing_results="(dry-run — no registry stats injected)",
            generation_id="dry-run",
            reflector_insights=reflections,
            failure_examples=failure_block,
        )
        print("=" * 72)
        print("SYSTEM PROMPT")
        print("=" * 72)
        print(SYSTEM_PROMPT)
        print("=" * 72)
        print(f"USER PROMPT  (failures injected: {len(failures)}, "
              f"reflection chars: {len(reflections)})")
        print("=" * 72)
        print(prompt)
        sys.exit(0)

    if args.count == 1:
        result = generate_strategy(
            target_regime=args.regime, model=args.model, provider=args.provider,
        )
        print(json.dumps({k: str(v) for k, v in result.items()}, indent=2))
    else:
        results = generate_batch(count=args.count)
        for r in results:
            print(json.dumps({k: str(v) for k, v in r.items()}, indent=2))
