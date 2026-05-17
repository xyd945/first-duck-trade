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
  dataframe['fgi']    Fear & Greed composite (PMACD + RoR + Money Flow + VIX + Gold).
                      Negative = Fear (extreme oversold macro), Positive = Greed.
                      Useful as a contrarian filter (e.g. only long when fgi < -10
                      = market is fearful = better risk/reward for longs).
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
) -> str:
    """Build the user prompt for strategy generation.

    reflector_insights: strategic lessons the reflector agent wrote after
    reviewing live trades. Pre-rendered markdown from
    `registry.load_recent_reflections()`.

    failure_examples: pre-rendered block of prior strategies that failed
    backtest, with their failure reason + entry logic. From
    `_format_failure_examples(registry.get_recent_failures(...))`.
    """

    prompt = f"""Generate a new Freqtrade trading strategy for SPOT crypto trading (LONG only, no shorting).

TARGET REGIME: {target_regime}
GENERATION ID: {generation_id}

"""

    if reflector_insights:
        prompt += f"""LESSONS FROM RECENT REFLECTIONS (trade review agent):
These are observations from live paper-trading. Apply the takeaways where they fit.
{reflector_insights}

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
# Generator
# ---------------------------------------------------------------------------
def generate_strategy(
    target_regime: str = "all",
    context: str = "",
    existing_results: str = "",
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 1,
    reflector_insights: str = "",
    failure_examples: str = "",
) -> dict:
    """
    Generate a new strategy using Claude API.

    Returns dict with:
      - success: bool
      - filepath: Path (if successful)
      - validation: ValidationResult
      - generation_id: str
      - error: str (if failed)
    """
    try:
        import anthropic
    except ImportError:
        return {"success": False, "error": "anthropic package not installed"}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"success": False, "error": "ANTHROPIC_API_KEY not set"}

    # Import validation pipeline
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from validation_pipeline import validate_strategy_file

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    generation_id = f"gen-{timestamp}"

    client = anthropic.Anthropic(api_key=api_key)

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
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_text = response.content[0].text
            code = extract_python_code(raw_text)

            # Save to file
            safe_regime = target_regime.replace("/", "_")
            filename = f"Strategy_{safe_regime}_{timestamp}_v{attempt}.py"
            filepath = CANDIDATES_DIR / filename

            filepath.write_text(code)
            log.info(f"Strategy written to: {filepath}")

            # Validate
            result = validate_strategy_file(filepath)
            log.info(f"Validation: {result}")

            if result.passed:
                return {
                    "success": True,
                    "filepath": filepath,
                    "validation": result,
                    "generation_id": attempt_id,
                }

            # If validation failed and we have retries left, feed error back
            if attempt < max_retries:
                log.warning(
                    f"Validation failed (attempt {attempt}), retrying with error feedback..."
                )
                existing_results += (
                    f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n"
                    f"{result}\n"
                    f"Fix these issues in the next attempt."
                )
            else:
                return {
                    "success": False,
                    "filepath": filepath,
                    "validation": result,
                    "generation_id": attempt_id,
                    "error": f"Validation failed after {max_retries + 1} attempts",
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


def generate_batch(
    count: int = 5,
    regimes: list = None,
    context: str = "",
    existing_results: str = "",
    reflector_insights: str = "",
    get_failures_for_regime=None,
) -> list:
    """Generate a batch of strategies across different regimes.

    get_failures_for_regime: optional callable(regime: str) -> str
        Returns a pre-formatted failure_examples block for the given regime.
        Lets the caller inject per-regime failure memory without re-querying
        the registry here.
    """
    if regimes is None:
        regimes = ["trending", "ranging", "breakout", "all"]

    results = []
    for i in range(count):
        regime = regimes[i % len(regimes)]
        log.info(f"=== Generating strategy {i+1}/{count} for regime: {regime} ===")
        failures = get_failures_for_regime(regime) if get_failures_for_regime else ""
        result = generate_strategy(
            target_regime=regime,
            context=context,
            existing_results=existing_results,
            reflector_insights=reflector_insights,
            failure_examples=failures,
        )
        results.append(result)

        if result["success"]:
            log.info(f"  SUCCESS: {result['filepath']}")
        else:
            log.warning(f"  FAILED: {result.get('error', 'unknown')}")

    passed = sum(1 for r in results if r["success"])
    log.info(f"Batch complete: {passed}/{count} strategies passed validation")
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
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument(
        "--dry-prompt",
        action="store_true",
        help="Print the assembled prompt using real registry data, without calling Claude",
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
        result = generate_strategy(target_regime=args.regime, model=args.model)
        print(json.dumps({k: str(v) for k, v in result.items()}, indent=2))
    else:
        results = generate_batch(count=args.count)
        for r in results:
            print(json.dumps({k: str(v) for k, v in r.items()}, indent=2))
