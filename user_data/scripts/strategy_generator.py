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
You write Python strategies that extend BaseGeneratedStrategy for crypto futures (BTC/ETH/SOL on USDT).

RULES:
1. The strategy MUST extend BaseGeneratedStrategy (import with: from base_generated import BaseGeneratedStrategy)
2. You MUST implement: populate_indicators, populate_entry_trend, populate_exit_trend
3. You MUST set class attributes: STRATEGY_THESIS, TARGET_REGIME, GENERATION_ID
4. Only use these imports: freqtrade.strategy, pandas, pandas_ta (as ta), numpy (as np)
5. NO file I/O, NO network calls, NO exec/eval, NO os/sys/subprocess
6. NO .shift(-N) — that's look-ahead bias (accessing future data)
7. NO .rolling(center=True) — that's also look-ahead bias
8. NO ta.vwap() — it requires DatetimeIndex which breaks in Freqtrade backtesting. Use EMA of typical price ((high+low+close)/3) instead.
9. Always use .shift(1) or more to reference past data for signals
9. Use vectorized pandas operations, NO for loops over rows
10. Timeframe is 1h. startup_candle_count should be >= 200.

AVAILABLE INDICATORS (via pandas_ta):
- ta.ema, ta.sma, ta.rsi, ta.macd, ta.bbands, ta.adx, ta.atr
- ta.stoch, ta.willr, ta.cci, ta.mfi, ta.obv
- ta.alma, ta.kc, ta.donchian, ta.ichimoku
- numpy: np.where, np.nan
- DataFrame: .rolling(), .shift(), .pct_change(), .rank()

OUTPUT: Return ONLY the Python code. No explanations, no markdown fences, just the .py file content.
"""


def build_generation_prompt(
    target_regime: str = "all",
    context: str = "",
    existing_results: str = "",
    generation_id: str = "",
) -> str:
    """Build the user prompt for strategy generation."""

    prompt = f"""Generate a new Freqtrade trading strategy for crypto futures.

TARGET REGIME: {target_regime}
GENERATION ID: {generation_id}

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
) -> list:
    """Generate a batch of strategies across different regimes."""
    if regimes is None:
        regimes = ["trending", "ranging", "breakout", "all"]

    results = []
    for i in range(count):
        regime = regimes[i % len(regimes)]
        log.info(f"=== Generating strategy {i+1}/{count} for regime: {regime} ===")
        result = generate_strategy(
            target_regime=regime,
            context=context,
            existing_results=existing_results,
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
    args = parser.parse_args()

    if args.count == 1:
        result = generate_strategy(target_regime=args.regime, model=args.model)
        print(json.dumps({k: str(v) for k, v in result.items()}, indent=2))
    else:
        results = generate_batch(count=args.count)
        for r in results:
            print(json.dumps({k: str(v) for k, v in r.items()}, indent=2))
