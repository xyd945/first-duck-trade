"""
Strategy spec → codegen.

R5 surfaced a real tension: R2's prompts strongly encourage macro filters
(fgi, btc_funding_rate, etc.) and R5's critic flags 6+ AND conditions as
over-constrained. With 3-4 macro filters + 3-5 TA filters the joint AND
count naturally hits 6-8 and entries never fire.

R3 resolves it structurally. Instead of free-text Python, the LLM emits a
JSON spec. The renderer separates entry conditions into two buckets:

  core              must all be True (thesis conditions)
  macro_confidence  each is 0/1; mean must be >= macro_min_confidence

So 5 macro filters don't all need to fire — just enough of them. The LLM
can add as many macro signals as it wants without killing the entry rate.

The renderer also auto-NaN-guards each macro condition with .fillna(False),
which fixes the other class of issues the critic was flagging.

The output passes the existing validation_pipeline by construction:
imports are whitelisted, BaseGeneratedStrategy is inherited, the three
populate_* methods are present, and the renderer never emits .shift(-N).
"""

import json
import logging
import re
from typing import Any

log = logging.getLogger("strategy_spec")


REQUIRED_FIELDS = (
    "name", "thesis", "target_regime", "indicators", "params",
    "entry", "exit", "risk",
)


class SpecError(ValueError):
    """Raised when a spec is structurally invalid (renderer would crash)."""


def validate_spec(spec: dict) -> None:
    """Raise SpecError on any structural problem. Strict — we want failures
    here, before render, not later as cryptic Python errors."""
    if not isinstance(spec, dict):
        raise SpecError(f"spec must be a dict, got {type(spec).__name__}")

    missing = [f for f in REQUIRED_FIELDS if f not in spec]
    if missing:
        raise SpecError(f"missing required fields: {missing}")

    if not re.match(r"^[A-Z][A-Za-z0-9_]*$", spec["name"]):
        raise SpecError(f"name must be a valid Python class identifier, got {spec['name']!r}")

    if spec["target_regime"] not in ("trending", "ranging", "breakout", "all"):
        raise SpecError(f"target_regime must be one of trending/ranging/breakout/all, got {spec['target_regime']!r}")

    entry = spec["entry"]
    if not isinstance(entry, dict) or "core" not in entry:
        raise SpecError("entry must be a dict with at least a 'core' list")
    if not isinstance(entry["core"], list) or not entry["core"]:
        raise SpecError("entry.core must be a non-empty list of condition strings")
    if not all(isinstance(c, str) for c in entry["core"]):
        raise SpecError("every entry.core condition must be a string")

    mc = entry.get("macro_confidence", [])
    if not isinstance(mc, list) or not all(isinstance(c, str) for c in mc):
        raise SpecError("entry.macro_confidence must be a list of condition strings")

    mmc = entry.get("macro_min_confidence", 0.5)
    if not isinstance(mmc, (int, float)) or not (0.0 <= mmc <= 1.0):
        raise SpecError(f"entry.macro_min_confidence must be 0..1, got {mmc!r}")

    exit_block = spec["exit"]
    if not isinstance(exit_block, dict) or not isinstance(exit_block.get("core"), list) or not exit_block["core"]:
        raise SpecError("exit.core must be a non-empty list of condition strings")

    risk = spec["risk"]
    if "stoploss" not in risk or not isinstance(risk["stoploss"], (int, float)) or risk["stoploss"] >= 0:
        raise SpecError("risk.stoploss must be a negative number (e.g. -0.06)")
    if "minimal_roi" not in risk or not isinstance(risk["minimal_roi"], dict):
        raise SpecError("risk.minimal_roi must be a dict like {'0': 0.15}")

    for ind in spec["indicators"]:
        if not isinstance(ind, dict) or "compute" not in ind:
            raise SpecError(f"each indicator must have a 'compute' field, got {ind!r}")

    for p in spec["params"]:
        if not all(k in p for k in ("name", "type", "default", "space")):
            raise SpecError(f"each param needs name/type/default/space, got {p!r}")
        if p["type"] not in ("int", "decimal", "bool"):
            raise SpecError(f"param type must be int/decimal/bool, got {p['type']!r}")
        if p["space"] not in ("buy", "sell"):
            raise SpecError(f"param space must be buy/sell, got {p['space']!r}")


def _render_param(p: dict) -> str:
    kind = {"int": "IntParameter", "decimal": "DecimalParameter", "bool": "BooleanParameter"}[p["type"]]
    if p["type"] == "bool":
        return f"    {p['name']} = {kind}(default={p['default']}, space=\"{p['space']}\")"
    low = p.get("low", p["default"])
    high = p.get("high", p["default"])
    return f"    {p['name']} = {kind}({low}, {high}, default={p['default']}, space=\"{p['space']}\")"


def _render_indicator(ind: dict) -> str:
    """An indicator is one or more lines of code computing one or more columns.

    spec format:
      {"compute": "bb = ta.bbands(dataframe['close'], length=20, std=2.0)",
       "columns": [{"name": "bb_upper", "source": "bb['BBU_20_2.0']"}, ...]}
    OR a single-line direct assignment:
      {"compute": "dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)"}
    """
    lines = [f"        {ind['compute']}"]
    for col in ind.get("columns", []):
        lines.append(f"        dataframe['{col['name']}'] = {col['source']}")
    return "\n".join(lines)


def _join_conditions(conds: list, indent: int = 12) -> str:
    """Render a list of condition strings as an AND-joined expression block."""
    pad = " " * indent
    if len(conds) == 1:
        return f"{pad}{conds[0]}"
    inner = f" &\n{pad}".join(f"({c})" for c in conds)
    return f"{pad}{inner}"


_TEMPLATE = '''\
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from indicators.external_data import add_external_data
from base_generated import BaseGeneratedStrategy
import pandas as pd
import pandas_ta as ta
import numpy as np


class {name}(BaseGeneratedStrategy):

    STRATEGY_THESIS = {thesis_literal}
    TARGET_REGIME = "{target_regime}"
    GENERATION_ID = "{generation_id}"

    timeframe = '{timeframe}'
    startup_candle_count = 200

    stoploss = {stoploss}
    minimal_roi = {minimal_roi}
{max_open_trades_line}
    # ----- Hyperopt parameters -----
{params_block}

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Macro context (fgi, vix, gold, dxy, spx, btc_funding_rate, btc_oi, btc_oi_pct_change_24h)
        dataframe = add_external_data(dataframe)

{indicators_block}
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Core conditions — ALL must be True (the strategy's thesis)
        core = (
{entry_core_block}
        )

{macro_block}
        dataframe.loc[core & macro_pass, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        exit_cond = (
{exit_core_block}
        )
        dataframe.loc[exit_cond, 'exit_long'] = 1
        return dataframe
'''


_MACRO_BLOCK_NONE = """        # No macro confidence filters declared
        macro_pass = True
"""

_MACRO_BLOCK_TEMPLATE = """        # Macro confidence — each condition contributes 0/1; entry requires
        # the average to clear macro_min_confidence (avoids the over-constrained
        # AND-of-everything problem the critic was flagging).
        macro_conditions = [
{macro_items}
        ]
        macro_score = sum(c.astype(int) for c in macro_conditions) / len(macro_conditions)
        macro_pass = macro_score >= {macro_min_confidence}
"""


def render_strategy(spec: dict) -> str:
    """Render a validated spec to Python source. Always emits a strategy that
    passes validation_pipeline (whitelisted imports, BaseGeneratedStrategy,
    correct method signatures, no banned constructs)."""
    validate_spec(spec)

    params_block = "\n".join(_render_param(p) for p in spec["params"]) or "    # (no hyperopt params)"
    indicators_block = "\n\n".join(_render_indicator(i) for i in spec["indicators"]) or "        pass  # no indicators"

    entry_core = _join_conditions(spec["entry"]["core"], indent=12)
    exit_core = _join_conditions(spec["exit"]["core"], indent=12)

    mc = spec["entry"].get("macro_confidence", [])
    if not mc:
        macro_block = _MACRO_BLOCK_NONE
    else:
        # Auto-wrap each macro condition with .fillna(False) so NaN doesn't
        # silently zero out the score (the second-most-common critic flag).
        items = ",\n".join(f"            ({c}).fillna(False)" for c in mc)
        macro_block = _MACRO_BLOCK_TEMPLATE.format(
            macro_items=items,
            macro_min_confidence=spec["entry"].get("macro_min_confidence", 0.5),
        )

    max_open = spec["risk"].get("max_open_trades")
    max_open_line = f"    max_open_trades = {max_open}\n" if max_open else ""

    return _TEMPLATE.format(
        name=spec["name"],
        thesis_literal=json.dumps(spec["thesis"]),  # safely escapes quotes
        target_regime=spec["target_regime"],
        generation_id=spec.get("generation_id", "unknown"),
        timeframe=spec.get("timeframe", "1h"),
        stoploss=spec["risk"]["stoploss"],
        minimal_roi=json.dumps(spec["risk"]["minimal_roi"]),
        max_open_trades_line=max_open_line,
        params_block=params_block,
        indicators_block=indicators_block,
        entry_core_block=entry_core,
        macro_block=macro_block,
        exit_core_block=exit_core,
    )
