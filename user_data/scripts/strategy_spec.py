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
    "name", "thesis", "archetype", "target_regime", "indicators", "params",
    "entry", "exit", "risk",
)


# Columns the renderer guarantees are present without any indicator declaration:
# OHLCV from Freqtrade + macro columns from add_external_data().
_OHLCV_COLUMNS = frozenset(("open", "high", "low", "close", "volume"))
_MACRO_COLUMNS = frozenset((
    "fgi", "vix", "gold", "dxy", "spx",
    "btc_funding_rate", "btc_oi", "btc_oi_pct_change_24h",
    "eth_btc_ratio", "eth_btc_change_7d", "alt_strength_zscore_30d",
))


# Catches `dataframe['x']` and `dataframe["x"]` references. Used both for
# detecting assignments in compute strings (when followed by `=`) and for
# detecting references in entry/exit conditions.
_COL_REF_RE = re.compile(r"""dataframe\[\s*['"]([^'"\]]+)['"]\s*\]""")
_COL_ASSIGN_RE = re.compile(r"""dataframe\[\s*['"]([^'"\]]+)['"]\s*\]\s*=(?!=)""")


def _declared_columns(spec: dict) -> set[str]:
    """Set of column names that will exist on the dataframe at entry/exit time.

    Includes OHLCV, macro injected by add_external_data, plus anything an
    indicator declares — either via inline ``dataframe['x'] = ...`` in its
    compute string, or via the ``columns: [{name: x, source: ...}]`` block
    (the renderer turns each into ``dataframe['x'] = <source>``).
    """
    declared = set(_OHLCV_COLUMNS) | set(_MACRO_COLUMNS)
    for ind in spec.get("indicators", []):
        for m in _COL_ASSIGN_RE.finditer(ind.get("compute", "")):
            declared.add(m.group(1))
        for col in ind.get("columns", []):
            name = col.get("name") if isinstance(col, dict) else None
            if name:
                declared.add(name)
    return declared


def _condition_strings(spec: dict) -> list[tuple[str, str]]:
    """Yield (section, condition_string) pairs from entry/exit logic."""
    out = []
    entry = spec.get("entry", {})
    for c in entry.get("core", []) or []:
        out.append(("entry.core", c))
    for c in entry.get("macro_confidence", []) or []:
        out.append(("entry.macro_confidence", c))
    for c in (spec.get("exit", {}).get("core", []) or []):
        out.append(("exit.core", c))
    return out


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

    # Phase 6: archetype is enforced. Lazy import so non-spec callers don't
    # need to load the archetype catalog.
    from archetypes import archetype_names, is_coherent
    valid_archetypes = archetype_names()
    if spec["archetype"] not in valid_archetypes:
        raise SpecError(
            f"archetype must be one of {valid_archetypes}, got {spec['archetype']!r}"
        )
    if not is_coherent(spec["archetype"], spec["target_regime"]):
        raise SpecError(
            f"archetype {spec['archetype']!r} does not cohere with target_regime "
            f"{spec['target_regime']!r} — see archetypes.ARCHETYPES coherent_regimes "
            f"for the valid pairing"
        )

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
        # An indicator that does `dataframe['x'] = ...` inline in its compute
        # AND also declares a `columns` block produces TWO assignments — the
        # second one is `dataframe['x'] = x` referencing a local var that was
        # never bound (the inline assign skipped the local). Caught
        # empirically: this was the NameError-class failure in trial #2.
        compute = ind["compute"]
        has_inline_assign = bool(_COL_ASSIGN_RE.search(compute))
        has_columns = bool(ind.get("columns"))
        if has_inline_assign and has_columns:
            raise SpecError(
                "indicator has BOTH an inline `dataframe['x'] = ...` "
                "assignment in 'compute' AND a 'columns' block. The renderer "
                "would emit a second `dataframe['x'] = x` referencing an "
                "undefined local variable. Use one form: either inline "
                "assignment with no columns, OR a local-var compute "
                f"(e.g. `bb = ta.bbands(...)`) with columns. Got: {ind!r}"
            )

    # Cross-check: every dataframe['x'] referenced in entry/exit conditions
    # must be available — declared by an indicator, OHLCV, or macro. Without
    # this, a spec that uses `dataframe['rsi']` in exit but forgets to
    # compute rsi crashes at backtest time with KeyError, wasting an LLM
    # turn (this was the KeyError-class failure across multiple trials).
    declared = _declared_columns(spec)
    for section, cond in _condition_strings(spec):
        for m in _COL_REF_RE.finditer(cond):
            col = m.group(1)
            if col not in declared:
                indicator_cols = sorted(declared - _OHLCV_COLUMNS - _MACRO_COLUMNS)
                raise SpecError(
                    f"{section} references dataframe[{col!r}] but no "
                    f"indicator declares this column. Declared indicators: "
                    f"{indicator_cols}. (OHLCV and macro columns are also "
                    f"available without declaration.) Offending condition: "
                    f"{cond!r}"
                )

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
    STRATEGY_ARCHETYPE = "{archetype}"
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
        # Macro context (fgi, vix, gold, dxy, spx, btc_funding_rate, btc_oi,
        # btc_oi_pct_change_24h, eth_btc_ratio, eth_btc_change_7d, alt_strength_zscore_30d)
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
        archetype=spec["archetype"],
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
