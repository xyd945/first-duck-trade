"""Tests for the indicator-form guidance added to SYSTEM_PROMPT_SPEC.

Trials #1-#5 showed the LLM repeatedly generating the broken pattern:
  {"compute": "dataframe['x'] = ta.foo(...)", "columns": [{"name": "x", ...}]}
even when the validator rejected it within the turn. The validator alone
couldn't help because the retry prompt only included the error message;
the LLM had no rule in the system prompt that would prevent the same
output again. These tests lock in the prompt guidance so a future edit
can't silently drop it.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


def _prompt():
    from strategy_generator import SYSTEM_PROMPT_SPEC
    return SYSTEM_PROMPT_SPEC


def test_prompt_documents_form_a_inline_assignment():
    """Form A: dataframe['x'] = ... in compute, NO columns block."""
    p = _prompt()
    assert "Form A" in p
    assert 'dataframe[\'rsi\'] = ta.rsi' in p


def test_prompt_documents_form_b_local_var_with_columns():
    """Form B: local var in compute, columns extracts from it."""
    p = _prompt()
    assert "Form B" in p
    assert 'bb = ta.bbands' in p
    assert '"name": "bb_lower"' in p
    assert "bb['BBL_20_2.0']" in p


def test_prompt_shows_the_forbidden_pattern_with_wrong_label():
    """The trap pattern must be shown explicitly as WRONG so the LLM
    can't pattern-match on partial similarity to a valid example."""
    p = _prompt()
    assert "WRONG" in p
    # The actual trap from trial #5 logs — inline + columns referencing the
    # name back into dataframe['x']
    assert 'dataframe[\'ema_20\'] = ta.ema' in p
    assert '"name": "ema_20"' in p


def test_prompt_promotes_indicator_form_rule_to_hard_rule():
    """The mixing pattern is now HARD RULE 8 — auto-rejected by validator."""
    p = _prompt()
    assert "8." in p
    # Find the rule body and confirm it mentions one-form-per-entry
    rule8_start = p.index("8. ")
    rule8_section = p[rule8_start:rule8_start + 400]
    assert "ONE" in rule8_section.upper() and "form" in rule8_section.lower()
    assert "auto-rejected" in rule8_section.lower() or "rejected" in rule8_section.lower()


def test_prompt_column_reference_rule_mentions_validator():
    """Rule 3 must surface the validator's behavior so the LLM knows
    why an exit referencing dataframe['rsi'] without an RSI indicator
    will get bounced — and what the error message will include."""
    p = _prompt()
    rule3_start = p.index("3. ")
    rule3_section = p[rule3_start:rule3_start + 400]
    # Should mention the cross-check or auto-rejection so the LLM knows
    # the validator is watching this specific class
    assert ("validator" in rule3_section.lower()
            or "cross-check" in rule3_section.lower()
            or "auto-reject" in rule3_section.lower())
