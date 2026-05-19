"""Tests for Phase 6: archetypes module + coherence matrix."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))

from archetypes import (
    ARCHETYPES,
    coherence_matrix,
    archetype_names,
    prompt_blurb_for,
    thesis_for,
    coherent_regimes_for,
    is_coherent,
    VALID_REGIMES,
)


# ---------------------------------------------------------------------------
# Catalog shape
# ---------------------------------------------------------------------------

def test_has_exactly_10_archetypes():
    """Phase 6 promises 10. Lock it down so additions are an explicit choice
    (each new archetype needs a coherence-matrix decision)."""
    assert len(ARCHETYPES) == 10


def test_every_archetype_has_required_fields():
    for name, spec in ARCHETYPES.items():
        assert "thesis" in spec, f"{name} missing thesis"
        assert "blurb" in spec, f"{name} missing blurb"
        assert "coherent_regimes" in spec, f"{name} missing coherent_regimes"
        assert spec["thesis"], f"{name} thesis is empty"
        assert spec["blurb"], f"{name} blurb is empty"
        assert spec["coherent_regimes"], f"{name} coherent_regimes is empty"


def test_every_archetype_uses_valid_regime_values():
    """coherent_regimes can only contain regime values the spec validator
    accepts. Catches typos like 'trendng'."""
    for name, spec in ARCHETYPES.items():
        for regime in spec["coherent_regimes"]:
            assert regime in VALID_REGIMES, f"{name} declares invalid regime {regime!r}"


def test_blurbs_mention_key_concepts():
    """Surface-level sanity: each archetype's blurb should mention at least
    one of its expected indicators. Defends against accidental blurb swap
    during refactors."""
    expected_mentions = {
        "momentum_continuation": ["adx", "ema"],
        "mean_reversion": ["rsi", "bollinger"],
        "breakout_volume": ["volume", "donchian"],
        "vol_squeeze": ["bollinger", "atr"],
        "vol_compression_mean_reversion": ["atr", "%b"],
        "funding_contrarian": ["btc_funding_rate", "funding"],
        "oi_cascade_followthrough": ["btc_oi", "de-lev"],
        "alt_strength_divergence": ["alt_strength_zscore_30d", "eth/btc"],
        "macro_led_risk_on": ["vix", "dxy"],
        "liquidity_sweep_followthrough": ["sweep", "reclaim"],
    }
    for archetype, terms in expected_mentions.items():
        blurb_lower = ARCHETYPES[archetype]["blurb"].lower()
        for term in terms:
            assert term.lower() in blurb_lower, (
                f"{archetype} blurb does not mention {term!r}"
            )


# ---------------------------------------------------------------------------
# Coherence matrix
# ---------------------------------------------------------------------------

def test_coherence_matrix_has_20_cells():
    """Locked count — moving cells in or out should be deliberate and the
    test updated accordingly so a casual edit doesn't silently change
    weekly compute load."""
    assert len(coherence_matrix()) == 20


def test_coherence_matrix_cells_are_unique():
    cells = coherence_matrix()
    assert len(cells) == len(set(cells))


def test_coherence_matrix_excludes_obvious_category_errors():
    """Explicit assertion: these incoherent pairs must NOT be in the matrix."""
    cells = set(coherence_matrix())
    # Mean reversion in trending = catching falling knives
    assert ("mean_reversion", "trending") not in cells
    # Momentum continuation in ranging = whipsaw hell
    assert ("momentum_continuation", "ranging") not in cells
    # Funding contrarian in ranging = trigger never fires
    assert ("funding_contrarian", "ranging") not in cells
    # OI cascade in ranging = no leveraged crowd to flush
    assert ("oi_cascade_followthrough", "ranging") not in cells
    # Breakout volume in trending = you're already in the move
    assert ("breakout_volume", "trending") not in cells


def test_coherence_matrix_includes_canonical_pairs():
    """Spot-check the obvious-fit pairs ARE present."""
    cells = set(coherence_matrix())
    assert ("momentum_continuation", "trending") in cells
    assert ("mean_reversion", "ranging") in cells
    assert ("breakout_volume", "breakout") in cells
    assert ("vol_squeeze", "breakout") in cells
    # The position-driven archetypes get an 'all' slot
    assert ("funding_contrarian", "all") in cells
    assert ("alt_strength_divergence", "all") in cells
    assert ("macro_led_risk_on", "all") in cells


def test_only_position_driven_archetypes_get_all_regime():
    """The 'all' regime is reserved for archetypes whose thesis is genuinely
    regime-independent. Trend-following claiming 'all' is overpromising."""
    cells = coherence_matrix()
    all_regime_archetypes = {a for a, r in cells if r == "all"}
    expected = {
        "funding_contrarian",        # positioning, regime-agnostic
        "oi_cascade_followthrough",  # liquidation cascades, regime-agnostic
        "alt_strength_divergence",   # cross-asset, regime-agnostic
        "macro_led_risk_on",         # macro thesis, regime-agnostic frame
    }
    assert all_regime_archetypes == expected


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_archetype_names_returns_definition_order():
    names = archetype_names()
    assert names == list(ARCHETYPES.keys())
    assert len(names) == 10


def test_prompt_blurb_returns_per_archetype_content():
    blurb = prompt_blurb_for("momentum_continuation")
    assert "trend" in blurb.lower()
    # Different archetype should give different blurb
    other = prompt_blurb_for("mean_reversion")
    assert blurb != other


def test_prompt_blurb_raises_on_unknown_archetype():
    """No silent fallback — caller must validate before requesting."""
    with pytest.raises(KeyError):
        prompt_blurb_for("invented_archetype")


def test_thesis_returns_one_line_per_archetype():
    for name in archetype_names():
        thesis = thesis_for(name)
        assert thesis  # non-empty
        # One-sentence: shouldn't have multiple paragraphs
        assert thesis.count("\n") == 0


def test_coherent_regimes_returns_list_copy():
    """Caller shouldn't be able to mutate the catalog through the helper."""
    regimes = coherent_regimes_for("momentum_continuation")
    regimes.append("garbage")
    # Catalog must still be clean
    assert "garbage" not in ARCHETYPES["momentum_continuation"]["coherent_regimes"]


# ---------------------------------------------------------------------------
# is_coherent
# ---------------------------------------------------------------------------

def test_is_coherent_true_for_canonical_pairs():
    assert is_coherent("momentum_continuation", "trending") is True
    assert is_coherent("mean_reversion", "ranging") is True
    assert is_coherent("funding_contrarian", "all") is True


def test_is_coherent_false_for_incoherent_pairs():
    assert is_coherent("mean_reversion", "trending") is False
    assert is_coherent("momentum_continuation", "ranging") is False


def test_is_coherent_false_for_unknown_archetype():
    """Defensive — caller might pass a typo."""
    assert is_coherent("not_a_real_archetype", "trending") is False
