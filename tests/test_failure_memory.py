"""Tests for the failure-memory feedback loop (Round 1).

Covers:
  - Registry migration adds failure_reason + failure_verdict idempotently
  - retire_strategy persists both fields
  - get_recent_failures filters correctly (by regime, excludes empty verdicts)
  - load_recent_reflections returns newest-first, bounded
  - Generator prompt contains/omits the two new sections cleanly
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "user_data" / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the registry module at a fresh tmp DB + reflections dir."""
    import strategy_registry as sr

    db = tmp_path / "test_registry.db"
    refl = tmp_path / "reflections"
    monkeypatch.setattr(sr, "DB_PATH", db)
    monkeypatch.setattr(sr, "REFLECTIONS_DIR", refl)
    sr.init_db()
    return sr


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_init_db_adds_failure_columns(isolated_registry):
    sr = isolated_registry
    conn = sr.get_db()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(strategies)").fetchall()}
    conn.close()
    assert "failure_reason" in cols
    assert "failure_verdict" in cols


def test_init_db_is_idempotent(isolated_registry):
    sr = isolated_registry
    # Calling again should not error — ALTER TABLE is gated on column presence
    sr.init_db()
    sr.init_db()


def test_migration_from_legacy_schema(tmp_path, monkeypatch):
    """Simulate an older DB that lacks the new columns."""
    import sqlite3
    import strategy_registry as sr

    db = tmp_path / "legacy.db"
    monkeypatch.setattr(sr, "DB_PATH", db)
    monkeypatch.setattr(sr, "REFLECTIONS_DIR", tmp_path / "reflections")

    # Create legacy schema without failure_* columns
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            filepath TEXT NOT NULL,
            thesis TEXT DEFAULT '',
            target_regime TEXT DEFAULT 'all',
            generation_id TEXT DEFAULT '',
            status TEXT DEFAULT 'candidate',
            created_at TEXT NOT NULL,
            promoted_at TEXT,
            retired_at TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Run migration
    sr.init_db()

    conn = sr.get_db()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(strategies)").fetchall()}
    conn.close()
    assert "failure_reason" in cols
    assert "failure_verdict" in cols


# ---------------------------------------------------------------------------
# retire_strategy persistence
# ---------------------------------------------------------------------------

def test_retire_strategy_persists_reason_and_verdict(isolated_registry, tmp_path):
    sr = isolated_registry
    sid = sr.register_strategy(
        name="TestFoo",
        filepath=str(tmp_path / "TestFoo.py"),
        thesis="RSI mean-reversion on BTC",
        target_regime="ranging",
    )
    sr.retire_strategy(sid, reason="Too few trades: 2", verdict="FAIL_TOO_FEW")

    conn = sr.get_db()
    row = conn.execute(
        "SELECT status, failure_reason, failure_verdict FROM strategies WHERE id = ?",
        (sid,),
    ).fetchone()
    conn.close()

    assert row["status"] == "retired"
    assert row["failure_reason"] == "Too few trades: 2"
    assert row["failure_verdict"] == "FAIL_TOO_FEW"


# ---------------------------------------------------------------------------
# get_recent_failures
# ---------------------------------------------------------------------------

def _seed_retired(sr, tmp_path, name, regime, verdict, reason):
    sid = sr.register_strategy(
        name=name,
        filepath=str(tmp_path / f"{name}.py"),
        thesis=f"Thesis for {name}",
        target_regime=regime,
    )
    sr.retire_strategy(sid, reason=reason, verdict=verdict)
    return sid


def test_get_recent_failures_excludes_empty_verdict(isolated_registry, tmp_path):
    """Auto-retires from pool overflow (empty verdict) must NOT appear."""
    sr = isolated_registry
    _seed_retired(sr, tmp_path, "WithVerdict", "all", "FAIL_BACKTEST", "crashed")

    # Manually retire one with no verdict (simulates pool overflow)
    sid = sr.register_strategy(
        name="NoVerdict", filepath=str(tmp_path / "NoVerdict.py"), target_regime="all",
    )
    conn = sr.get_db()
    conn.execute(
        "UPDATE strategies SET status='retired', retired_at='2026-01-01' WHERE id=?", (sid,),
    )
    conn.commit()
    conn.close()

    failures = sr.get_recent_failures(k=10)
    names = [f["name"] for f in failures]
    assert "WithVerdict" in names
    assert "NoVerdict" not in names


def test_get_recent_failures_filters_by_regime(isolated_registry, tmp_path):
    sr = isolated_registry
    _seed_retired(sr, tmp_path, "Trender", "trending", "FAIL_UNPROFITABLE", "loss")
    _seed_retired(sr, tmp_path, "Ranger", "ranging", "FAIL_UNPROFITABLE", "loss")
    _seed_retired(sr, tmp_path, "AllRegime", "all", "FAIL_UNPROFITABLE", "loss")

    trending = {f["name"] for f in sr.get_recent_failures(k=10, regime="trending")}
    assert "Trender" in trending
    assert "AllRegime" in trending  # 'all' always matches
    assert "Ranger" not in trending

    everything = {f["name"] for f in sr.get_recent_failures(k=10)}
    assert everything == {"Trender", "Ranger", "AllRegime"}


def test_get_recent_failures_limits_k(isolated_registry, tmp_path):
    sr = isolated_registry
    for i in range(5):
        _seed_retired(sr, tmp_path, f"S{i}", "all", "FAIL_TOO_FEW", "0 trades")
    assert len(sr.get_recent_failures(k=3)) == 3


# ---------------------------------------------------------------------------
# load_recent_reflections
# ---------------------------------------------------------------------------

def test_load_recent_reflections_empty_when_dir_missing(isolated_registry):
    sr = isolated_registry
    assert sr.load_recent_reflections() == ""


def test_load_recent_reflections_newest_first(isolated_registry):
    sr = isolated_registry
    sr.REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    import os
    import time
    old = sr.REFLECTIONS_DIR / "reflection-20260101-000000.md"
    new = sr.REFLECTIONS_DIR / "reflection-20260201-000000.md"
    old.write_text("OLD_REFLECTION_CONTENT")
    time.sleep(0.01)
    new.write_text("NEW_REFLECTION_CONTENT")
    # Force mtimes so the test is deterministic
    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_800_000_000, 1_800_000_000))

    text = sr.load_recent_reflections(n=2)
    # Newest must appear before oldest
    assert text.index("NEW_REFLECTION_CONTENT") < text.index("OLD_REFLECTION_CONTENT")


def test_load_recent_reflections_truncates(isolated_registry):
    sr = isolated_registry
    sr.REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (sr.REFLECTIONS_DIR / "reflection-20260101-000000.md").write_text("X" * 10_000)
    text = sr.load_recent_reflections(n=1, max_chars=500)
    assert len(text) <= 500 + len("\n\n[...truncated]")
    assert "[...truncated]" in text


# ---------------------------------------------------------------------------
# Generator prompt assembly
# ---------------------------------------------------------------------------

def test_prompt_includes_sections_when_given():
    from strategy_generator import build_generation_prompt

    prompt = build_generation_prompt(
        target_regime="trending",
        reflector_insights="Lesson: entry filters too tight.",
        failure_examples="#1 [FAIL_UNPROFITABLE] thesis: X\n   why: Y",
    )
    assert "LESSONS FROM RECENT REFLECTIONS" in prompt
    assert "entry filters too tight" in prompt
    assert "RECENT FAILURES TO AVOID" in prompt
    assert "FAIL_UNPROFITABLE" in prompt


def test_prompt_omits_sections_when_empty():
    from strategy_generator import build_generation_prompt

    prompt = build_generation_prompt(target_regime="trending")
    assert "LESSONS FROM RECENT REFLECTIONS" not in prompt
    assert "RECENT FAILURES TO AVOID" not in prompt
    # Core prompt still renders
    assert "TARGET REGIME: trending" in prompt


def test_format_failure_examples_renders_thesis_and_reason():
    from strategy_generator import _format_failure_examples

    rows = [{
        "name": "TestStrat",
        "thesis": "Bollinger squeeze + RSI oversold",
        "target_regime": "ranging",
        "failure_reason": "Too few trades: 2",
        "failure_verdict": "FAIL_TOO_FEW",
        "total_trades": 2,
        "profit_total_pct": -0.5,
        "sharpe": -0.1,
        "code_excerpt": "def populate_entry_trend(self, dataframe, metadata):\n    dataframe['enter_long'] = 0",
    }]
    block = _format_failure_examples(rows)
    assert "FAIL_TOO_FEW" in block
    assert "Bollinger squeeze" in block
    assert "Too few trades: 2" in block
    assert "trades=2" in block
    assert "populate_entry_trend" in block


def test_format_failure_examples_empty():
    from strategy_generator import _format_failure_examples
    assert _format_failure_examples([]) == ""
