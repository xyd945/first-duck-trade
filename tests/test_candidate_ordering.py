"""Tests for get_candidates() FIFO ordering — the candidate-fairness fix.

Before this fix, get_candidates() ordered by ``created_at DESC`` (newest
first). The orchestrator's job_backtest_candidates loop slices the result
at [:25]. With newest-first ordering, any candidate that lands at
position 26+ on Sunday would never be evaluated — and as the next week's
generation pushed it further back, it would stay starved until
archetype-eviction retired it without a single full backtest.

These tests pin the FIFO ordering so a future edit can't quietly
reintroduce starvation.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the registry at a fresh on-disk SQLite for each test."""
    import strategy_registry
    db_path = tmp_path / "test_registry.db"
    monkeypatch.setattr(strategy_registry, "DB_PATH", db_path)
    strategy_registry.init_db()
    return strategy_registry


def _register(reg, name: str, created_at: datetime, archetype: str = "mean_reversion"):
    """Register a candidate with an explicit created_at, bypassing the
    'now()' default so tests can control the ordering deterministically."""
    import sqlite3
    conn = sqlite3.connect(reg.DB_PATH)
    conn.execute(
        "INSERT INTO strategies (name, filepath, thesis, target_regime, "
        "generation_id, archetype, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?)",
        (name, f"/tmp/{name}.py", "test", "ranging", "gen-test", archetype,
         created_at.isoformat()),
    )
    conn.commit()
    conn.close()


def test_get_candidates_returns_oldest_first(isolated_registry):
    reg = isolated_registry
    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    _register(reg, "Newest",  base + timedelta(days=2))
    _register(reg, "Middle",  base + timedelta(days=1))
    _register(reg, "Oldest",  base)

    names = [c["name"] for c in reg.get_candidates()]
    assert names == ["Oldest", "Middle", "Newest"], (
        "FIFO: oldest candidates must be scored first to prevent starvation"
    )


def test_get_candidates_uses_id_tiebreaker_when_created_at_collides(isolated_registry):
    """Batch generation registers many candidates within a single second.
    The id tiebreaker must yield deterministic ordering."""
    reg = isolated_registry
    same_second = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    _register(reg, "FirstWritten",  same_second)
    _register(reg, "SecondWritten", same_second)
    _register(reg, "ThirdWritten",  same_second)

    names = [c["name"] for c in reg.get_candidates()]
    assert names == ["FirstWritten", "SecondWritten", "ThirdWritten"], (
        "Within the same created_at second, registration order (id ASC) wins"
    )


def test_get_candidates_excludes_active_and_retired(isolated_registry):
    """Only status='candidate' rows count toward the pool the backtest job
    iterates. Sanity check we didn't change that filter."""
    import sqlite3
    reg = isolated_registry
    _register(reg, "ActiveOne", datetime(2026, 5, 1, tzinfo=timezone.utc))
    _register(reg, "RetiredOne", datetime(2026, 5, 2, tzinfo=timezone.utc))
    _register(reg, "CandidateOne", datetime(2026, 5, 3, tzinfo=timezone.utc))

    conn = sqlite3.connect(reg.DB_PATH)
    conn.execute("UPDATE strategies SET status='active'  WHERE name='ActiveOne'")
    conn.execute("UPDATE strategies SET status='retired' WHERE name='RetiredOne'")
    conn.commit()
    conn.close()

    names = [c["name"] for c in reg.get_candidates()]
    assert names == ["CandidateOne"]


def test_starvation_scenario_resolves_within_two_runs(isolated_registry):
    """Concrete failure case: 30 candidates registered across two weeks,
    backtest cap of 25 per run. With newest-first, the 5 oldest would
    starve. With oldest-first FIFO, all 30 get processed in 2 runs."""
    reg = isolated_registry
    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    for i in range(30):
        _register(reg, f"Cand_{i:02d}", base + timedelta(minutes=i))

    candidates = reg.get_candidates()
    assert len(candidates) == 30

    # Run 1: process the first 25 (oldest)
    run1_names = [c["name"] for c in candidates[:25]]
    assert run1_names[0] == "Cand_00"
    assert run1_names[-1] == "Cand_24"

    # Simulate run 1: those 25 leave the candidate pool (promoted/retired)
    import sqlite3
    conn = sqlite3.connect(reg.DB_PATH)
    for name in run1_names:
        conn.execute("UPDATE strategies SET status='retired' WHERE name=?", (name,))
    conn.commit()
    conn.close()

    # Run 2: remaining 5 must surface — these would have been the SKIPPED
    # ones under the old newest-first ordering
    run2 = reg.get_candidates()
    assert [c["name"] for c in run2] == [f"Cand_{i:02d}" for i in range(25, 30)]


def test_get_candidates_ordering_matches_orchestrator_loop_assumption():
    """Cheap source-level check: confirm orchestrator's [:25] slice still
    expects to consume from the head of get_candidates(). If someone ever
    flips this to [-25:] or shuffles, both this test and the registry
    docstring need to update together."""
    src = (ROOT / "user_data" / "scripts" / "orchestrator.py").read_text()
    assert "candidates = get_candidates()" in src
    assert "for cand in candidates[:25]:" in src, (
        "Orchestrator's backtest loop expects to slice from the head; if "
        "you change this slice, revisit get_candidates() ordering too"
    )
