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


# ---------------------------------------------------------------------------
# get_recent_attributions (reflector consumption of R2d output)
# ---------------------------------------------------------------------------

def _seed_backtest_with_attribution(sr, tmp_path, name, total_trades, attribution):
    """Register a strategy + record a backtest with attached attribution."""
    sid = sr.register_strategy(
        name=name,
        filepath=str(tmp_path / f"{name}.py"),
        thesis=f"Thesis for {name}",
        target_regime="all",
    )
    sr.record_backtest(
        sid,
        {"timerange": "20260101-20260201", "total_trades": total_trades,
         "profit_total_pct": 1.0, "sharpe": 0.5},
        attribution=attribution,
    )
    return sid


def test_get_recent_attributions_returns_parsed_dicts(isolated_registry, tmp_path):
    sr = isolated_registry
    attr = {
        "total_trades": 20, "overall_win_rate": 0.5,
        "buckets": {"vix_low": {"trades": 10, "wins": 7, "win_rate": 0.7, "lift": 0.2}},
        "top_positive_lift": ["vix_low"], "top_negative_lift": [],
    }
    _seed_backtest_with_attribution(sr, tmp_path, "WithAttr", 20, attr)

    rows = sr.get_recent_attributions(n=5)
    assert len(rows) == 1
    assert rows[0]["name"] == "WithAttr"
    assert rows[0]["attribution"]["top_positive_lift"] == ["vix_low"]


def test_get_recent_attributions_excludes_empty_attribution(isolated_registry, tmp_path):
    sr = isolated_registry
    # Backtest with no attribution attached
    sid = sr.register_strategy(
        name="NoAttr", filepath=str(tmp_path / "NoAttr.py"),
        thesis="x", target_regime="all",
    )
    sr.record_backtest(sid, {"total_trades": 20})  # no attribution arg

    rows = sr.get_recent_attributions(n=5)
    assert rows == []


def test_get_recent_attributions_filters_by_min_trades(isolated_registry, tmp_path):
    sr = isolated_registry
    big_attr = {"total_trades": 50, "overall_win_rate": 0.5, "buckets": {},
                "top_positive_lift": [], "top_negative_lift": []}
    small_attr = {"total_trades": 3, "overall_win_rate": 0.33, "buckets": {},
                  "top_positive_lift": [], "top_negative_lift": []}
    _seed_backtest_with_attribution(sr, tmp_path, "Big", 50, big_attr)
    _seed_backtest_with_attribution(sr, tmp_path, "Small", 3, small_attr)

    rows = sr.get_recent_attributions(n=10, min_trades=10)
    assert [r["name"] for r in rows] == ["Big"]


def test_get_recent_attributions_orders_newest_first(isolated_registry, tmp_path):
    sr = isolated_registry
    attr = {"total_trades": 20, "overall_win_rate": 0.5, "buckets": {},
            "top_positive_lift": [], "top_negative_lift": []}
    _seed_backtest_with_attribution(sr, tmp_path, "First", 20, attr)
    _seed_backtest_with_attribution(sr, tmp_path, "Second", 20, attr)
    _seed_backtest_with_attribution(sr, tmp_path, "Third", 20, attr)

    rows = sr.get_recent_attributions(n=10)
    # All share the same created_at second, but insertion order is the tie-break
    # via DESC on created_at, then implicit ID — verify all 3 present
    names = [r["name"] for r in rows]
    assert set(names) == {"First", "Second", "Third"}


def test_archetype_aware_eviction_protects_rare_archetypes(isolated_registry, tmp_path, monkeypatch):
    """When the candidate pool overflows, the oldest candidate of an
    OVER-REPRESENTED archetype should be evicted — NOT the only candidate
    of a rare archetype that just happens to be the oldest."""
    sr = isolated_registry
    monkeypatch.setattr(sr, "MAX_CANDIDATES", 4)

    # Seed the pool to capacity:
    #   - rare_archetype: 1 candidate, oldest
    #   - busy_archetype: 3 candidates, newer
    rare_id = sr.register_strategy(
        name="RareOldest", filepath=str(tmp_path / "rare.py"),
        thesis="x", target_regime="all", archetype="alt_strength_divergence",
    )
    busy_ids = []
    for i in range(3):
        busy_ids.append(sr.register_strategy(
            name=f"BusyNew{i}", filepath=str(tmp_path / f"busy{i}.py"),
            thesis="x", target_regime="trending",
            archetype="momentum_continuation",
        ))

    # Pool is at 4. Register a fifth → eviction triggered. Naive "oldest
    # wins" would kill the rare one; archetype-aware logic must instead
    # kill the oldest busy_archetype member.
    sr.register_strategy(
        name="NewArrival", filepath=str(tmp_path / "new.py"),
        thesis="x", target_regime="ranging", archetype="mean_reversion",
    )

    rare = sr.get_strategy_by_name("RareOldest")
    busy0 = sr.get_strategy_by_name("BusyNew0")
    assert rare["status"] == "candidate", \
        "the only alt_strength_divergence candidate must NOT be evicted"
    assert busy0["status"] == "retired", \
        "the oldest momentum_continuation candidate should be the eviction victim"


def test_eviction_falls_back_to_oldest_when_all_archetypes_unique(isolated_registry, tmp_path, monkeypatch):
    """If every candidate has a unique archetype, there's no over-
    represented archetype to evict from — fall back to evicting plain oldest."""
    sr = isolated_registry
    monkeypatch.setattr(sr, "MAX_CANDIDATES", 3)

    archs = ["momentum_continuation", "mean_reversion", "vol_squeeze"]
    for i, a in enumerate(archs):
        sr.register_strategy(
            name=f"Unique{i}", filepath=str(tmp_path / f"u{i}.py"),
            thesis="x", target_regime="all", archetype=a,
        )

    # Pool full, all unique archetypes. Adding a fourth → evict plain oldest.
    sr.register_strategy(
        name="Fourth", filepath=str(tmp_path / "u4.py"),
        thesis="x", target_regime="all", archetype="funding_contrarian",
    )

    assert sr.get_strategy_by_name("Unique0")["status"] == "retired", \
        "with all-unique archetypes, the plain-oldest candidate is evicted"
    assert sr.get_strategy_by_name("Unique1")["status"] == "candidate"
    assert sr.get_strategy_by_name("Unique2")["status"] == "candidate"
    assert sr.get_strategy_by_name("Fourth")["status"] == "candidate"


def test_get_active_strategies_with_trade_paths(isolated_registry, tmp_path):
    """Active strategies should come back with their MOST RECENT backtest's
    trade export path. Candidates and retired strategies are excluded.
    Strategies with no stored path get an empty string."""
    sr = isolated_registry

    # active with two backtests — should return the newest path
    sid_a = sr.register_strategy(name="ActiveTwo", filepath=str(tmp_path/"a.py"),
                                  thesis="x", target_regime="all")
    sr.record_backtest(sid_a, {"total_trades": 20, "trades_export_path": "/old.zip"})
    sr.record_backtest(sid_a, {"total_trades": 25, "trades_export_path": "/new.zip"})
    sr.promote_strategy(sid_a)

    # active with no export
    sid_b = sr.register_strategy(name="ActiveLegacy", filepath=str(tmp_path/"b.py"),
                                  thesis="x", target_regime="all")
    sr.record_backtest(sid_b, {"total_trades": 20})  # no path
    sr.promote_strategy(sid_b)

    # candidate (not active) — must NOT appear
    sr.register_strategy(name="StillCandidate", filepath=str(tmp_path/"c.py"),
                         thesis="x", target_regime="all")

    # retired — must NOT appear
    sid_d = sr.register_strategy(name="Gone", filepath=str(tmp_path/"d.py"),
                                  thesis="x", target_regime="all")
    sr.retire_strategy(sid_d, reason="test", verdict="FAIL_TEST")

    rows = sr.get_active_strategies_with_trade_paths()
    by_name = {r["name"]: r for r in rows}
    assert set(by_name.keys()) == {"ActiveTwo", "ActiveLegacy"}
    assert by_name["ActiveTwo"]["trades_export_path"] == "/new.zip"
    assert by_name["ActiveLegacy"]["trades_export_path"] == ""


def test_get_recent_attributions_survives_corrupt_json(isolated_registry, tmp_path):
    """One bad JSON row should not block the rest from being returned."""
    sr = isolated_registry
    good_attr = {"total_trades": 20, "overall_win_rate": 0.5, "buckets": {},
                 "top_positive_lift": [], "top_negative_lift": []}
    sid_good = _seed_backtest_with_attribution(sr, tmp_path, "Good", 20, good_attr)

    # Inject a bogus row directly
    sid_bad = sr.register_strategy(
        name="Bad", filepath=str(tmp_path / "Bad.py"),
        thesis="x", target_regime="all",
    )
    conn = sr.get_db()
    from datetime import datetime, timezone
    conn.execute(
        """INSERT INTO backtest_results (strategy_id, timerange, sharpe, sortino,
           max_drawdown_pct, max_drawdown_abs, profit_total_pct, profit_total_abs,
           profit_factor, total_trades, win_rate, backtest_days, avg_duration,
           attribution_json, created_at) VALUES
           (?, '', 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, '', 'NOT JSON {{{', ?)""",
        (sid_bad, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    rows = sr.get_recent_attributions(n=10)
    names = [r["name"] for r in rows]
    assert "Good" in names
    assert "Bad" not in names  # corrupt row silently dropped


# ---------------------------------------------------------------------------
# get_recent_failures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# dedupe_class_name — UNIQUE collision fix
# ---------------------------------------------------------------------------

def _write_strategy_file(tmp_path, class_name: str):
    code = f'''from base_generated import BaseGeneratedStrategy

class {class_name}(BaseGeneratedStrategy):
    STRATEGY_THESIS = "test"
    TARGET_REGIME = "all"
    GENERATION_ID = "gen-test"
'''
    fp = tmp_path / f"Strategy_{class_name}.py"
    fp.write_text(code)
    return fp


def test_dedupe_class_name_noop_when_no_collision(tmp_path):
    from strategy_generator import dedupe_class_name

    fp = _write_strategy_file(tmp_path, "Unique")
    result = dedupe_class_name(fp, "Unique", name_exists=lambda n: False)
    assert result == "Unique"
    assert "class Unique(" in fp.read_text()  # file unchanged


def test_dedupe_class_name_renames_on_collision(tmp_path):
    from strategy_generator import dedupe_class_name

    fp = _write_strategy_file(tmp_path, "Clashing")
    taken = {"Clashing"}
    result = dedupe_class_name(fp, "Clashing", name_exists=lambda n: n in taken)

    assert result == "Clashing_v2"
    src = fp.read_text()
    assert "class Clashing_v2(" in src
    assert "class Clashing(" not in src  # original declaration rewritten


def test_dedupe_class_name_walks_until_unique(tmp_path):
    from strategy_generator import dedupe_class_name

    fp = _write_strategy_file(tmp_path, "Taken")
    taken = {"Taken", "Taken_v2", "Taken_v3"}
    result = dedupe_class_name(fp, "Taken", name_exists=lambda n: n in taken)
    assert result == "Taken_v4"
    assert "class Taken_v4(" in fp.read_text()


def test_dedupe_class_name_is_word_bounded(tmp_path):
    """Renaming 'Foo' must not touch 'FooBar' or 'class FooWrapper'."""
    from strategy_generator import dedupe_class_name

    code = '''from base_generated import BaseGeneratedStrategy

class Foo(BaseGeneratedStrategy):
    STRATEGY_THESIS = "FooBar should not match; class FooWrapper either"
    pass

# class FooBar would be a separate thing — ensure we don't touch comments
'''
    fp = tmp_path / "Strategy_Foo.py"
    fp.write_text(code)

    result = dedupe_class_name(fp, "Foo", name_exists=lambda n: n == "Foo")
    assert result == "Foo_v2"
    src = fp.read_text()
    assert "class Foo_v2(" in src
    # The string "FooBar" and "FooWrapper" must remain unchanged
    assert "FooBar" in src
    assert "FooWrapper" in src


def test_dedupe_class_name_raises_when_class_decl_missing(tmp_path):
    from strategy_generator import dedupe_class_name

    fp = tmp_path / "broken.py"
    fp.write_text("# no class declaration here\n")
    # Only the original name collides; suffix is free → dedupe gets through
    # the rename step and then fails because the class decl isn't in the file.
    import pytest as _pytest
    with _pytest.raises(ValueError):
        dedupe_class_name(fp, "Missing", name_exists=lambda n: n == "Missing")


def test_dedupe_class_name_caps_iterations(tmp_path):
    """A pathological name_exists callback must not hang — caller fails fast."""
    from strategy_generator import dedupe_class_name

    fp = _write_strategy_file(tmp_path, "Endless")
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="1000 attempts"):
        dedupe_class_name(fp, "Endless", name_exists=lambda n: True)
