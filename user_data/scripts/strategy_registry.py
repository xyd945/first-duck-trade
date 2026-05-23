"""
Strategy Registry — SQLite-based strategy lifecycle management.

Tracks all strategies through their lifecycle:
  candidate -> active -> retired

The orchestrator is the single writer. All other components read only.

Schema:
  strategies: id, name, filepath, thesis, target_regime, generation_id,
              status, created_at, promoted_at, retired_at,
              failure_reason, failure_verdict
  backtest_results: id, strategy_id, timerange, sharpe, max_drawdown_pct,
                    profit_total_pct, profit_factor, total_trades, win_rate,
                    backtest_days, created_at
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("strategy_registry")

BASE_DIR = Path(__file__).resolve().parent.parent  # user_data/
DB_PATH = BASE_DIR / "data" / "strategy_registry.db"
REFLECTIONS_DIR = BASE_DIR / "data" / "reflections"

# Pool limits
MAX_ACTIVE = 10
# Bumped 30 → 100 for Phase 6. Weekly generation produces 20 strategies (one per
# coherence-matrix cell). Backtest cycle evaluates up to 10/run. In steady state
# the pool sits around 30-40 candidates; 100 gives ~3 weeks of buffer even if
# backtests timeout or get delayed. Higher than strictly necessary, but cheap —
# the cost is a single int column in SQLite — and it protects diversity by
# making auto-evictions rare.
MAX_CANDIDATES = 100


def get_db() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _column_names(conn: sqlite3.Connection, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in rows}


def _migrate_failure_columns(conn: sqlite3.Connection):
    """Add failure_reason + failure_verdict columns to existing DBs."""
    cols = _column_names(conn, "strategies")
    if "failure_reason" not in cols:
        conn.execute("ALTER TABLE strategies ADD COLUMN failure_reason TEXT DEFAULT ''")
        log.info("Migrated strategies: added failure_reason")
    if "failure_verdict" not in cols:
        conn.execute("ALTER TABLE strategies ADD COLUMN failure_verdict TEXT DEFAULT ''")
        log.info("Migrated strategies: added failure_verdict")


def _migrate_attribution_column(conn: sqlite3.Connection):
    """R2d: add attribution_json column to backtest_results for per-trade
    macro-bucket attribution. Stored as JSON text for forward-compatibility."""
    cols = _column_names(conn, "backtest_results")
    if "attribution_json" not in cols:
        conn.execute(
            "ALTER TABLE backtest_results ADD COLUMN attribution_json TEXT DEFAULT ''"
        )
        log.info("Migrated backtest_results: added attribution_json")


def _migrate_trades_export_path_column(conn: sqlite3.Connection):
    """R7.4: add trades_export_path so the correlation gate can locate each
    promoted strategy's most-recent trade list (it's the raw input it needs
    to compute daily-return correlation between strategies)."""
    cols = _column_names(conn, "backtest_results")
    if "trades_export_path" not in cols:
        conn.execute(
            "ALTER TABLE backtest_results ADD COLUMN trades_export_path TEXT DEFAULT ''"
        )
        log.info("Migrated backtest_results: added trades_export_path")


def _migrate_archetype_column(conn: sqlite3.Connection):
    """Phase 6: add archetype column to strategies so failure memory and
    attribution can be queried/aggregated per-archetype (e.g. "retire the
    vol_squeeze archetype — 0% promotion rate after 8 weeks")."""
    cols = _column_names(conn, "strategies")
    if "archetype" not in cols:
        conn.execute(
            "ALTER TABLE strategies ADD COLUMN archetype TEXT DEFAULT ''"
        )
        log.info("Migrated strategies: added archetype")


def _migrate_deployment_lifecycle_columns(conn: sqlite3.Connection):
    """Deployment lifecycle Phase 1: split research_status from
    deployment_status. The old `status` column conflated "passed
    research gates" with "actually trading on OKX". The new columns
    keep them orthogonal so the registry can be honest about what's
    really running.

    Existing `status='active'` rows backfill to
    `research_status='approved', deployment_status='not_deployed'`.
    They become candidates for the reconciler (Phase 2+) to deploy.
    The old `status` column is preserved as a compatibility shim
    through Phase 4 cutover and dropped in a follow-up PR.

    See docs/deployment-lifecycle.md for the full lifecycle spec.
    """
    cols = _column_names(conn, "strategies")
    new_cols = (
        ("research_status",          "TEXT DEFAULT ''"),
        ("deployment_status",        "TEXT DEFAULT 'not_deployed'"),
        ("deployed_at",              "TEXT"),
        ("last_deployment_error",    "TEXT DEFAULT ''"),
        ("deployment_blocked_until", "TEXT"),
    )
    for name, decl in new_cols:
        if name not in cols:
            conn.execute(f"ALTER TABLE strategies ADD COLUMN {name} {decl}")
            log.info(f"Migrated strategies: added {name}")

    # Backfill research_status from the legacy status column. Only fills
    # rows where research_status is unset — rerunning the migration is a
    # no-op.
    conn.execute("""
        UPDATE strategies
        SET research_status = CASE status
            WHEN 'active'    THEN 'approved'
            WHEN 'retired'   THEN 'retired'
            WHEN 'candidate' THEN 'candidate'
            ELSE 'candidate'
        END
        WHERE research_status = '' OR research_status IS NULL
    """)

    # Indexes for the reconciler's hot path (Phase 2): "find approved
    # strategies eligible to deploy" and "find currently deployed
    # strategies to compare against running containers".
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategies_research_status "
        "ON strategies(research_status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategies_deployment_status "
        "ON strategies(deployment_status)"
    )


def _migrate_backtest_data_end_at(conn: sqlite3.Connection):
    """Deployment lifecycle Phase 1: distinguish "when the backtest
    was invoked" (existing `created_at`) from "the END of the data
    the backtest ran on". Without this, an eligibility filter on
    "recent backtest" can be satisfied by re-running yesterday on
    candles that end three months ago. The deployment-eligibility
    filter (Phase 2+) checks both.
    """
    cols = _column_names(conn, "backtest_results")
    if "backtest_data_end_at" not in cols:
        conn.execute(
            "ALTER TABLE backtest_results ADD COLUMN backtest_data_end_at TEXT DEFAULT ''"
        )
        log.info("Migrated backtest_results: added backtest_data_end_at")


def _backfill_backtest_data_end_at(conn: sqlite3.Connection):
    """Populate backtest_data_end_at for rows that pre-date the column.

    Existing strategies promoted before this PR have empty
    backtest_data_end_at, which fails the deployment-eligibility
    freshness filter — they look "stale data" even though most were
    actually backtested against current data days ago.

    Strategy: parse the existing ``timerange`` field (format
    "YYYYMMDD-YYYYMMDD") and use its end as backtest_data_end_at. When
    timerange is empty or malformed (rare — only true for very old
    rows backtested without an explicit --timerange), fall back to
    ``created_at`` (the time the backtest was invoked is a reasonable
    proxy when no better signal exists). Idempotent: rows already
    populated by the regular write path are not touched.
    """
    # Only do work if column exists (regular migration ran) AND there
    # are rows to backfill.
    cols = _column_names(conn, "backtest_results")
    if "backtest_data_end_at" not in cols:
        return  # _migrate_backtest_data_end_at hasn't run yet — should never happen

    rows = conn.execute(
        "SELECT id, timerange, created_at FROM backtest_results "
        "WHERE backtest_data_end_at IS NULL OR backtest_data_end_at = ''"
    ).fetchall()
    if not rows:
        return

    n_from_timerange = 0
    n_from_created = 0
    for r in rows:
        row_id = r["id"] if isinstance(r, sqlite3.Row) else r[0]
        tr = (r["timerange"] if isinstance(r, sqlite3.Row) else r[1]) or ""
        created = (r["created_at"] if isinstance(r, sqlite3.Row) else r[2]) or ""

        # Try timerange first: "YYYYMMDD-YYYYMMDD" or similar.
        # Take everything after the LAST hyphen so e.g. "20251101-20260101"
        # gives "20260101" and we then format as ISO date.
        end_iso = ""
        if "-" in tr:
            end_part = tr.rsplit("-", 1)[1].strip()
            if len(end_part) == 8 and end_part.isdigit():
                end_iso = f"{end_part[:4]}-{end_part[4:6]}-{end_part[6:8]} 00:00:00"
                n_from_timerange += 1

        if not end_iso and created:
            end_iso = created  # fallback
            n_from_created += 1

        if end_iso:
            conn.execute(
                "UPDATE backtest_results SET backtest_data_end_at = ? WHERE id = ?",
                (end_iso, row_id),
            )

    log.info(
        f"Backfilled backtest_data_end_at for {len(rows)} rows "
        f"({n_from_timerange} from timerange, {n_from_created} from created_at)"
    )


def init_db():
    """Create tables if they don't exist, then run migrations."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            filepath TEXT NOT NULL,
            thesis TEXT DEFAULT '',
            target_regime TEXT DEFAULT 'all',
            generation_id TEXT DEFAULT '',
            status TEXT DEFAULT 'candidate'
                CHECK(status IN ('candidate', 'active', 'retired')),
            created_at TEXT NOT NULL,
            promoted_at TEXT,
            retired_at TEXT,
            failure_reason TEXT DEFAULT '',
            failure_verdict TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL REFERENCES strategies(id),
            timerange TEXT,
            sharpe REAL DEFAULT 0,
            sortino REAL DEFAULT 0,
            max_drawdown_pct REAL DEFAULT 0,
            max_drawdown_abs REAL DEFAULT 0,
            profit_total_pct REAL DEFAULT 0,
            profit_total_abs REAL DEFAULT 0,
            profit_factor REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            backtest_days INTEGER DEFAULT 0,
            avg_duration TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies(status);
        CREATE INDEX IF NOT EXISTS idx_strategies_regime ON strategies(target_regime);
        CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_id);
    """)
    _migrate_failure_columns(conn)
    _migrate_attribution_column(conn)
    _migrate_trades_export_path_column(conn)
    _migrate_archetype_column(conn)
    _migrate_deployment_lifecycle_columns(conn)
    _migrate_backtest_data_end_at(conn)
    _backfill_backtest_data_end_at(conn)
    conn.commit()
    conn.close()
    log.info(f"Registry initialized at {DB_PATH}")


# ---------------------------------------------------------------------------
# Strategy CRUD
# ---------------------------------------------------------------------------

def register_strategy(
    name: str,
    filepath: str,
    thesis: str = "",
    target_regime: str = "all",
    generation_id: str = "",
    archetype: str = "",
) -> int:
    """Register a new candidate strategy. Returns strategy ID.

    `archetype` (Phase 6) is the enum value from archetypes.py — used by
    failure memory and attribution queries to filter/aggregate per-archetype.
    Empty string for legacy strategies registered before Phase 6.
    """
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Check candidate pool limit
    count = conn.execute(
        "SELECT COUNT(*) FROM strategies WHERE status = 'candidate'"
    ).fetchone()[0]

    if count >= MAX_CANDIDATES:
        # Phase 6: archetype-aware eviction. Naive "evict oldest" risks
        # killing the only candidate of a rare archetype (e.g. an
        # `oi_cascade_followthrough` that's been waiting for backtest
        # while five `momentum_continuation` candidates queue up). Prefer
        # to evict the oldest candidate of an archetype that's already
        # over-represented in the queue. Fall back to plain oldest if
        # no archetype has duplicates (all unique → no diversity to
        # protect, just trim the head).
        victim = conn.execute("""
            SELECT id, name, archetype FROM strategies
            WHERE status = 'candidate'
              AND archetype IN (
                SELECT archetype FROM strategies
                WHERE status = 'candidate' AND archetype != ''
                GROUP BY archetype HAVING COUNT(*) >= 2
              )
            ORDER BY created_at ASC LIMIT 1
        """).fetchone()
        if victim is None:
            # No archetype has 2+ candidates — fall back to plain oldest.
            victim = conn.execute(
                "SELECT id, name, archetype FROM strategies WHERE status = 'candidate' "
                "ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
        if victim:
            conn.execute(
                "UPDATE strategies SET status = 'retired', retired_at = ? WHERE id = ?",
                (now, victim["id"]),
            )
            log.info(
                f"Auto-retired oldest candidate: {victim['name']} "
                f"(archetype={victim['archetype'] or 'legacy'})"
            )

    cursor = conn.execute(
        """INSERT INTO strategies (name, filepath, thesis, target_regime, generation_id,
           archetype, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?)""",
        (name, str(filepath), thesis, target_regime, generation_id, archetype, now),
    )
    conn.commit()
    strategy_id = cursor.lastrowid
    conn.close()
    log.info(f"Registered strategy: {name} (id={strategy_id}, "
             f"regime={target_regime}, archetype={archetype or 'legacy'})")
    return strategy_id


def record_backtest(strategy_id: int, results: dict, attribution: dict | None = None):
    """Record backtest results for a strategy.

    `attribution` is the R2d per-trade macro-bucket attribution dict
    (see trade_attribution.attribute_trades). Stored as JSON text;
    None or {} is persisted as an empty string for backwards compat.

    `results["trades_export_path"]` (if present) is persisted so the R7.4
    correlation gate can locate this strategy's trade list later. Optional
    — strategies backtested before R2d won't have it.
    """
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()

    attribution_json = json.dumps(attribution) if attribution else ""

    conn.execute(
        """INSERT INTO backtest_results (strategy_id, timerange, sharpe, sortino,
           max_drawdown_pct, max_drawdown_abs, profit_total_pct, profit_total_abs,
           profit_factor, total_trades, win_rate, backtest_days, avg_duration,
           attribution_json, trades_export_path, backtest_data_end_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            strategy_id,
            results.get("timerange", ""),
            results.get("sharpe", 0),
            results.get("sortino", 0),
            results.get("max_drawdown_pct", 0),
            results.get("max_drawdown_abs", 0),
            results.get("profit_total_pct", 0),
            results.get("profit_total_abs", 0),
            results.get("profit_factor", 0),
            results.get("total_trades", 0),
            results.get("win_rate", 0),
            results.get("backtest_days", 0),
            results.get("avg_duration", ""),
            attribution_json,
            results.get("trades_export_path", ""),
            # backtest_data_end_at: when the BACKTEST RAN AGAINST WHAT data.
            # Required for deployment-eligibility freshness check — see
            # docs/deployment-lifecycle.md. Empty string if parser couldn't
            # extract it (legacy backtests pre-this-PR); the backfill
            # migration covers existing rows.
            results.get("backtest_data_end_at", ""),
            now,
        ),
    )
    conn.commit()
    conn.close()
    log.info(f"Recorded backtest for strategy_id={strategy_id}: "
             f"Sharpe={results.get('sharpe', 0)}, "
             f"Profit={results.get('profit_total_pct', 0)}%"
             + (f", attribution buckets={len(attribution.get('buckets', {}))}"
                if attribution else ""))


def promote_strategy(strategy_id: int):
    """Promote a candidate to active status."""
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Check active pool limit
    count = conn.execute(
        "SELECT COUNT(*) FROM strategies WHERE status = 'active'"
    ).fetchone()[0]

    if count >= MAX_ACTIVE:
        # Demote the worst-performing active strategy
        worst = conn.execute("""
            SELECT s.id, s.name, COALESCE(MAX(br.sharpe), -999) as best_sharpe
            FROM strategies s
            LEFT JOIN backtest_results br ON s.id = br.strategy_id
            WHERE s.status = 'active'
            GROUP BY s.id
            ORDER BY best_sharpe ASC
            LIMIT 1
        """).fetchone()
        if worst:
            conn.execute(
                "UPDATE strategies SET status = 'retired', retired_at = ? WHERE id = ?",
                (now, worst["id"]),
            )
            log.info(f"Auto-retired weakest active strategy: {worst['name']}")

    conn.execute(
        "UPDATE strategies SET status = 'active', promoted_at = ? WHERE id = ?",
        (now, strategy_id),
    )
    conn.commit()
    conn.close()
    log.info(f"Promoted strategy_id={strategy_id} to active")


def retire_strategy(strategy_id: int, reason: str = "", verdict: str = ""):
    """Retire a strategy and persist the failure reason + verdict.

    `verdict` is a short code (e.g. FAIL_MINI, FAIL_FULL, FAIL_TOO_FEW,
    FAIL_SANITY, DEMOTED_POOL, AUTO_RETIRED) used by the generator's
    failure memory to avoid repeating known-bad approaches.
    """
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """UPDATE strategies
           SET status = 'retired', retired_at = ?,
               failure_reason = ?, failure_verdict = ?
           WHERE id = ?""",
        (now, reason or "", verdict or "", strategy_id),
    )
    conn.commit()
    conn.close()
    log.info(f"Retired strategy_id={strategy_id} [{verdict}]: {reason}")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_active_strategies() -> list:
    """Get all active strategies."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM strategies WHERE status = 'active' ORDER BY promoted_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_candidates() -> list:
    """Get all candidate strategies, oldest first (FIFO).

    Order matters because ``job_backtest_candidates`` processes a bounded
    slice per run (currently 25). With newest-first ordering, a candidate
    that lands at position 26+ on Sunday never gets evaluated — and the
    next Sunday it's even further back as fresh generations push it down,
    until archetype-eviction retires it without ever scoring it. FIFO
    guarantees every candidate eventually gets a full backtest, at most
    one week after registration assuming weekly generation stays below
    the per-run cap. ``id ASC`` tiebreaks deterministic ordering when
    two candidates share a ``created_at`` second (common in batch
    registration).
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM strategies WHERE status = 'candidate' "
        "ORDER BY created_at ASC, id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_hyperopt_candidates(limit: int = 3, max_age_days: int = 14) -> list:
    """Return recently-retired strategies that hyperopt should attempt to rescue.

    Eligible: status='retired' with failure_verdict in FAIL_TOO_FEW or
    FAIL_UNPROFITABLE (NOT FAIL_BACKTEST — those crashed so the code is broken).
    Restricted to retirements within the last `max_age_days` so we don't
    re-process strategies the user already saw and dismissed.

    Sort: highest total_trades first. Strategies with some trades but no edge
    are MUCH easier to rescue than strategies with zero trades; the latter
    are over-constrained at the logic level, not just the threshold level.
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT s.id, s.name, s.filepath, s.thesis, s.target_regime, s.generation_id,
               s.failure_verdict, s.failure_reason, s.retired_at,
               COALESCE(br.total_trades, 0) AS total_trades,
               COALESCE(br.sharpe, 0) AS sharpe,
               COALESCE(br.profit_total_pct, 0) AS profit_total_pct
        FROM strategies s
        LEFT JOIN backtest_results br
          ON br.id = (SELECT MAX(id) FROM backtest_results WHERE strategy_id = s.id)
        WHERE s.status = 'retired'
          AND s.failure_verdict IN ('FAIL_TOO_FEW', 'FAIL_UNPROFITABLE')
          AND s.retired_at >= datetime('now', ?)
        ORDER BY total_trades DESC, s.retired_at DESC
        LIMIT ?
    """, (f"-{max_age_days} days", limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_hyperopt_outcome(
    strategy_id: int,
    verdict: str,
    reason: str = "",
    promote: bool = False,
) -> None:
    """Update a hyperopted strategy's verdict + optionally promote to active.

    verdict should be HYPEROPT_PROMOTE (rescued) or HYPEROPT_NO_EDGE (still
    failing). promote=True flips status to 'active' AND clears retired_at —
    use ONLY when verdict=HYPEROPT_PROMOTE.
    """
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    if promote:
        conn.execute(
            """UPDATE strategies
               SET status = 'active', promoted_at = ?, retired_at = NULL,
                   failure_verdict = ?, failure_reason = ?
               WHERE id = ?""",
            (now, verdict, reason, strategy_id),
        )
        log.info(f"Hyperopt PROMOTED strategy_id={strategy_id}: {reason}")
    else:
        conn.execute(
            """UPDATE strategies
               SET failure_verdict = ?, failure_reason = ?
               WHERE id = ?""",
            (verdict, reason, strategy_id),
        )
        log.info(f"Hyperopt outcome strategy_id={strategy_id} [{verdict}]: {reason}")
    conn.commit()
    conn.close()


def get_best_strategy_for_regime(regime: str) -> dict | None:
    """Get the best active strategy for a given regime, ranked by Sharpe on latest backtest."""
    conn = get_db()
    row = conn.execute("""
        SELECT s.*, br.sharpe, br.profit_total_pct, br.max_drawdown_pct, br.total_trades
        FROM strategies s
        JOIN backtest_results br ON s.id = br.strategy_id
        WHERE s.status = 'active'
          AND (s.target_regime = ? OR s.target_regime = 'all')
          AND br.total_trades >= 20
        ORDER BY br.sharpe DESC, br.max_drawdown_pct ASC
        LIMIT 1
    """, (regime,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_strategy_by_name(name: str) -> dict | None:
    """Look up a strategy by class name."""
    conn = get_db()
    row = conn.execute("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_strategies(status: str = None) -> list:
    """Get strategies, optionally filtered by status."""
    conn = get_db()
    if status:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM strategies ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Deployment lifecycle queries (Phase 1 — used by the reconciler in Phase 2+)
# ---------------------------------------------------------------------------

def get_deployment_eligible(
    *,
    min_trades: int = 20,
    max_drawdown_pct: float = 15.0,
    max_backtest_age_days: int = 30,
    max_data_age_days: int = 30,
) -> list:
    """Strategies that pass every hard filter and are candidates for the
    greedy correlation-aware selection step.

    Filters (all must hold against the LATEST backtest_results row per
    strategy):

      research_status = 'approved'
      deployment_status IN ('not_deployed', 'stopped') and NOT
          currently inside its deployment_blocked_until cooldown
      latest backtest:
          total_trades       >= min_trades
          profit_total_pct   >  0
          sharpe             >  0
          max_drawdown_pct   <  max_drawdown_pct
          created_at         within max_backtest_age_days
          backtest_data_end_at within max_data_age_days
              (recent BACKTEST RUN is not enough — the underlying market
               data must also be recent; a re-run of a stale candle set
               doesn't count as a fresh validation)

    Returned in NO particular order. The reconciler sorts and applies
    greedy correlation skip on top.
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT s.id, s.name, s.archetype, s.target_regime,
               COALESCE(br.sharpe, 0)            AS sharpe,
               COALESCE(br.profit_total_pct, 0)  AS profit_total_pct,
               COALESCE(br.max_drawdown_pct, 0)  AS max_drawdown_pct,
               COALESCE(br.total_trades, 0)      AS total_trades,
               COALESCE(br.trades_export_path, '') AS trades_export_path,
               br.created_at                      AS backtest_at,
               COALESCE(br.backtest_data_end_at, '') AS backtest_data_end_at
        FROM strategies s
        LEFT JOIN backtest_results br
          ON br.id = (SELECT MAX(id) FROM backtest_results WHERE strategy_id = s.id)
        WHERE s.research_status = 'approved'
          AND s.deployment_status IN ('not_deployed', 'stopped')
          AND (s.deployment_blocked_until IS NULL
               OR datetime(s.deployment_blocked_until) <= datetime('now'))
          AND COALESCE(br.total_trades, 0)     >= ?
          AND COALESCE(br.profit_total_pct, 0) > 0
          AND COALESCE(br.sharpe, 0)           > 0
          AND COALESCE(br.max_drawdown_pct, 0) < ?
          AND br.created_at >= datetime('now', ?)
          AND br.backtest_data_end_at != ''
          AND br.backtest_data_end_at >= datetime('now', ?)
    """, (
        min_trades,
        max_drawdown_pct,
        f"-{max_backtest_age_days} days",
        f"-{max_data_age_days} days",
    )).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_currently_deployed() -> list:
    """Strategies the registry believes are deployed RIGHT NOW.

    Used by:
      - reconciler diff against `docker ps --filter label=first_duck.role=deployed-strategy`
      - replacement evaluation (compare challenger to currently-deployed)
      - drift alarm (Phase 4)
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT id, name, archetype, target_regime, deployed_at
        FROM strategies
        WHERE deployment_status = 'deployed'
        ORDER BY deployed_at ASC NULLS LAST
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_active_strategies_with_trade_paths() -> list:
    """Return active strategies paired with the file path of their most recent
    trade export (R7.4 correlation gate input).

    Each row: {id, name, target_regime, trades_export_path}. Strategies whose
    latest backtest has no stored trade export — pre-R2d strategies or
    backtests run with export_trades=False — get an empty string for the
    path. The correlation gate must skip those gracefully.
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT s.id, s.name, s.target_regime,
               COALESCE(br.trades_export_path, '') AS trades_export_path
        FROM strategies s
        LEFT JOIN backtest_results br
          ON br.id = (SELECT MAX(id) FROM backtest_results WHERE strategy_id = s.id)
        WHERE s.status = 'active'
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_attributions(n: int = 10, min_trades: int = 10) -> list:
    """Return the N most recent backtest rows that have stored attribution
    data and at least `min_trades` trades (so noise from 2-trade backtests
    doesn't pollute the reflector prompt).

    Each row carries the parsed attribution dict under the 'attribution' key.
    Rows whose attribution_json can't be parsed are silently skipped — the
    reflector should never be blocked by one bad row.
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT s.name, s.target_regime, s.archetype, s.status, s.thesis,
               br.total_trades, br.profit_total_pct, br.sharpe,
               br.attribution_json, br.created_at
        FROM backtest_results br
        JOIN strategies s ON s.id = br.strategy_id
        WHERE br.attribution_json != ''
          AND br.total_trades >= ?
        ORDER BY br.created_at DESC
        LIMIT ?
    """, (min_trades, n)).fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        try:
            d["attribution"] = json.loads(d["attribution_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        results.append(d)
    return results


def get_recent_failures(k: int = 8, regime: str | None = None) -> list:
    """Return the most recently retired candidates with a populated failure_verdict.

    Used by the generator to build a "don't repeat these" section of the prompt.
    Auto-retires from pool overflow (empty failure_verdict) are excluded.

    Each row is enriched with the strategy's latest backtest metrics (when
    available) and a short code excerpt, so the LLM sees *why* the strategy
    failed — not just that it did.
    """
    conn = get_db()
    params: list = []
    regime_clause = ""
    if regime and regime != "all":
        regime_clause = "AND (s.target_regime = ? OR s.target_regime = 'all')"
        params.append(regime)
    params.append(k)

    rows = conn.execute(f"""
        SELECT s.id, s.name, s.thesis, s.target_regime, s.archetype, s.generation_id,
               s.filepath, s.failure_reason, s.failure_verdict, s.retired_at,
               br.sharpe, br.profit_total_pct, br.total_trades, br.max_drawdown_pct
        FROM strategies s
        LEFT JOIN backtest_results br
          ON br.id = (SELECT MAX(id) FROM backtest_results WHERE strategy_id = s.id)
        WHERE s.status = 'retired'
          AND s.failure_verdict != ''
          {regime_clause}
        ORDER BY s.retired_at DESC
        LIMIT ?
    """, params).fetchall()
    conn.close()

    failures = []
    for row in rows:
        d = dict(row)
        d["code_excerpt"] = _extract_entry_logic(d.get("filepath", ""))
        failures.append(d)
    return failures


def _extract_entry_logic(filepath: str, max_lines: int = 30) -> str:
    """Pull populate_entry_trend body (roughly) as a compact excerpt."""
    try:
        src = Path(filepath).read_text()
    except Exception:
        return ""
    lines = src.splitlines()
    out: list = []
    capture = False
    indent = None
    for line in lines:
        if not capture and "def populate_entry_trend" in line:
            capture = True
            indent = len(line) - len(line.lstrip())
            out.append(line.strip())
            continue
        if capture:
            if line.strip() == "":
                out.append("")
                continue
            cur = len(line) - len(line.lstrip())
            if cur <= indent and line.strip() and not line.lstrip().startswith("#"):
                break
            out.append(line.rstrip())
            if len(out) >= max_lines:
                out.append("    # ... (truncated)")
                break
    return "\n".join(out)


def load_recent_reflections(n: int = 2, max_chars: int = 4000) -> str:
    """Read the latest N reflector markdown files, newest first, concatenated.

    Returns empty string if the reflections dir doesn't exist or has no files.
    Truncated to max_chars to keep the prompt bounded.
    """
    if not REFLECTIONS_DIR.exists():
        return ""
    files = sorted(
        REFLECTIONS_DIR.glob("reflection-*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:n]
    if not files:
        return ""
    chunks = []
    for f in files:
        try:
            chunks.append(f"--- {f.name} ---\n{f.read_text()}")
        except Exception as e:
            log.warning(f"Failed to read reflection {f}: {e}")
    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated]"
    return text


def get_registry_stats() -> dict:
    """Get summary statistics of the registry."""
    conn = get_db()
    stats = {}
    for status in ("candidate", "active", "retired"):
        stats[status] = conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE status = ?", (status,)
        ).fetchone()[0]
    stats["total_backtests"] = conn.execute(
        "SELECT COUNT(*) FROM backtest_results"
    ).fetchone()[0]
    conn.close()
    return stats


# ---------------------------------------------------------------------------
# Init on import
# ---------------------------------------------------------------------------
init_db()
