"""
Strategy Registry — SQLite-based strategy lifecycle management.

Tracks all strategies through their lifecycle:
  candidate -> active -> retired

The orchestrator is the single writer. All other components read only.

Schema:
  strategies: id, name, filepath, thesis, target_regime, generation_id,
              status, created_at, promoted_at, retired_at
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

# Pool limits
MAX_ACTIVE = 10
MAX_CANDIDATES = 30


def get_db() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
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
            retired_at TEXT
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
) -> int:
    """Register a new candidate strategy. Returns strategy ID."""
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Check candidate pool limit
    count = conn.execute(
        "SELECT COUNT(*) FROM strategies WHERE status = 'candidate'"
    ).fetchone()[0]

    if count >= MAX_CANDIDATES:
        # Retire the oldest candidate
        oldest = conn.execute(
            "SELECT id, name FROM strategies WHERE status = 'candidate' "
            "ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if oldest:
            conn.execute(
                "UPDATE strategies SET status = 'retired', retired_at = ? WHERE id = ?",
                (now, oldest["id"]),
            )
            log.info(f"Auto-retired oldest candidate: {oldest['name']}")

    cursor = conn.execute(
        """INSERT INTO strategies (name, filepath, thesis, target_regime, generation_id,
           status, created_at) VALUES (?, ?, ?, ?, ?, 'candidate', ?)""",
        (name, str(filepath), thesis, target_regime, generation_id, now),
    )
    conn.commit()
    strategy_id = cursor.lastrowid
    conn.close()
    log.info(f"Registered strategy: {name} (id={strategy_id}, regime={target_regime})")
    return strategy_id


def record_backtest(strategy_id: int, results: dict):
    """Record backtest results for a strategy."""
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO backtest_results (strategy_id, timerange, sharpe, sortino,
           max_drawdown_pct, max_drawdown_abs, profit_total_pct, profit_total_abs,
           profit_factor, total_trades, win_rate, backtest_days, avg_duration, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            now,
        ),
    )
    conn.commit()
    conn.close()
    log.info(f"Recorded backtest for strategy_id={strategy_id}: "
             f"Sharpe={results.get('sharpe', 0)}, "
             f"Profit={results.get('profit_total_pct', 0)}%")


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


def retire_strategy(strategy_id: int, reason: str = ""):
    """Retire a strategy (demote from active or remove from candidate pool)."""
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE strategies SET status = 'retired', retired_at = ? WHERE id = ?",
        (now, strategy_id),
    )
    conn.commit()
    conn.close()
    log.info(f"Retired strategy_id={strategy_id}: {reason}")


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
    """Get all candidate strategies."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM strategies WHERE status = 'candidate' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
