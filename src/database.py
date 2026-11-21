"""SQLite storage (standard-library sqlite3, no ORM).

Schema versions, tracked with ``PRAGMA user_version``:
  0/1  original single ``history`` table (no owner)
  2    Phase 0: users, sessions, per-user data tables, history ownership

Conventions
-----------
* Every timestamp is UTC text ``YYYY-MM-DD HH:MM:SS``. New tables fill it in
  through a SQLite default. Legacy ``history.created_at`` values keep their
  ``... UTC`` suffix; both forms sort and compare correctly as text.
* ``history.user_id IS NULL`` means *unowned* (legacy) row. Scoping is strict:
  every history function matches ``user_id IS ?``, so ``user_id=None`` touches
  only unowned rows and a user can never reach another user's rows or legacy
  rows. Legacy rows are never assigned to anyone automatically.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA_VERSION = 2
_NOW = "(strftime('%Y-%m-%d %H:%M:%S', 'now'))"  # SQLite 'now' is UTC

# --- version 1: the original schema, byte-for-byte compatible ---------------
_V1_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        input_text TEXT NOT NULL,
        overall_verdict TEXT NOT NULL,
        overall_confidence REAL NOT NULL,
        summary TEXT NOT NULL,
        result_json TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_history_created ON history (created_at DESC)",
)

# --- version 2: accounts and per-user feature tables ------------------------
_V2_STATEMENTS = (
    f"""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
        password_hash TEXT NOT NULL,          -- self-describing (algorithm, params, salt, hash)
        language TEXT NOT NULL DEFAULT 'en',  -- preferred UI language
        created_at TEXT NOT NULL DEFAULT {_NOW},
        last_login_at TEXT
    )""",
    # Only a SHA-256 of the random session token is stored, never the token.
    f"""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT {_NOW},
        expires_at TEXT NOT NULL,
        last_seen_at TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions (expires_at)",
    f"""CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        source_type TEXT NOT NULL,            -- text | url | image
        source_url TEXT,
        source_title TEXT,
        language TEXT,
        short_summary TEXT NOT NULL,
        detailed_summary TEXT NOT NULL,
        key_points_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL DEFAULT {_NOW}
    )""",
    "CREATE INDEX IF NOT EXISTS idx_summaries_user ON summaries (user_id, created_at DESC)",
    f"""CREATE TABLE IF NOT EXISTS user_interests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('category', 'custom')),
        value TEXT NOT NULL COLLATE NOCASE,
        created_at TEXT NOT NULL DEFAULT {_NOW},
        UNIQUE (user_id, kind, value)
    )""",
    f"""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title TEXT NOT NULL DEFAULT '',
        language TEXT,
        created_at TEXT NOT NULL DEFAULT {_NOW},
        updated_at TEXT NOT NULL DEFAULT {_NOW}
    )""",
    "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations (user_id, updated_at DESC)",
    f"""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        sources_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL DEFAULT {_NOW}
    )""",
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id)",
    # Public web articles only (no user text). Source of the personalised feed
    # cache and, with article_hits, of the trending-headlines calculation.
    f"""CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        source_name TEXT NOT NULL DEFAULT '',
        domain TEXT NOT NULL DEFAULT '',
        published_date TEXT,                  -- UTC text when known, else NULL
        language TEXT,
        category TEXT,
        first_seen_at TEXT NOT NULL DEFAULT {_NOW},
        last_seen_at TEXT NOT NULL DEFAULT {_NOW}
    )""",
    "CREATE INDEX IF NOT EXISTS idx_articles_published ON news_articles (published_date)",
    # One row per time an article was shown/used. Trending = distinct users and
    # hits inside a time window (computed in the dashboard phase).
    f"""CREATE TABLE IF NOT EXISTS article_hits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
        origin TEXT NOT NULL,                 -- news | evidence | ask
        hit_at TEXT NOT NULL DEFAULT {_NOW}
    )""",
    "CREATE INDEX IF NOT EXISTS idx_hits_time ON article_hits (hit_at)",
    "CREATE INDEX IF NOT EXISTS idx_hits_article ON article_hits (article_id)",
    # Anonymous feature-usage events for the daily-activity chart. If an
    # account is deleted the row stays, with user_id set to NULL.
    f"""CREATE TABLE IF NOT EXISTS activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
        feature TEXT NOT NULL,                -- check | summarize | news | ask | ocr
        created_at TEXT NOT NULL DEFAULT {_NOW}
    )""",
    "CREATE INDEX IF NOT EXISTS idx_activity_time ON activity_log (created_at)",
)

# New nullable columns on the existing history table (NULL = unknown/unowned).
_HISTORY_NEW_COLUMNS = (
    ("user_id", "INTEGER REFERENCES users(id) ON DELETE CASCADE"),
    ("input_type", "TEXT"),    # text | url | image
    ("language", "TEXT"),      # en | hi | mr
    ("sentiment", "TEXT"),     # positive | negative | neutral
    ("category", "TEXT"),
)


# --------------------------------------------------------------- connection
def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)  # default 5 s busy timeout
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite
    return conn


# ---------------------------------------------------------------- migration
def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    for stmt in _V2_STATEMENTS:
        conn.execute(stmt)
    existing = _columns(conn, "history")
    for name, ddl in _HISTORY_NEW_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE history ADD COLUMN {name} {ddl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user "
                 "ON history (user_id, created_at DESC)")


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    for stmt in _V1_STATEMENTS:
        conn.execute(stmt)
    if from_version < 2:
        _migrate_to_v2(conn)


def init_db(db_path: str) -> None:
    """Create or upgrade the database. Safe to call repeatedly and concurrently.

    The whole upgrade runs in one ``BEGIN IMMEDIATE`` transaction: it is
    all-or-nothing, and a second process that arrives meanwhile waits, then
    sees the new version and does nothing.
    """
    with closing(_connect(db_path)) as conn:
        if _user_version(conn) >= SCHEMA_VERSION:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = _user_version(conn)  # re-check now that we hold the lock
            if version < SCHEMA_VERSION:
                _migrate(conn, version)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ------------------------------------------------------------------ history
def _check_user_id(user_id: int | None) -> int | None:
    if user_id is None:
        return None
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise ValueError("user_id must be an int or None")
    return user_id


def save_analysis(db_path: str, result: dict, user_id: int | None = None) -> int:
    user_id = _check_user_id(user_id)
    init_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO history (created_at, input_text, overall_verdict, "
            "overall_confidence, summary, result_json, user_id, input_type, "
            "language, sentiment, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (result["created_at"], result["input_text"], result["overall_verdict"],
             float(result["overall_confidence"]), result["summary"],
             json.dumps(result, ensure_ascii=False), user_id,
             result.get("input_type"), result.get("language"),
             result.get("sentiment"), result.get("category")))
        return int(cur.lastrowid)


def list_history(db_path: str, limit: int = 50, user_id: int | None = None) -> list[dict]:
    user_id = _check_user_id(user_id)
    init_db(db_path)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, created_at, input_text, overall_verdict, overall_confidence, "
            "input_type, language, sentiment, category "
            "FROM history WHERE user_id IS ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_analysis(db_path: str, analysis_id: int, user_id: int | None = None) -> dict | None:
    user_id = _check_user_id(user_id)
    init_db(db_path)
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT result_json FROM history WHERE id = ? AND user_id IS ?",
                           (analysis_id, user_id)).fetchone()
    if not row:
        return None
    data = json.loads(row["result_json"])
    data["id"] = analysis_id
    return data


def delete_analysis(db_path: str, analysis_id: int, user_id: int | None = None) -> bool:
    user_id = _check_user_id(user_id)
    init_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM history WHERE id = ? AND user_id IS ?",
                            (analysis_id, user_id)).rowcount > 0


def clear_history(db_path: str, user_id: int | None = None) -> int:
    """Delete one owner's history only (``None`` = the unowned legacy rows)."""
    user_id = _check_user_id(user_id)
    init_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM history WHERE user_id IS ?", (user_id,)).rowcount