"""
Persistent chat memory stored in SQLite.
Each conversation is a named session.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)   # ← create on Windows too
DB_PATH = _CACHE_DIR / "chat_memory.db"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    return conn


_conn = _db()


# ── Sessions ─────────────────────────────────────────────────────────────────

def create_session(name: Optional[str] = None, model: str = "") -> int:
    name = name or f"Chat {datetime.now().strftime('%b %d %H:%M')}"
    cur = _conn.execute(
        "INSERT INTO sessions (name, model) VALUES (?,?)", (name, model)
    )
    _conn.commit()
    return cur.lastrowid


def list_sessions() -> List[Dict]:
    rows = _conn.execute(
        """SELECT s.id, s.name, s.model, s.created_at, s.updated_at,
                  COUNT(m.id) as msg_count
           FROM sessions s
           LEFT JOIN messages m ON m.session_id = s.id
           GROUP BY s.id
           ORDER BY s.updated_at DESC"""
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "model": r[2],
         "created_at": r[3], "updated_at": r[4], "msg_count": r[5]}
        for r in rows
    ]


def delete_session(session_id: int):
    _conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    _conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    _conn.commit()


def rename_session(session_id: int, name: str):
    _conn.execute("UPDATE sessions SET name=? WHERE id=?", (name, session_id))
    _conn.commit()


def update_session_model(session_id: int, model: str):
    _conn.execute("UPDATE sessions SET model=? WHERE id=?", (model, session_id))
    _conn.commit()


# ── Messages ─────────────────────────────────────────────────────────────────

def add_message(session_id: int, role: str, content: str):
    _conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
        (session_id, role, content)
    )
    _conn.execute(
        "UPDATE sessions SET updated_at=datetime('now') WHERE id=?",
        (session_id,)
    )
    _conn.commit()


def get_messages(session_id: int, limit: int = 50) -> List[Dict]:
    rows = _conn.execute(
        """SELECT role, content, ts FROM messages
           WHERE session_id=?
           ORDER BY id DESC LIMIT ?""",
        (session_id, limit)
    ).fetchall()
    return [{"role": r[0], "content": r[1], "ts": r[2]} for r in reversed(rows)]


def get_history_for_llm(session_id: int, max_messages: int = 20) -> List[Dict]:
    """Return recent messages in Ollama chat format."""
    msgs = get_messages(session_id, limit=max_messages)
    return [{"role": m["role"], "content": m["content"]} for m in msgs]
