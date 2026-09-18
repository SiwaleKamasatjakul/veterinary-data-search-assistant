"""Chat history storage.

Changes from the Linux original:
* The DB path comes from config (absolute) instead of the bare relative string
  ``"chat_history.db"``, which created a stray empty database in whatever
  directory the server happened to be launched from.
* One connection helper instead of three duplicated ``sqlite3.connect`` calls.
* Timezone-aware UTC timestamps (``datetime.utcnow()`` is deprecated in 3.12).
"""

import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

from tools.config_loader import ConfigManager

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    user_message TEXT,
    ai_response  TEXT,
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);
"""


def _connect() -> sqlite3.Connection:
    db_path = ConfigManager.get_config_database()["CHAT_HISTORY_DB"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


class DatabaseProcess:
    @staticmethod
    def init_database() -> None:
        with _connect() as conn:
            conn.executescript(SCHEMA)

    @staticmethod
    def save_message(session_id: str, user_message: str, ai_response: str) -> None:
        timestamp = datetime.now(timezone.utc)
        with _connect() as conn:
            conn.execute(
                """INSERT INTO chat_history (session_id, user_message, ai_response, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (session_id, user_message, ai_response, timestamp),
            )

    @staticmethod
    def get_chat_history(session_id: str, limit: int = 20) -> List[Tuple[str, str]]:
        """Return [(user_message, ai_response), ...] oldest-first for this session."""
        with _connect() as conn:
            rows = conn.execute(
                """SELECT user_message, ai_response FROM chat_history
                   WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return list(reversed(rows))
