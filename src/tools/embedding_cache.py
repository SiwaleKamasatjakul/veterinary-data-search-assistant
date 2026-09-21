"""Embedding cache — so a rebuild does not re-pay for text that did not change.

Keyed by ``(model, sha256(text))``. Change one record's wording and only that
record is re-sent to the API; the other nineteen come from disk.

This only makes sense for corpus-independent embedders (OpenAI). TF-IDF vectors
depend on the whole corpus, so they are never cached here — a TF-IDF rebuild
always refits from scratch.

Stored as a SQLite table rather than a pickle so it is inspectable, append-safe,
and survives interruption mid-rebuild.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, text_hash)
);
"""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def get_many(self, model: str, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        """Return {text_hash: vector} for whatever is already cached."""
        hashes = [text_hash(t) for t in texts]
        found: Dict[str, np.ndarray] = {}
        with sqlite3.connect(self.path) as conn:
            # chunked IN clause — SQLite caps host parameters around 999
            for start in range(0, len(hashes), 500):
                chunk = hashes[start : start + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT text_hash, dim, vector FROM embedding_cache "
                    f"WHERE model = ? AND text_hash IN ({placeholders})",
                    (model, *chunk),
                ).fetchall()
                for h, dim, blob in rows:
                    found[h] = np.frombuffer(blob, dtype="float32").reshape(dim)
        return found

    def put_many(self, model: str, texts: Sequence[str], vectors: np.ndarray) -> None:
        rows = [
            (model, text_hash(t), int(v.shape[0]), v.astype("float32").tobytes())
            for t, v in zip(texts, vectors)
        ]
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (model, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def encode_with_cache(self, embedder, texts: Sequence[str]) -> Tuple[np.ndarray, int, int]:
        """Encode ``texts``, sending only uncached ones to the embedder.

        Returns ``(vectors, n_from_cache, n_embedded)``.
        """
        texts = list(texts)
        cached = self.get_many(embedder.name, texts)

        missing_idx: List[int] = []
        missing_txt: List[str] = []
        for i, t in enumerate(texts):
            if text_hash(t) not in cached:
                missing_idx.append(i)
                missing_txt.append(t)

        if missing_txt:
            logger.info("Embedding %d new / %d total", len(missing_txt), len(texts))
            fresh = embedder.encode(missing_txt)
            self.put_many(embedder.name, missing_txt, fresh)
            for slot, i in enumerate(missing_idx):
                cached[text_hash(texts[i])] = fresh[slot]

        vectors = np.vstack([cached[text_hash(t)] for t in texts]).astype("float32")
        return vectors, len(texts) - len(missing_txt), len(missing_txt)

    def stats(self) -> dict:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT model, COUNT(*) FROM embedding_cache GROUP BY model"
            ).fetchall()
        return {"path": str(self.path), "by_model": dict(rows)}

    def clear(self, model: str | None = None) -> int:
        with sqlite3.connect(self.path) as conn:
            if model:
                cur = conn.execute("DELETE FROM embedding_cache WHERE model = ?", (model,))
            else:
                cur = conn.execute("DELETE FROM embedding_cache")
            return cur.rowcount
