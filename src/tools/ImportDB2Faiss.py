"""Veterinary document store + vector index.

    VetDB     SQLite CRUD over the vet_doc table
    VetFAISS  build / search the vector index

Backend-agnostic: the actual vectorisation lives in ``embedders.py`` and is
chosen by ``retrieval.backend`` in config ("tfidf" or "openai").

Three design points worth knowing
---------------------------------

**1. Cosine similarity, higher is better.**
All embedders return L2-normalised vectors, so ``IndexFlatIP`` (inner product)
gives cosine similarity directly, in ``-1..1``. One score scale for both
backends, and the threshold reads naturally: keep hits at or above
``min_similarity``.

*This is a polarity flip from the earlier version*, which used ``IndexFlatL2``
and an upper bound on distance. If you have old notes saying "lower is better",
they are out of date. Relationship for unit vectors: ``L2² = 2(1 − cos)``.

**2. The index maps by data_id, not by position.**
``IndexIDMap2`` stores each vector against its ``data_id``, so a search returns
database IDs. Two consequences: deleting a row can no longer silently shift the
mapping and return the wrong record, and lookup fetches only the top-k rows
instead of ``SELECT *`` over the whole table.

**3. TF-IDF must be rebuilt wholesale; OpenAI vectors can be cached.**
``embedder.is_corpus_coupled`` says which. TF-IDF weights every term by how rare
it is across the corpus, so adding one record invalidates every vector. OpenAI
vectors are independent, so unchanged records are served from the embedding
cache and only new or edited text is re-sent.

A note on the data: in vet_doc the ``symptoms`` column holds the disease name
and ``cause`` holds the symptom list (see ARCHITECTURE.md §6.2). ``_search_text``
is the single place that decides what gets embedded.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from tools.config_loader import ConfigManager
from tools.embedders import make_embedder
from tools.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS vet_doc (
    data_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    symptoms  TEXT NOT NULL,
    cause     TEXT,
    diagnosis TEXT,
    treatment TEXT
);
"""


class VetDB:
    @staticmethod
    def _connect(db_name) -> sqlite3.Connection:
        db_path = Path(db_name)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def create_tables(db_name) -> None:
        with VetDB._connect(db_name) as conn:
            conn.executescript(SCHEMA)

    @staticmethod
    def get_all_vet_doc(db_name) -> List[dict]:
        VetDB.create_tables(db_name)
        with VetDB._connect(db_name) as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM vet_doc ORDER BY data_id")]

    @staticmethod
    def get_by_ids(db_name, data_ids) -> dict:
        """Fetch only the rows a search actually hit. Returns {data_id: row}."""
        data_ids = [int(i) for i in data_ids]
        if not data_ids:
            return {}
        placeholders = ",".join("?" * len(data_ids))
        with VetDB._connect(db_name) as conn:
            rows = conn.execute(
                f"SELECT * FROM vet_doc WHERE data_id IN ({placeholders})", data_ids
            ).fetchall()
        return {r["data_id"]: dict(r) for r in rows}

    @staticmethod
    def insert_many(db_name, records) -> int:
        VetDB.create_tables(db_name)
        inserted = 0
        with VetDB._connect(db_name) as conn:
            for r in records:
                conn.execute(
                    """INSERT INTO vet_doc (symptoms, cause, diagnosis, treatment)
                       VALUES (:symptoms, :cause, :diagnosis, :treatment)""",
                    {
                        "symptoms": r["symptoms"],
                        "cause": r.get("cause"),
                        "diagnosis": r.get("diagnosis"),
                        "treatment": r.get("treatment"),
                    },
                )
                inserted += 1
        return inserted


def _search_text(record: dict) -> str:
    """The text a query is matched against — the symptom description.

    Write this the way an OWNER would describe the problem, including colloquial
    synonyms. With the tfidf backend a word absent from here can never match.
    """
    return ((record.get("cause") or "") + " " + (record.get("symptoms") or "")).strip()


# ---------------------------------------------------------------------------
# In-process cache: the index and embedder are immutable between rebuilds, so
# there is no reason to read them from disk on every request. Keyed on the
# index file's mtime, which a rebuild changes — so the cache self-invalidates.
_RUNTIME: dict = {}


class VetFAISS:
    # -- build -----------------------------------------------------------
    @staticmethod
    def build_index(db_name=None) -> int:
        cfg = ConfigManager.get_config_database()
        retrieval = ConfigManager.get_retrieval_config()
        db_name = db_name or cfg["DB_NAME"]

        records = VetDB.get_all_vet_doc(db_name)
        if not records:
            logger.warning("vet_doc is empty — nothing to index")
            return 0

        corpus = [_search_text(r) for r in records]
        ids = np.array([r["data_id"] for r in records], dtype="int64")

        embedder = make_embedder(retrieval["backend"], retrieval["embedding_model"])

        if embedder.is_corpus_coupled:
            # TF-IDF: vocabulary and IDF depend on the whole corpus
            embedder.fit(corpus)
            vectors = embedder.encode(corpus)
            detail = ""  # cache does not apply — TF-IDF always refits wholesale
        else:
            # OpenAI: independent vectors, so reuse whatever is already cached
            cache = EmbeddingCache(cfg["EMBEDDING_CACHE"])
            vectors, from_cache, embedded = cache.encode_with_cache(embedder, corpus)
            detail = f" ({from_cache} from cache, {embedded} newly embedded)"

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(vectors, ids)

        cfg["FAISS_INDEX_FILE"].parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(cfg["FAISS_INDEX_FILE"]))
        embedder.save(cfg["EMBEDDER_FILE"])
        cfg["INDEX_META_FILE"].write_text(
            json.dumps(
                {
                    "backend": retrieval["backend"],
                    "embedder": embedder.name,
                    "dim": int(vectors.shape[1]),
                    "records": len(records),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _RUNTIME.clear()

        print(
            f"✅ Indexed {len(records)} records "
            f"[{retrieval['backend']}/{embedder.name}, dim={vectors.shape[1]}]{detail}"
        )
        return len(records)

    # -- load ------------------------------------------------------------
    @staticmethod
    def _load():
        cfg = ConfigManager.get_config_database()
        retrieval = ConfigManager.get_retrieval_config()
        index_file = cfg["FAISS_INDEX_FILE"]

        if not index_file.exists() or not cfg["EMBEDDER_FILE"].exists():
            logger.info("Index missing — building")
            VetFAISS.build_index(cfg["DB_NAME"])

        if not cfg["INDEX_META_FILE"].exists():
            # An index with no metadata predates the embedder refactor: it may be
            # an IndexFlatL2 keyed by position, which this code would misread as
            # data_ids. Rebuild rather than trust it.
            logger.info("Index has no metadata (pre-refactor artifact) — rebuilding")
            VetFAISS.build_index(cfg["DB_NAME"])

        meta = json.loads(cfg["INDEX_META_FILE"].read_text(encoding="utf-8"))
        if meta.get("backend") and meta["backend"] != retrieval["backend"]:
            logger.info(
                "Index was built with backend %r but config says %r — rebuilding",
                meta["backend"], retrieval["backend"],
            )
            VetFAISS.build_index(cfg["DB_NAME"])

        key = (str(index_file), index_file.stat().st_mtime_ns)
        if key not in _RUNTIME:
            _RUNTIME.clear()
            embedder = make_embedder(retrieval["backend"], retrieval["embedding_model"])
            embedder.load(cfg["EMBEDDER_FILE"])
            _RUNTIME[key] = (faiss.read_index(str(index_file)), embedder)
        return _RUNTIME[key]

    # -- search ----------------------------------------------------------
    @staticmethod
    def search_vet_doc(
        query_text: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> List[dict]:
        """Closest vet_doc records for a free-text symptom query.

        ``score`` in each result is cosine similarity in -1..1. **Higher is
        better** — this is the opposite of the old distance-based version.
        """
        cfg = ConfigManager.get_config_database()
        retrieval = ConfigManager.get_retrieval_config()
        top_k = top_k or retrieval["top_k"]
        if min_similarity is None:
            min_similarity = retrieval["min_similarity"]

        if not query_text or not query_text.strip():
            return []

        index, embedder = VetFAISS._load()
        if index.ntotal == 0:
            return []

        query_vec = embedder.encode([query_text])
        scores, ids = index.search(query_vec, min(top_k, index.ntotal))

        hit_ids = [int(i) for i in ids[0] if i != -1]
        rows = VetDB.get_by_ids(cfg["DB_NAME"], hit_ids)

        results = []
        for score, data_id in zip(scores[0], ids[0]):
            if data_id == -1:
                continue
            if float(score) < min_similarity:
                continue
            rec = rows.get(int(data_id))
            if not rec:
                logger.warning("Index references data_id %s which is not in vet_doc", data_id)
                continue
            results.append(
                {
                    "data_id": rec["data_id"],
                    "disease": rec["symptoms"],
                    "symptoms": rec["cause"],
                    "cause": rec["diagnosis"],
                    "treatment": rec["treatment"],
                    "score": round(float(score), 4),
                }
            )
        return results

    @staticmethod
    def index_info() -> dict:
        cfg = ConfigManager.get_config_database()
        if not cfg["INDEX_META_FILE"].exists():
            return {}
        return json.loads(cfg["INDEX_META_FILE"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    count = VetFAISS.build_index()
    if count:
        query = "ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน มีสะเก็ดแห้งเป็นขุย"
        print(f"\n🔍 {query}")
        for hit in VetFAISS.search_vet_doc(query, top_k=3, min_similarity=-1):
            print(f"  [{hit['score']}] {hit['disease']}")
