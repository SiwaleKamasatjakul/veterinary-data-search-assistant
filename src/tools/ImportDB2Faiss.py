"""Veterinary document store + TF-IDF/FAISS index.

Changes from the Linux original — the reason the old artifacts could not move
between machines:

* The index is saved with ``faiss.write_index`` / ``faiss.read_index`` instead of
  ``pickle.dump(faiss_index)``. The old pickle embedded the SWIG module name
  ``faiss.swigfaiss_avx2`` (an x86-64 AVX2 build), so loading it on Apple Silicon
  raised ``ModuleNotFoundError: No module named 'faiss.swigfaiss_avx2'``.
  ``write_index`` produces an architecture-independent file.
* The vectorizer is saved with ``joblib`` (still a pickle under the hood, but the
  index no longer depends on it, and it is rebuilt by ``build_index``).
* Result assembly used ``if i < len(records) and j < len(D[0]) or D[0][j] >= threshold``
  — ``and`` binds tighter than ``or``, so an out-of-range FAISS id (-1 on a short
  index) could still pass the filter and raise IndexError. Now bounds are checked
  first and the distance filter is separate.
* Distances from ``IndexFlatL2`` are L2 *distances*: smaller is better. The old
  code compared them against a ``similarity_threshold`` of 0.8 as if bigger were
  better. The filter is now ``distance <= max_distance``.
* Scores are converted to plain floats so the results are JSON-serialisable
  (numpy.float32 is not).

A note on the data: in vet_doc the ``symptoms`` column actually holds the disease
name and ``cause`` holds the symptom list. The original indexed column 2 (``cause``),
which is the right text to search on, so that behaviour is preserved here and the
fields are surfaced under honest names in the results.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import faiss
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tools.config_loader import ConfigManager

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
            rows = conn.execute("SELECT * FROM vet_doc ORDER BY data_id").fetchall()
        return [dict(r) for r in rows]

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
    """The text a query is matched against — the symptom description."""
    return (record.get("cause") or "") + " " + (record.get("symptoms") or "")


class VetFAISS:
    @staticmethod
    def build_index(db_name=None) -> int:
        """Fit the TF-IDF vectorizer, build the FAISS index, write both to disk."""
        cfg = ConfigManager.get_config_database()
        db_name = db_name or cfg["DB_NAME"]
        index_file = cfg["FAISS_INDEX_FILE"]
        vectorizer_file = cfg["TFIDF_VECTORIZER_FILE"]

        records = VetDB.get_all_vet_doc(db_name)
        if not records:
            print("⚠️  vet_doc is empty — nothing to index.")
            return 0

        corpus = [_search_text(r) for r in records]

        # char ngrams, because Thai has no spaces between words: a word-level
        # tokenizer would treat a whole phrase as one token and match almost nothing.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf_matrix = vectorizer.fit_transform(corpus)

        vectors = np.asarray(tfidf_matrix.todense(), dtype="float32")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        index_file.parent.mkdir(parents=True, exist_ok=True)
        vectorizer_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_file))
        joblib.dump(vectorizer, vectorizer_file)

        print(f"✅ Indexed {len(records)} records (dim={vectors.shape[1]}) → {index_file.name}")
        return len(records)

    @staticmethod
    def _load(db_name, index_file, vectorizer_file):
        if not index_file.exists() or not vectorizer_file.exists():
            print("⚠️  Index or vectorizer missing — building now...")
            VetFAISS.build_index(db_name)
        return faiss.read_index(str(index_file)), joblib.load(vectorizer_file)

    @staticmethod
    def search_vet_doc(query_text: str, top_k: int | None = None, max_distance: float | None = None):
        """Return the closest vet_doc records for a free-text symptom query."""
        cfg = ConfigManager.get_config_database()
        retrieval = ConfigManager.get_retrieval_config()
        top_k = top_k or retrieval["top_k"]
        max_distance = retrieval["max_distance"] if max_distance is None else max_distance

        if not query_text or not query_text.strip():
            return []

        index, vectorizer = VetFAISS._load(
            cfg["DB_NAME"], cfg["FAISS_INDEX_FILE"], cfg["TFIDF_VECTORIZER_FILE"]
        )
        records = VetDB.get_all_vet_doc(cfg["DB_NAME"])
        if not records or index.ntotal == 0:
            return []

        query_vec = np.asarray(
            vectorizer.transform([query_text]).todense(), dtype="float32"
        )
        distances, ids = index.search(query_vec, min(top_k, index.ntotal))

        results = []
        for distance, record_pos in zip(distances[0], ids[0]):
            if record_pos < 0 or record_pos >= len(records):
                continue
            if float(distance) > max_distance:
                continue
            rec = records[int(record_pos)]
            results.append(
                {
                    "data_id": rec["data_id"],
                    "disease": rec["symptoms"],
                    "symptoms": rec["cause"],
                    "cause": rec["diagnosis"],
                    "treatment": rec["treatment"],
                    "distance": round(float(distance), 4),
                }
            )
        return results


if __name__ == "__main__":
    count = VetFAISS.build_index()
    if count:
        query = "ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน มีสะเก็ดแห้งเป็นขุย"
        print(f"\n🔍 {query}")
        for hit in VetFAISS.search_vet_doc(query, top_k=3):
            print(f"  [{hit['distance']}] {hit['disease']}")
