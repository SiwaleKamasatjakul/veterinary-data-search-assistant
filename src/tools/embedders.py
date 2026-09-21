"""Pluggable embedding backends.

Two implementations behind one interface, chosen by ``retrieval.backend`` in
config/llm_config.json:

    "tfidf"   TfidfEmbedder   — local, free, no network. Matches on characters.
    "openai"  OpenAIEmbedder  — text-embedding-3-small. Matches on meaning.

Both return **L2-normalised float32** vectors. That single invariant is what lets
the rest of the codebase stay backend-agnostic: with unit vectors, the FAISS
inner product IS cosine similarity, so one score scale (-1..1, higher is better)
covers both backends.

The key behavioural difference is not quality, it is *coupling*:

* TF-IDF vectors are defined relative to the corpus. Adding one record changes
  the vocabulary and the IDF weights, so every vector must be recomputed.
  ``is_corpus_coupled = True``.
* OpenAI vectors are independent. Each text embeds to the same 1536 numbers
  regardless of what else is in the index, so new records can be appended and
  unchanged records can be served from cache. ``is_corpus_coupled = False``.

``ImportDB2Faiss`` reads that flag to decide whether a rebuild may reuse cached
vectors or must start from scratch.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Dimensions of the OpenAI models we support, so we can size the index without
# a network round trip.
OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so inner product == cosine similarity."""
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class BaseEmbedder(ABC):
    name: str
    dim: int
    is_corpus_coupled: bool

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> None:
        """Prepare the embedder for this corpus. No-op for API embedders."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) matrix of unit vectors."""

    @abstractmethod
    def save(self, path) -> None: ...

    @abstractmethod
    def load(self, path) -> None: ...

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])


# ---------------------------------------------------------------------------
class TfidfEmbedder(BaseEmbedder):
    """Character n-gram TF-IDF. Local, free, Thai-friendly.

    ``char_wb`` with 2–4 grams because Thai has no spaces between words, so a
    word-level tokenizer would treat a whole phrase as one token.
    """

    is_corpus_coupled = True

    def __init__(self, ngram_range=(2, 4)) -> None:
        self.name = f"tfidf-char_wb-{ngram_range[0]}_{ngram_range[1]}"
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.dim = 0

    def fit(self, corpus: Sequence[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range)
        self.vectorizer.fit(list(corpus))
        self.dim = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF fitted: %d records -> %d dims", len(corpus), self.dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfidfEmbedder.fit() must run before encode()")
        # sklearn already L2-normalises rows, but normalise again so the
        # invariant holds regardless of vectorizer settings.
        return l2_normalize(self.vectorizer.transform(list(texts)).toarray())

    def save(self, path) -> None:
        import joblib

        joblib.dump({"vectorizer": self.vectorizer, "ngram_range": self.ngram_range}, path)

    def load(self, path) -> None:
        import joblib

        blob = joblib.load(path)
        self.vectorizer = blob["vectorizer"]
        self.ngram_range = blob.get("ngram_range", (2, 4))
        self.dim = len(self.vectorizer.vocabulary_)


# ---------------------------------------------------------------------------
class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings. Matches meaning, not characters.

    Catches the paraphrase case TF-IDF cannot: a user writing ``ฉี่`` when the
    record says ``ปัสสาวะ`` shares no characters, but the two are close in
    embedding space.

    Costs money per call, so ``ImportDB2Faiss`` wraps this in a cache — see
    ``embedding_cache.py``. Only text that has actually changed is re-sent.
    """

    is_corpus_coupled = False

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128) -> None:
        self.name = model
        self.model = model
        self.batch_size = batch_size
        self.dim = OPENAI_DIMS.get(model, 1536)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from tools.paths import ENV_FILE, load_env

            # Defensive: paths.py loads .env on import, but this makes the
            # dependency explicit for anyone importing embedders directly.
            load_env()

            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                where = (
                    f"{ENV_FILE} exists but has no non-empty OPENAI_API_KEY"
                    if ENV_FILE.exists()
                    else f"no .env file at {ENV_FILE}"
                )
                raise RuntimeError(
                    "retrieval.backend is 'openai' but OPENAI_API_KEY is not set.\n"
                    f"  checked: {where}\n"
                    "  Fix: put OPENAI_API_KEY=sk-... in that .env (no quotes, no "
                    "spaces around '='),\n"
                    "  or switch retrieval.backend to 'tfidf' in config/llm_config.json."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        return self._client

    def fit(self, corpus: Sequence[str]) -> None:
        """No-op. Each text embeds independently of the corpus."""
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]  # API rejects empty strings
        vectors: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            # The API may return items out of order; index is authoritative.
            ordered = sorted(response.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in ordered)
            logger.info("Embedded %d/%d texts", min(start + len(batch), len(texts)), len(texts))

        matrix = np.asarray(vectors, dtype="float32")
        self.dim = matrix.shape[1]
        return l2_normalize(matrix)

    def save(self, path) -> None:
        """Nothing to persist — the model is remote. Written for symmetry."""
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps({"model": self.model, "dim": self.dim}), encoding="utf-8")

    def load(self, path) -> None:
        import json
        from pathlib import Path

        p = Path(path)
        if p.exists():
            blob = json.loads(p.read_text(encoding="utf-8"))
            self.model = blob.get("model", self.model)
            self.name = self.model
            self.dim = blob.get("dim", self.dim)


# ---------------------------------------------------------------------------
def make_embedder(backend: str, model: str = "text-embedding-3-small") -> BaseEmbedder:
    backend = (backend or "tfidf").lower()
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "openai":
        return OpenAIEmbedder(model=model)
    raise ValueError(f"Unknown retrieval.backend {backend!r}. Use 'tfidf' or 'openai'.")
