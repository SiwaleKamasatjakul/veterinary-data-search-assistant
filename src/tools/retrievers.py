"""Retrievers: lexical (BM25), dense (vector), and hybrid fusion.

All three share one interface — ``search(query, top_k) -> [(data_id, score)]``,
best first, row-level (chunks already max-pooled).

Why BM25 on character n-grams
-----------------------------
Standard BM25 tokenises on whitespace. Thai has no spaces between words, so
whole phrases become single tokens and almost nothing matches. Tokenising into
character 3-grams gives BM25 the same sub-word granularity the TF-IDF path uses,
while keeping BM25's two real advantages over cosine TF-IDF: saturating term
frequency (the 20th occurrence of a term adds almost nothing) and explicit
document-length normalisation.

Why reciprocal rank fusion for hybrid
-------------------------------------
BM25 scores are unbounded; cosine lives in -1..1. Combining them by weighted sum
means normalising two distributions that drift apart as the corpus changes —
fragile, and it needs retuning every time you add records.

RRF ignores scores entirely and fuses **ranks**:

    score(d) = Σ  1 / (k + rank_i(d))

It has one constant (``k``, conventionally 60), needs no normalisation, and is
robust to either retriever producing a weird score scale. Cheaper to maintain
than a tuned alpha, which matters more than the last point of accuracy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from tools.chunking import Chunk

RRF_K = 60


def char_ngrams(text: str, n: int = 3) -> List[str]:
    t = f" {(text or '').strip()} "
    if len(t) < n:
        return [t]
    return [t[i : i + n] for i in range(len(t) - n + 1)]


# ---------------------------------------------------------------------------
class BM25Retriever:
    """Okapi BM25 over character n-grams, max-pooled to row level."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, ngram: int = 3) -> None:
        self.k1, self.b, self.ngram = k1, b, ngram
        self.chunks: List[Chunk] = []
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.lengths: List[int] = []
        self.avg_len = 0.0

    def fit(self, chunks: Sequence[Chunk]) -> "BM25Retriever":
        self.chunks = list(chunks)
        self.tf, self.lengths = [], []
        self.df = defaultdict(int)

        for chunk in self.chunks:
            grams = char_ngrams(chunk.text, self.ngram)
            counts: Dict[str, int] = defaultdict(int)
            for g in grams:
                counts[g] += 1
            self.tf.append(dict(counts))
            self.lengths.append(len(grams))
            for g in counts:
                self.df[g] += 1

        n = len(self.chunks)
        self.avg_len = sum(self.lengths) / max(n, 1)
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at 0 so a
        # gram present in every chunk cannot contribute negatively.
        self.idf = {
            g: max(math.log((n - d + 0.5) / (d + 0.5) + 1.0), 0.0) for g, d in self.df.items()
        }
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_grams = char_ngrams(query, self.ngram)
        if not q_grams or not self.chunks:
            return []

        q_counts: Dict[str, int] = defaultdict(int)
        for g in q_grams:
            q_counts[g] += 1

        best: Dict[int, float] = {}
        for i, chunk in enumerate(self.chunks):
            tf, dl = self.tf[i], self.lengths[i]
            score = 0.0
            for g, qn in q_counts.items():
                f = tf.get(g)
                if not f:
                    continue
                idf = self.idf.get(g, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1e-9))
                score += idf * (f * (self.k1 + 1) / denom) * qn
            if score > 0:
                # max-pool: a row scores at its single best chunk
                if score > best.get(chunk.data_id, float("-inf")):
                    best[chunk.data_id] = score

        return sorted(best.items(), key=lambda kv: -kv[1])[:top_k]


# ---------------------------------------------------------------------------
class DenseRetriever:
    """Any embedder from tools.embedders, over any chunking strategy."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.name = f"dense:{embedder.name}"
        self.chunks: List[Chunk] = []
        self.matrix: np.ndarray | None = None

    def fit(self, chunks: Sequence[Chunk]) -> "DenseRetriever":
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        if self.embedder.is_corpus_coupled:
            self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)  # rows are unit vectors
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.matrix is None or not self.chunks:
            return []
        q = self.embedder.encode([query])[0]
        sims = self.matrix @ q  # unit vectors -> cosine similarity

        best: Dict[int, float] = {}
        for chunk, sim in zip(self.chunks, sims):
            s = float(sim)
            if s > best.get(chunk.data_id, float("-inf")):
                best[chunk.data_id] = s
        return sorted(best.items(), key=lambda kv: -kv[1])[:top_k]


# ---------------------------------------------------------------------------
class HybridRetriever:
    """Reciprocal rank fusion of any number of retrievers."""

    def __init__(self, retrievers: Sequence, k: int = RRF_K, weights: Sequence[float] | None = None):
        self.retrievers = list(retrievers)
        self.k = k
        self.weights = list(weights) if weights else [1.0] * len(self.retrievers)
        self.name = "hybrid(" + "+".join(r.name for r in self.retrievers) + ")"

    def fit(self, chunks: Sequence[Chunk]) -> "HybridRetriever":
        for r in self.retrievers:
            r.fit(chunks)
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        # Pull deeper than top_k from each so fusion has material to work with.
        depth = max(top_k * 4, 20)
        fused: Dict[int, float] = defaultdict(float)
        for retriever, weight in zip(self.retrievers, self.weights):
            for rank, (data_id, _score) in enumerate(retriever.search(query, depth), start=1):
                fused[data_id] += weight / (self.k + rank)
        return sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
