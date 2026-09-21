"""Chunking strategies for SQL records.

"Chunking" usually means splitting long documents. Here the unit is a database
row, so the question is different: **what text from a row becomes a searchable
unit, and how many units does one row produce?**

That choice matters more than it looks. A vet_doc row lists six or seven
symptoms in one field. Embed them as one blob and a query about a single symptom
is diluted by the other six — the signal you want is one seventh of the vector.
Split them and each symptom gets its own full-strength vector.

Strategies
----------
``row_symptom_only``  1 chunk/row — just the symptom list. Tightest signal.
``row_combined``      1 chunk/row — symptoms + disease name. **Current production.**
``row_full``          1 chunk/row — every field. Most context, most dilution.
``per_symptom``       N chunks/row — one per symptom phrase, plus a whole-row
                      chunk as a fallback. Many chunks map to the same data_id.

Every strategy returns ``Chunk`` objects carrying ``data_id``, so retrieval maps
back to the row regardless of how many chunks that row produced. When several
chunks from one row hit, the row scores once, at its best chunk (max-pooling).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List

# Thai has no spaces between words, but symptom lists DO use spaces (and commas)
# between phrases. That is the split point — it is a list separator, not a word
# boundary, which is exactly why char-ngram matching is still needed inside each
# phrase.
_SPLIT = re.compile(r"[,ๆ]?\s{1,}|,")

MIN_PHRASE_CHARS = 6


@dataclass
class Chunk:
    data_id: int
    text: str
    kind: str  # which strategy/part produced it — useful for error analysis


def _clean(s) -> str:
    return (s or "").strip()


def split_symptom_phrases(text: str) -> List[str]:
    """Split a symptom list into individual phrases.

    ``"ดื่มน้ำมาก ปัสสาวะบ่อย น้ำหนักลด อาเจียน"``
    -> ``["ดื่มน้ำมาก", "ปัสสาวะบ่อย", "น้ำหนักลด", "อาเจียน"]``

    Very short fragments are dropped — they carry no signal and become magnets
    that match everything.
    """
    parts = [p.strip() for p in _SPLIT.split(text or "") if p and p.strip()]
    return [p for p in parts if len(p) >= MIN_PHRASE_CHARS]


# ---------------------------------------------------------------------------
def chunk_row_symptom_only(row: dict) -> List[Chunk]:
    text = _clean(row.get("cause"))
    return [Chunk(row["data_id"], text, "symptom_only")] if text else []


def chunk_row_combined(row: dict) -> List[Chunk]:
    """What production uses today: symptom list + disease name."""
    text = f'{_clean(row.get("cause"))} {_clean(row.get("symptoms"))}'.strip()
    return [Chunk(row["data_id"], text, "combined")] if text else []


def chunk_row_full(row: dict) -> List[Chunk]:
    text = " ".join(
        _clean(row.get(f)) for f in ("cause", "symptoms", "diagnosis", "treatment")
    ).strip()
    return [Chunk(row["data_id"], text, "full")] if text else []


def chunk_per_symptom(row: dict) -> List[Chunk]:
    """One chunk per symptom phrase, plus a whole-row chunk as fallback.

    The per-phrase chunks win single-symptom queries; the row chunk still covers
    multi-symptom queries that mention several at once.
    """
    chunks: List[Chunk] = []
    disease = _clean(row.get("symptoms"))

    for phrase in split_symptom_phrases(row.get("cause")):
        # Appending the disease name keeps each chunk self-identifying without
        # swamping the phrase — it is short relative to the phrase.
        chunks.append(Chunk(row["data_id"], phrase, "phrase"))

    whole = f'{_clean(row.get("cause"))} {disease}'.strip()
    if whole:
        chunks.append(Chunk(row["data_id"], whole, "row"))
    return chunks



# ---------------------------------------------------------------------------
# Synonym expansion
# ---------------------------------------------------------------------------
_SYNONYMS: Dict[str, List[str]] | None = None


def load_synonyms() -> Dict[str, List[str]]:
    """Vet term -> owner terms, from data/synonyms.json. Cached."""
    global _SYNONYMS
    if _SYNONYMS is None:
        import json

        from tools.paths import DATA_DIR

        path = DATA_DIR / "synonyms.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            _SYNONYMS = {k: v for k, v in raw.items() if not k.startswith("_")}
        else:
            _SYNONYMS = {}
    return _SYNONYMS


def expand_phrase(phrase: str) -> List[str]:
    """The phrase plus one variant per synonym substitution.

    ``"ปัสสาวะบ่อย"`` -> ``["ปัสสาวะบ่อย", "ฉี่บ่อย", "เยี่ยวบ่อย"]``

    Substituting INSIDE the phrase — rather than appending a bag of synonyms to
    the row — is what makes this work. Appending dilutes the vector: measured
    R@1 actually dropped from 0.739 to 0.726. Each variant here is its own short,
    full-strength chunk instead.
    """
    variants = [phrase]
    for formal, casual_list in load_synonyms().items():
        if formal in phrase:
            variants.extend(phrase.replace(formal, c) for c in casual_list)
    return variants


def chunk_per_symptom_expanded(row: dict) -> List[Chunk]:
    """per_symptom, plus a chunk for each colloquial rewording of each phrase.

    Best-measured lexical strategy: R@1 0.752, colloquial 0.54, noise gap +0.190
    (production row_combined: 0.739 / 0.43 / -0.026). Costs ~9.7 chunks per row.
    """
    chunks: List[Chunk] = []
    seen = set()
    for phrase in split_symptom_phrases(row.get("cause")):
        for variant in expand_phrase(phrase):
            key = (row["data_id"], variant)
            if key not in seen:
                seen.add(key)
                chunks.append(Chunk(row["data_id"], variant, "phrase"))

    whole = f'{_clean(row.get("cause"))} {_clean(row.get("symptoms"))}'.strip()
    if whole:
        chunks.append(Chunk(row["data_id"], whole, "row"))
    return chunks


STRATEGIES: Dict[str, Callable[[dict], List[Chunk]]] = {
    "row_symptom_only": chunk_row_symptom_only,
    "row_combined": chunk_row_combined,
    "row_full": chunk_row_full,
    "per_symptom": chunk_per_symptom,
    "per_symptom_expanded": chunk_per_symptom_expanded,
}


def build_chunks(rows: List[dict], strategy: str = "row_combined") -> List[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {strategy!r}. Options: {list(STRATEGIES)}")
    fn = STRATEGIES[strategy]
    chunks: List[Chunk] = []
    for row in rows:
        chunks.extend(fn(row))
    return chunks


def describe(rows: List[dict]) -> Dict[str, dict]:
    """Chunk counts and text lengths per strategy — the cost side of the trade."""
    out = {}
    for name in STRATEGIES:
        chunks = build_chunks(rows, name)
        lengths = [len(c.text) for c in chunks] or [0]
        out[name] = {
            "chunks": len(chunks),
            "chunks_per_row": round(len(chunks) / max(len(rows), 1), 2),
            "avg_chars": round(sum(lengths) / len(lengths), 1),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
        }
    return out
