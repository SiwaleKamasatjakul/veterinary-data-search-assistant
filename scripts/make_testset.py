#!/usr/bin/env python3
"""Generate a labelled retrieval testset from vet_doc.

    python scripts/make_testset.py                    # -> data/testset.csv
    python scripts/make_testset.py --per-record 4
    python scripts/make_testset.py --out data/other.csv

Why generate instead of hand-writing
------------------------------------
18 hand-written cases tell you *whether* retrieval works. They do not tell you
*why* it fails, because every case mixes several difficulties at once. Generated
cases carry a ``category`` label, so accuracy can be broken down:

    exact       the record's own symptom text, verbatim
    single      ONE symptom phrase from the record
    partial     two or three phrases, order shuffled
    colloquial  a phrase with vet vocabulary swapped for owner vocabulary
    typo        a phrase with character-level noise
    negative    off-topic; nothing should be returned

That breakdown separates two very different diagnoses:

* **exact/single failing** → a DATA problem. The record cannot even find itself,
  or two records are too similar to tell apart. No retriever fixes that.
* **exact passing but colloquial failing** → an ALGORITHM problem. The record is
  fine; character matching cannot bridge the vocabulary gap. Semantic embeddings
  or hybrid search is the answer.

The generated set is deterministic (fixed seed) so runs are comparable.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from tools.chunking import split_symptom_phrases  # noqa: E402
from tools.config_loader import ConfigManager  # noqa: E402
from tools.ImportDB2Faiss import VetDB  # noqa: E402

# Owner vocabulary -> vet vocabulary. These are the pairs that break character
# matching: same meaning, no shared characters.
COLLOQUIAL = {
    "ปัสสาวะ": ["ฉี่", "เยี่ยว"],
    "อุจจาระ": ["อึ", "ขี้"],
    "อาเจียน": ["อ้วก", "ขย้อน"],
    "เบื่ออาหาร": ["ไม่กินข้าว", "ไม่ยอมกิน"],
    "น้ำหนักลด": ["ผอมลง", "ซูบลง"],
    "ซึม": ["ไม่ร่าเริง", "นอนทั้งวัน"],
    "หายใจลำบาก": ["หอบ", "หายใจแรง"],
    "ขนร่วง": ["ขนหลุด", "ขนบาง"],
    "คัน": ["เกาบ่อย", "เกาไม่หยุด"],
    "ท้องเสีย": ["ถ่ายเหลว", "ท้องร่วง"],
    "เหงือก": ["ขอบฟัน"],
    "ผิวหนัง": ["หนัง"],
}

PREFIXES = ["แมว", "แมวของฉัน", "น้องแมว", "แมวที่บ้าน", ""]
SUFFIXES = ["", " เป็นอะไร", " ควรทำยังไง", " อันตรายไหม", " หลายวันแล้ว"]

NEGATIVES = [
    "วันนี้อากาศดีมาก",
    "ราคาอาหารแมวยี่ห้อนี้เท่าไหร่",
    "ช่วยแนะนำร้านกาแฟแถวอโศก",
    "รถยนต์ไฟฟ้ารุ่นไหนดี",
    "สอนวิธีเขียนโปรแกรม python",
    "แมวชอบนอนตรงไหนมากที่สุด",
    "พรุ่งนี้ฝนจะตกไหม",
    "อยากเปลี่ยนงานใหม่",
    "ทรายแมวยี่ห้อไหนดีที่สุด",
    "ขอสูตรทำขนมเค้ก",
]


def add_typos(text: str, rng: random.Random, rate: float = 0.08) -> str:
    """Character-level noise: drop, duplicate or swap."""
    chars = list(text)
    n = max(1, int(len(chars) * rate))
    for _ in range(n):
        if len(chars) < 3:
            break
        i = rng.randrange(len(chars) - 1)
        op = rng.choice(["drop", "dup", "swap"])
        if op == "drop":
            chars.pop(i)
        elif op == "dup":
            chars.insert(i, chars[i])
        else:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def colloquialise(text: str, rng: random.Random) -> tuple[str, bool]:
    """Swap vet vocabulary for owner vocabulary. Returns (text, did_change)."""
    changed = False
    for formal, casual in COLLOQUIAL.items():
        if formal in text:
            text = text.replace(formal, rng.choice(casual))
            changed = True
    return text, changed


def decorate(text: str, rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)}{text}{rng.choice(SUFFIXES)}".strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "testset.csv")
    parser.add_argument("--per-record", type=int, default=3, help="single-symptom cases per record")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cfg = ConfigManager.get_config_database()
    rows = VetDB.get_all_vet_doc(cfg["DB_NAME"])
    if not rows:
        print("❌ vet_doc is empty.")
        return 1

    cases = []

    def add(query, data_id, category, note=""):
        q = (query or "").strip()
        if q:
            cases.append(
                {
                    "query": q,
                    "expected_data_id": data_id,
                    "category": category,
                    "note": note,
                }
            )

    for row in rows:
        rid = row["data_id"]
        disease = row["symptoms"]
        symptoms = row["cause"] or ""
        phrases = split_symptom_phrases(symptoms)

        # exact — the record's own text
        add(symptoms, rid, "exact", disease)

        # single — one phrase at a time
        for phrase in rng.sample(phrases, min(args.per_record, len(phrases))):
            add(decorate(phrase, rng), rid, "single", phrase)

        # partial — a few phrases, shuffled
        if len(phrases) >= 3:
            picked = rng.sample(phrases, 3)
            rng.shuffle(picked)
            add(decorate(" ".join(picked), rng), rid, "partial")

        # colloquial — owner vocabulary
        for phrase in phrases:
            casual, changed = colloquialise(phrase, rng)
            if changed:
                add(decorate(casual, rng), rid, "colloquial", f"from: {phrase}")

        # typo — character noise on one phrase
        if phrases:
            add(decorate(add_typos(rng.choice(phrases), rng), rng), rid, "typo")

    for neg in NEGATIVES:
        add(neg, -1, "negative", "nothing should be retrieved")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "expected_data_id", "category", "note"])
        writer.writeheader()
        writer.writerows(cases)

    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1

    print(f"✅ {len(cases)} cases from {len(rows)} records → {args.out.relative_to(BASE_DIR)}\n")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"   {cat:<12} {n:>4}")
    print("\nNext:  python scripts/benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
