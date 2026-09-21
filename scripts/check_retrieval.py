#!/usr/bin/env python3
"""Retrieval regression check — turns "search feels right" into a number.

Reads data/retrieval_tests.csv (query -> expected disease) and asserts the
expected record comes back as the top hit. Run it after every data change, and
after switching backends.

    python scripts/check_retrieval.py                 # active backend
    python scripts/check_retrieval.py --verbose
    python scripts/check_retrieval.py --backend openai   # override for one run
    python scripts/check_retrieval.py --tune             # suggest min_similarity

Exit code 0 when everything passes, 1 otherwise — drops straight into CI.

SCORES ARE COSINE SIMILARITY, -1..1, HIGHER IS BETTER.
(The earlier version used L2 distance where lower was better. If you have notes
saying "lower is better", they predate the embedder refactor.)

Why this exists
---------------
With the tfidf backend, adding a record changes the vocabulary and IDF weights,
which re-scores EVERY existing record — a new document can quietly steal the top
spot from a correct one. That has already happened once here: adding the
Constipation record broke the FLUTD query, the most safety-critical query in the
set. With the openai backend vectors are independent, so that particular failure
mode disappears, but a near-duplicate record can still outrank the right one.

The margin column is what to watch. A pass with a thin margin is fragile.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

DEFAULT_CSV = BASE_DIR / "data" / "retrieval_tests.csv"
FRAGILE_MARGIN = 0.05

# Queries with nothing relevant in the knowledge base. The best score any of
# these gets is the noise floor — min_similarity must sit above it.
OFF_TOPIC = [
    "วันนี้อากาศดีมาก",
    "ราคาอาหารแมวยี่ห้อนี้เท่าไหร่",
    "ช่วยแนะนำร้านกาแฟแถวอโศก",
    "รถยนต์ไฟฟ้ารุ่นไหนดี",
    "สอนวิธีเขียนโปรแกรม python",
    "แมวชอบนอนตรงไหนมากที่สุด",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--backend", choices=["tfidf", "openai"], help="override for this run")
    parser.add_argument("--tune", action="store_true", help="suggest a min_similarity value")
    args = parser.parse_args()

    if args.backend:
        os.environ["RETRIEVAL_BACKEND"] = args.backend

    from tools.config_loader import ConfigManager  # noqa: E402  (after env override)
    from tools.ImportDB2Faiss import VetFAISS  # noqa: E402

    if not args.csv.exists():
        print(f"❌ No test file at {args.csv}")
        return 1

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        cases = [r for r in csv.DictReader(f) if (r.get("query") or "").strip()]
    if not cases:
        print("❌ No test cases found.")
        return 1

    retrieval = ConfigManager.get_retrieval_config()
    backend = retrieval["backend"]
    min_similarity = retrieval["min_similarity"]

    passed, failed, fragile, filtered, correct_scores = 0, [], [], [], []

    for case in cases:
        query = case["query"].strip()
        expected = case["expected_disease_contains"].strip()
        note = (case.get("note") or "").strip()

        hits = VetFAISS.search_vet_doc(query, top_k=2, min_similarity=-1)
        if not hits:
            failed.append((query, expected, "no results at all", note))
            continue

        top = hits[0]
        margin = round(top["score"] - hits[1]["score"], 4) if len(hits) > 1 else 99.0
        ok = expected.lower() in top["disease"].lower()

        if ok:
            passed += 1
            correct_scores.append(top["score"])
            if margin < FRAGILE_MARGIN:
                fragile.append((query, top["disease"], margin))
            if top["score"] < min_similarity:
                filtered.append((query, top["disease"], top["score"]))
            if args.verbose:
                print(f"✓ {query}\n    {top['disease']}  score={top['score']}  margin={margin}")
        else:
            failed.append((query, expected, f"got {top['disease']} (score={top['score']})", note))
            if args.verbose:
                print(f"✗ {query}\n    expected ~{expected}, got {top['disease']}")

    # -- noise floor -------------------------------------------------------
    noise = []
    for query in OFF_TOPIC:
        hits = VetFAISS.search_vet_doc(query, top_k=1, min_similarity=-1)
        if hits:
            noise.append((hits[0]["score"], query, hits[0]["disease"]))

    # -- report ------------------------------------------------------------
    total = len(cases)
    info = VetFAISS.index_info()
    print(f"\n{'='*72}")
    print(f"backend: {backend}  ({info.get('embedder', '?')}, dim={info.get('dim', '?')})")
    print(f"Top-1 accuracy: {passed}/{total}  ({passed/total*100:.0f}%)")
    print(f"min_similarity in config: {min_similarity}")

    if correct_scores:
        print(
            f"\ncorrect-hit scores : min {min(correct_scores):.4f}  "
            f"max {max(correct_scores):.4f}"
        )
    if noise:
        best_noise = max(n[0] for n in noise)
        print(f"off-topic noise    : max {best_noise:.4f}  ({noise[0][1] if noise else ''})")

    if failed:
        print(f"\n❌ {len(failed)} FAILED:")
        for query, expected, got, note in failed:
            print(f"   {query}")
            print(f"      expected ~{expected} | {got}")
            if note:
                print(f"      note: {note}")

    if filtered:
        print(f"\n⚠️  {len(filtered)} correct hit(s) FILTERED OUT by min_similarity={min_similarity}:")
        for query, disease, score in filtered:
            print(f"   {query}  →  {disease} at score={score}")
        print("   Lower retrieval.min_similarity, or improve the record text.")

    if fragile:
        print(f"\n⚠️  {len(fragile)} thin margin(s) (<{FRAGILE_MARGIN}) — likely to flip:")
        for query, disease, margin in fragile:
            print(f"   {query}  →  {disease}  (margin {margin})")
        print("   Add distinguishing words to the record's symptom text.")

    # -- threshold suggestion ---------------------------------------------
    if args.tune and correct_scores and noise:
        lowest_correct = min(correct_scores)
        highest_noise = max(n[0] for n in noise)
        print(f"\n{'-'*72}\nTHRESHOLD TUNING ({backend})")
        print(f"  lowest correct hit : {lowest_correct:.4f}")
        print(f"  highest off-topic  : {highest_noise:.4f}")
        if lowest_correct > highest_noise:
            suggested = round((lowest_correct + highest_noise) / 2, 2)
            print(f"  ✅ clean separation — suggested min_similarity: {suggested}")
        else:
            print("  ⚠️  OVERLAP: some off-topic queries score above a correct hit.")
            print("     No threshold separates them cleanly. Either improve the weak")
            print("     record's text, or accept noise and let the prompt handle it.")
            print(f"     Least-bad value: {round(highest_noise + 0.01, 2)} (drops some correct hits)")

    if not failed and not filtered:
        print("\n✅ All good.")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
