#!/usr/bin/env python3
"""Seed vet_doc (if empty) and rebuild the FAISS index + TF-IDF vectorizer.

    python scripts/build_index.py            # build, seeding only if the table is empty
    python scripts/build_index.py --reseed    # wipe vet_doc and reload from seed JSON

Run this after editing data/seed_vet_doc.json or adding rows to vet_doc.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from tools.config_loader import ConfigManager  # noqa: E402
from tools.ImportDB2Faiss import VetDB, VetFAISS  # noqa: E402

SEED_FILE = BASE_DIR / "data" / "seed_vet_doc.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reseed", action="store_true", help="delete all rows, reload seed JSON")
    args = parser.parse_args()

    cfg = ConfigManager.get_config_database()
    db_name = cfg["DB_NAME"]
    VetDB.create_tables(db_name)

    if args.reseed:
        with sqlite3.connect(db_name) as conn:
            conn.execute("DELETE FROM vet_doc")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='vet_doc'")
        print("🗑️  Cleared vet_doc")

    if not VetDB.get_all_vet_doc(db_name):
        if not SEED_FILE.exists():
            print(f"❌ No rows in vet_doc and no seed file at {SEED_FILE}")
            return 1
        records = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        print(f"🌱 Seeded {VetDB.insert_many(db_name, records)} records from {SEED_FILE.name}")

    count = VetFAISS.build_index(db_name)

    query = "แมวขนร่วงเป็นวง ๆ ผิวหนังแดง คัน"
    print(f"\n🔍 Smoke test: {query}")
    for hit in VetFAISS.search_vet_doc(query, top_k=3):
        print(f"   [{hit['distance']}] {hit['disease']}")

    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
