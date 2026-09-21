#!/usr/bin/env python3
"""Add veterinary records to vet_doc, then rebuild the search index.

Unlike ``build_index.py --reseed`` (which wipes the table and reloads the seed
file), this script ADDS to what is already there. Use it to grow the knowledge
base over time.

Four input modes
----------------
    python scripts/add_vet_doc.py --json data/new_records.json
    python scripts/add_vet_doc.py --csv  data/new_records.csv
    python scripts/add_vet_doc.py --interactive
    python scripts/add_vet_doc.py --disease "โรคX" --symptoms "..." --cause "..." --treatment "..."

Useful flags
------------
    --dry-run       show what would be added, change nothing
    --no-rebuild    skip the index rebuild (you MUST rebuild before searching)
    --allow-dup     add even if the disease name already exists
    --template      write a CSV template you can hand to a vet, then exit

IMPORTANT — column naming
-------------------------
The vet_doc table's column names are shifted (see ARCHITECTURE.md §6.2):

    DB column    actually holds        this script's flag
    ---------    ------------------    ------------------
    symptoms  -> the disease name      --disease
    cause     -> the symptom list      --symptoms
    diagnosis -> the cause             --cause
    treatment -> the treatment         --treatment

This script speaks in honest names and maps to the real columns for you, so you
do not have to remember the shift. JSON and CSV input use the honest names too.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from tools.config_loader import ConfigManager  # noqa: E402
from tools.ImportDB2Faiss import VetDB, VetFAISS  # noqa: E402

# honest name -> real DB column
FIELD_MAP = {
    "disease": "symptoms",
    "symptoms": "cause",
    "cause": "diagnosis",
    "treatment": "treatment",
}
REQUIRED = ("disease", "symptoms")

CSV_TEMPLATE = """disease,symptoms,cause,treatment
โรคหอบหืดในแมว (Feline Asthma),"ไอเรื้อรัง หายใจมีเสียงวี้ด หอบ นั่งยืดคอหายใจ","ภูมิแพ้ต่อสารก่อภูมิแพ้ในอากาศ เช่น ฝุ่น ควัน เกสร","ให้ยาขยายหลอดลมและสเตียรอยด์ตามสัตวแพทย์สั่ง ลดสารก่อภูมิแพ้ในบ้าน"
"""


# ---------------------------------------------------------------------------
# Input readers — all return a list of dicts using the honest field names
# ---------------------------------------------------------------------------
def read_json(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return [_normalize(r, str(path)) for r in data]


def read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig: strips Excel BOM
        return [_normalize(row, str(path)) for row in csv.DictReader(f)]


def read_interactive() -> List[dict]:
    print("Enter records. Blank disease name to finish.\n")
    records = []
    while True:
        disease = input("Disease name (e.g. โรคหวัดแมว): ").strip()
        if not disease:
            break
        record = {
            "disease": disease,
            "symptoms": input("  Symptoms (what the owner would describe): ").strip(),
            "cause": input("  Cause / pathogen: ").strip(),
            "treatment": input("  Treatment: ").strip(),
        }
        records.append(_normalize(record, "interactive"))
        print(f"  ✓ queued: {disease}\n")
    return records


def _normalize(row: dict, source: str) -> dict:
    """Accept honest names or raw DB column names; validate; strip whitespace."""
    record = {}
    for honest in FIELD_MAP:
        value = row.get(honest)
        if value is None:
            # tolerate files written with the raw (shifted) column names
            value = row.get(FIELD_MAP[honest])
        record[honest] = (value or "").strip()

    missing = [f for f in REQUIRED if not record[f]]
    if missing:
        raise SystemExit(
            f"❌ {source}: record is missing required field(s) {missing}\n"
            f"   got: {json.dumps(row, ensure_ascii=False)}"
        )
    return record


# ---------------------------------------------------------------------------
def existing_diseases(db_name) -> set:
    return {r["symptoms"].strip() for r in VetDB.get_all_vet_doc(db_name)}


def insert(db_name, records: List[dict]) -> int:
    rows = [{FIELD_MAP[k]: v for k, v in r.items()} for r in records]
    with sqlite3.connect(db_name) as conn:
        conn.executemany(
            """INSERT INTO vet_doc (symptoms, cause, diagnosis, treatment)
               VALUES (:symptoms, :cause, :diagnosis, :treatment)""",
            rows,
        )
    return len(rows)


def sync_seed_file(records: List[dict], seed_path: Path) -> None:
    """Mirror the additions into seed_vet_doc.json so a --reseed keeps them."""
    if not seed_path.exists():
        return
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    seed.extend({FIELD_MAP[k]: v for k, v in r.items()} for r in records)
    seed_path.write_text(
        json.dumps(seed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add records to vet_doc and rebuild the search index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json", type=Path, help="JSON file (array of records)")
    source.add_argument("--csv", type=Path, help="CSV file with a header row")
    source.add_argument("--interactive", action="store_true", help="type records in")
    source.add_argument("--disease", help="add one record from the command line")

    parser.add_argument("--symptoms", help="with --disease")
    parser.add_argument("--cause", default="", help="with --disease")
    parser.add_argument("--treatment", default="", help="with --disease")

    parser.add_argument("--dry-run", action="store_true", help="preview only")
    parser.add_argument("--no-rebuild", action="store_true", help="skip index rebuild")
    parser.add_argument("--allow-dup", action="store_true", help="skip the duplicate check")
    parser.add_argument("--template", action="store_true", help="write a CSV template and exit")
    args = parser.parse_args()

    if args.template:
        out = BASE_DIR / "data" / "new_records_template.csv"
        out.write_text(CSV_TEMPLATE, encoding="utf-8")
        print(f"📄 Template written to {out.relative_to(BASE_DIR)}")
        print("   Fill it in (Excel or Numbers), then:")
        print(f"   python scripts/add_vet_doc.py --csv {out.relative_to(BASE_DIR)}")
        return 0

    # -- gather input ------------------------------------------------------
    if args.json:
        records = read_json(args.json)
    elif args.csv:
        records = read_csv(args.csv)
    elif args.interactive:
        records = read_interactive()
    elif args.disease:
        if not args.symptoms:
            parser.error("--disease also requires --symptoms")
        records = [
            _normalize(
                {
                    "disease": args.disease,
                    "symptoms": args.symptoms,
                    "cause": args.cause,
                    "treatment": args.treatment,
                },
                "command line",
            )
        ]
    else:
        parser.error("pick one of --json / --csv / --interactive / --disease (or --template)")

    if not records:
        print("Nothing to add.")
        return 0

    # -- duplicate check ---------------------------------------------------
    cfg = ConfigManager.get_config_database()
    db_name = cfg["DB_NAME"]
    VetDB.create_tables(db_name)

    if not args.allow_dup:
        known = existing_diseases(db_name)
        seen, kept, skipped = set(), [], []
        for r in records:
            if r["disease"] in known or r["disease"] in seen:
                skipped.append(r["disease"])
            else:
                seen.add(r["disease"])
                kept.append(r)
        for name in skipped:
            print(f"⏭️  skipped (already present): {name}")
        records = kept

    if not records:
        print("\nAll records were duplicates — nothing added.")
        return 0

    # -- preview -----------------------------------------------------------
    print(f"\n📋 {len(records)} record(s) to add:\n")
    for r in records:
        print(f"  • {r['disease']}")
        print(f"      symptoms : {r['symptoms'][:70]}{'…' if len(r['symptoms']) > 70 else ''}")
        if not r["symptoms"]:
            print("      ⚠️  empty symptoms — this record will never be retrieved")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    # -- write -------------------------------------------------------------
    added = insert(db_name, records)
    sync_seed_file(records, BASE_DIR / "data" / "seed_vet_doc.json")
    total = len(VetDB.get_all_vet_doc(db_name))
    print(f"\n✅ Added {added} record(s). vet_doc now has {total}.")
    print("   (also appended to data/seed_vet_doc.json)")

    # -- rebuild -----------------------------------------------------------
    if args.no_rebuild:
        print("\n⚠️  Index NOT rebuilt. New records are invisible to search until you run:")
        print("     python scripts/build_index.py")
        return 0

    print("\n🔄 Rebuilding index...")
    VetFAISS.build_index(db_name)

    # -- verify ------------------------------------------------------------
    print("\n🔍 Verifying each new record is retrievable by its own symptom text:")
    failures = 0
    for r in records:
        hits = VetFAISS.search_vet_doc(r["symptoms"], top_k=1, min_similarity=-1)
        if hits and hits[0]["disease"] == r["disease"]:
            print(f"   ✓ {r['disease']}  (similarity {hits[0]['score']})")
        else:
            got = hits[0]["disease"] if hits else "nothing"
            print(f"   ✗ {r['disease']} → retrieved {got}")
            failures += 1

    if failures:
        print(f"\n⚠️  {failures} record(s) did not retrieve themselves. Usually means the")
        print("   symptom text is too short, or too similar to an existing record.")
        return 1

    print("\nDone. Restart the API to pick up the new index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
