#!/usr/bin/env python3
"""Benchmark retrieval strategies, and diagnose whether the DATA is the problem.

    python scripts/benchmark.py                     # full matrix, tfidf only
    python scripts/benchmark.py --with-openai       # include OpenAI embeddings ($)
    python scripts/benchmark.py --data-health       # diagnostics only, no matrix
    python scripts/benchmark.py --no-save           # print only, write nothing

Every run saves to benchmark_results/ automatically:

    benchmark_results/
      latest_report.html      <- open this in a browser; colour-coded table
      latest_summary.txt      <- the console output
      latest_results.csv      <- the matrix, for Excel
      latest_failures.csv     <- every wrong answer, with query and what it returned
      run_20260918_065248/    <- timestamped copy of all four, kept per run

Runs entirely in memory. It never touches the production index, so it is safe to
run against a live project.

Metrics
-------
Recall@1   correct row is the top hit          — what the user experiences at top_k=1
Recall@3   correct row is in the top 3         — headroom a reranker could recover
MRR        1/rank of the correct row, averaged — sensitive to near-misses
noise_gap  (lowest correct score) − (best off-topic score). **Positive means a
           threshold exists that keeps every correct hit and rejects every
           off-topic query. Negative means no threshold works** — the single most
           useful number here, and one that plain accuracy hides.

Reading the per-category table
------------------------------
    exact/single low   → DATA problem. Records cannot find themselves, or are
                         too similar to separate. Changing retriever will not help.
    exact high,
    colloquial low     → ALGORITHM problem. Vocabulary gap; semantic embeddings
                         or hybrid search is the fix.
    typo low           → tokenisation problem; character n-grams should absorb this.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from tools.chunking import build_chunks, describe, split_symptom_phrases  # noqa: E402
from tools.config_loader import ConfigManager  # noqa: E402
from tools.embedders import TfidfEmbedder, make_embedder  # noqa: E402
from tools.ImportDB2Faiss import VetDB  # noqa: E402
from tools.retrievers import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402

TESTSET = BASE_DIR / "data" / "testset.csv"
CHUNKINGS = [
    "row_symptom_only",
    "row_combined",
    "row_full",
    "per_symptom",
    "per_symptom_expanded",
]



class Tee:
    """Print to the terminal AND capture everything for the summary file."""

    def __init__(self):
        self.lines = []

    def __call__(self, *parts, sep=" "):
        text = sep.join(str(p) for p in parts) if parts else ""
        print(text)
        self.lines.append(text)

    def text(self) -> str:
        return "\n".join(self.lines)


def write_html_report(path: Path, results: List[dict], categories: List[str],
                      summary_text: str, failures: List[dict]) -> None:
    """Self-contained HTML — open it in a browser, no server needed."""

    def cell(value, good_high=True, lo=0.0, hi=1.0):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"<td>{value}</td>"
        pct = 0.0 if hi == lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
        if not good_high:
            pct = 1 - pct
        hue = 120 * pct  # red -> green
        return f'<td style="background:hsl({hue:.0f},70%,92%)">{v:.3f}</td>'

    head = "".join(f"<th>{c}</th>" for c in ["chunking", "retriever", "R@1", "R@3", "MRR",
                                             "noise gap", "ms"] + categories)
    body = []
    best = max(r["recall@1"] for r in results)
    for r in results:
        gap = r["noise_gap"]
        gap_ok = isinstance(gap, (int, float)) and gap > 0
        star = " ★" if r["recall@1"] == best else ""
        body.append(
            "<tr>"
            f'<td class="k">{r["chunking"]}</td><td class="k">{r["retriever"]}{star}</td>'
            + cell(r["recall@1"]) + cell(r["recall@3"]) + cell(r["mrr"])
            + f'<td class="{"ok" if gap_ok else "bad"}">{gap}</td>'
            + f'<td>{r["median_ms"]}</td>'
            + "".join(cell(r.get(f"cat_{c}", "")) for c in categories)
            + "</tr>"
        )

    fail_rows = "".join(
        f'<tr><td>{f["config"]}</td><td>{f["category"]}</td><td>{f["query"]}</td>'
        f'<td>{f["expected"]}</td><td>{f["got"]}</td></tr>'
        for f in failures[:300]
    )

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Retrieval benchmark</title>
<style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      margin:0;padding:32px;background:#fafafa;color:#1a1a1a}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:32px 0 8px}}
 .meta{{color:#666;font-size:13px;margin-bottom:24px}}
 table{{border-collapse:collapse;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1);
        font-variant-numeric:tabular-nums;margin-bottom:8px}}
 th,td{{padding:6px 10px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap}}
 th{{background:#f4f4f5;font-weight:600;text-align:right;position:sticky;top:0}}
 td.k,th:first-child,th:nth-child(2){{text-align:left}}
 td.ok{{background:#dcfce7;color:#166534;font-weight:600}}
 td.bad{{background:#fee2e2;color:#991b1b}}
 pre{{background:#fff;padding:16px;border-radius:6px;overflow-x:auto;font-size:12px;
      box-shadow:0 1px 3px rgba(0,0,0,.1)}}
 .legend{{font-size:12px;color:#666;margin-bottom:24px}}
 details{{margin-top:8px}} summary{{cursor:pointer;font-weight:600}}
 @media (prefers-color-scheme:dark){{
   body{{background:#18181b;color:#e4e4e7}} table,pre{{background:#27272a}}
   th{{background:#3f3f46}} th,td{{border-color:#3f3f46}}
   td.ok{{background:#14532d;color:#bbf7d0}} td.bad{{background:#7f1d1d;color:#fecaca}}
   .meta,.legend{{color:#a1a1aa}}
 }}
</style></head><body>
<h1>Retrieval benchmark</h1>
<div class="meta">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &middot;
  {len(results)} configurations</div>
<div class="legend">
  <b>R@1</b> correct record is the top hit &middot;
  <b>R@3</b> in the top 3 &middot;
  <b>noise gap</b> <span style="color:#166534">green = a threshold works</span>,
  <span style="color:#991b1b">red = no threshold separates hits from noise</span> &middot;
  &#9733; best R@1
</div>
<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>

<h2>Console output</h2>
<pre>{summary_text.replace("&", "&amp;").replace("<", "&lt;")}</pre>

<h2>Failures ({len(failures)})</h2>
<details><summary>show wrong answers</summary>
<table><thead><tr><th>config</th><th>category</th><th>query</th>
<th>expected</th><th>got</th></tr></thead><tbody>{fail_rows}</tbody></table>
</details>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
def load_cases(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("query") or "").strip()]
    for r in rows:
        r["expected_data_id"] = int(r["expected_data_id"])
    return rows


def evaluate(retriever, cases: List[dict], top_k: int = 5) -> dict:
    hits1 = hits3 = 0
    rr: List[float] = []
    per_cat: Dict[str, List[int]] = defaultdict(list)
    correct_scores: List[float] = []
    noise_scores: List[float] = []
    failures: List[dict] = []
    latencies: List[float] = []

    positives = [c for c in cases if c["expected_data_id"] > 0]

    for case in cases:
        t0 = time.perf_counter()
        results = retriever.search(case["query"], top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        expected = case["expected_data_id"]
        ids = [rid for rid, _ in results]

        if expected < 0:  # negative case
            if results:
                noise_scores.append(results[0][1])
            continue

        rank = ids.index(expected) + 1 if expected in ids else 0
        ok1 = rank == 1
        per_cat[case["category"]].append(1 if ok1 else 0)

        if ok1:
            hits1 += 1
            correct_scores.append(results[0][1])
        else:
            failures.append(
                {
                    "query": case["query"],
                    "category": case["category"],
                    "expected": expected,
                    "got": ids[0] if ids else None,
                    "rank": rank,
                }
            )
        if 0 < rank <= 3:
            hits3 += 1
        rr.append(1.0 / rank if rank else 0.0)

    n = max(len(positives), 1)
    noise_gap = (
        min(correct_scores) - max(noise_scores)
        if correct_scores and noise_scores
        else float("nan")
    )
    return {
        "recall@1": hits1 / n,
        "recall@3": hits3 / n,
        "mrr": statistics.mean(rr) if rr else 0.0,
        "noise_gap": noise_gap,
        "median_ms": statistics.median(latencies) if latencies else 0.0,
        "per_category": {k: statistics.mean(v) for k, v in per_cat.items()},
        "failures": failures,
    }


# ---------------------------------------------------------------------------
def data_health(rows: List[dict], out=print) -> None:
    """Problems no retriever can fix. Run this before tuning anything."""
    out("\n" + "=" * 78)
    out("DATA HEALTH")
    out("=" * 78)

    problems = 0

    # -- thin or empty symptom text ---------------------------------------
    thin = [r for r in rows if len((r["cause"] or "").strip()) < 20]
    if thin:
        problems += len(thin)
        out(f"\n⚠️  {len(thin)} record(s) with < 20 chars of symptom text (will rarely match):")
        for r in thin:
            out(f"     id={r['data_id']:<3} {r['symptoms'][:40]}  → {(r['cause'] or '')!r}")

    # -- few phrases -------------------------------------------------------
    few = [r for r in rows if len(split_symptom_phrases(r["cause"])) < 3]
    if few:
        out(f"\n⚠️  {len(few)} record(s) with < 3 distinct symptom phrases:")
        for r in few:
            out(f"     id={r['data_id']:<3} {r['symptoms'][:40]}")

    # -- near-duplicate records -------------------------------------------
    embedder = TfidfEmbedder()
    texts = [f'{r["cause"] or ""} {r["symptoms"]}' for r in rows]
    embedder.fit(texts)
    matrix = embedder.encode(texts)
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, 0.0)

    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if sim[i, j] > 0.45:
                pairs.append((float(sim[i, j]), rows[i], rows[j]))
    pairs.sort(key=lambda p: -p[0])

    if pairs:
        problems += len(pairs)
        out(f"\n⚠️  {len(pairs)} confusable record pair(s) (cosine > 0.45) — these compete:")
        for s, a, b in pairs[:8]:
            out(f"     {s:.3f}  id={a['data_id']} {a['symptoms'][:32]}")
            out(f"            id={b['data_id']} {b['symptoms'][:32]}")
    else:
        out("\n✅ No confusable record pairs above 0.45.")

    # -- self-retrieval ----------------------------------------------------
    chunks = build_chunks(rows, "row_combined")
    dense = DenseRetriever(TfidfEmbedder()).fit(chunks)
    self_fail, magnets = [], defaultdict(int)
    for r in rows:
        res = dense.search(r["cause"] or r["symptoms"], top_k=3)
        if not res or res[0][0] != r["data_id"]:
            self_fail.append((r, res[0][0] if res else None))
        for rid, _ in res:
            magnets[rid] += 1

    if self_fail:
        problems += len(self_fail)
        out(f"\n❌ {len(self_fail)} record(s) FAIL to retrieve themselves — a hard data problem:")
        for r, got in self_fail:
            out(f"     id={r['data_id']} {r['symptoms'][:38]} → returned id={got}")
    else:
        out("✅ Every record retrieves itself from its own symptom text.")

    # -- magnet records ----------------------------------------------------
    hot = [(rid, c) for rid, c in magnets.items() if c > max(3, len(rows) * 0.25)]
    if hot:
        out(f"\n⚠️  {len(hot)} 'magnet' record(s) appearing in many unrelated top-3s:")
        by_id = {r["data_id"]: r for r in rows}
        for rid, c in sorted(hot, key=lambda kv: -kv[1]):
            out(f"     id={rid} appears {c}x — {by_id[rid]['symptoms'][:40]}")

    out(f"\n{'─'*78}\nTotal data issues flagged: {problems}")
    out("Records:", len(rows))

    out("\nChunking cost per strategy:")
    for name, stats in describe(rows).items():
        out(
            f"   {name:<18} {stats['chunks']:>4} chunks "
            f"({stats['chunks_per_row']:>4} per row)  avg {stats['avg_chars']:>6.1f} chars"
        )


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=TESTSET)
    parser.add_argument("--with-openai", action="store_true", help="include OpenAI embeddings ($)")
    parser.add_argument("--data-health", action="store_true", help="diagnostics only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BASE_DIR / "benchmark_results",
        help="folder for saved reports (default: benchmark_results/)",
    )
    parser.add_argument("--no-save", action="store_true", help="print only, save nothing")
    parser.add_argument("--csv", type=Path, help="also write the matrix to this exact path")
    parser.add_argument("--show-failures", type=int, default=0, help="print N failures per config")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = ConfigManager.get_config_database()
    rows = VetDB.get_all_vet_doc(cfg["DB_NAME"])
    if not rows:
        print("❌ vet_doc is empty.")
        return 1

    out = Tee()
    data_health(rows, out)
    if args.data_health:
        _save(args, out, [], [], [])
        return 0

    if not args.testset.exists():
        print(f"\n❌ No testset at {args.testset}. Run: python scripts/make_testset.py")
        return 1
    cases = load_cases(args.testset)

    # -- build the matrix --------------------------------------------------
    configs = []
    for chunking in CHUNKINGS:
        chunks = build_chunks(rows, chunking)
        configs.append((chunking, "bm25", BM25Retriever().fit(chunks)))
        configs.append((chunking, "tfidf", DenseRetriever(TfidfEmbedder()).fit(chunks)))
        configs.append(
            (
                chunking,
                "hybrid(bm25+tfidf)",
                HybridRetriever([BM25Retriever(), DenseRetriever(TfidfEmbedder())]).fit(chunks),
            )
        )
        if args.with_openai:
            oa = make_embedder("openai", ConfigManager.get_retrieval_config()["embedding_model"])
            configs.append((chunking, "openai", DenseRetriever(oa).fit(chunks)))
            configs.append(
                (
                    chunking,
                    "hybrid(bm25+openai)",
                    HybridRetriever(
                        [BM25Retriever(), DenseRetriever(make_embedder("openai"))]
                    ).fit(chunks),
                )
            )

    out("\n" + "=" * 78)
    out(f"BENCHMARK — {len(cases)} cases, {len(rows)} records, {len(configs)} configs")
    out("=" * 78)

    categories = sorted({c["category"] for c in cases if c["expected_data_id"] > 0})
    header = f"{'chunking':<22}{'retriever':<21}{'R@1':>6}{'R@3':>6}{'MRR':>6}{'gap':>9}{'ms':>7}  "
    header += "".join(f"{c[:6]:>8}" for c in categories)
    out("\n" + header)
    out("-" * len(header))

    results = []
    all_failures: List[dict] = []
    for chunking, name, retriever in configs:
        m = evaluate(retriever, cases, args.top_k)
        for f in m["failures"]:
            all_failures.append({**f, "config": f"{chunking} + {name}"})
        gap = m["noise_gap"]
        line = (
            f"{chunking:<22}{name:<21}"
            f"{m['recall@1']:>6.3f}{m['recall@3']:>6.3f}{m['mrr']:>6.3f}"
            f"{gap:>9.3f}{m['median_ms']:>7.2f}  "
        )
        line += "".join(f"{m['per_category'].get(c, 0):>8.2f}" for c in categories)
        out(line)

        row = {
            "chunking": chunking,
            "retriever": name,
            "recall@1": round(m["recall@1"], 4),
            "recall@3": round(m["recall@3"], 4),
            "mrr": round(m["mrr"], 4),
            "noise_gap": round(gap, 4) if gap == gap else "",
            "median_ms": round(m["median_ms"], 3),
        }
        row.update({f"cat_{c}": round(m["per_category"].get(c, 0), 4) for c in categories})
        results.append(row)

        if args.show_failures and m["failures"]:
            for f in m["failures"][: args.show_failures]:
                out(f"      ✗ [{f['category']}] {f['query'][:46]} → id={f['got']} (want {f['expected']})")

    # -- verdict -----------------------------------------------------------
    def gap_of(r):
        return r["noise_gap"] if isinstance(r["noise_gap"], (int, float)) else float("-inf")

    by_recall = max(results, key=lambda r: (r["recall@1"], r["mrr"]))
    usable = [r for r in results if gap_of(r) > 0]
    recommended = max(usable, key=lambda r: (r["recall@1"], r["mrr"])) if usable else None
    baseline = next(
        (r for r in results if r["chunking"] == "row_combined" and r["retriever"] == "tfidf"), None
    )

    out("\n" + "=" * 78)
    out("VERDICT")
    out("=" * 78)
    if baseline:
        out(
            f"  production   {baseline['chunking']} + {baseline['retriever']}"
            f"   R@1={baseline['recall@1']:.3f}  gap={gap_of(baseline):+.3f}"
        )
    out(
        f"  best R@1     {by_recall['chunking']} + {by_recall['retriever']}"
        f"   R@1={by_recall['recall@1']:.3f}  gap={gap_of(by_recall):+.3f}"
        + ("   ⚠️  gap<=0: no threshold separates hits from noise" if gap_of(by_recall) <= 0 else "")
    )
    if recommended:
        delta = recommended["recall@1"] - (baseline["recall@1"] if baseline else 0)
        out(
            f"  RECOMMENDED  {recommended['chunking']} + {recommended['retriever']}"
            f"   R@1={recommended['recall@1']:.3f}  gap={gap_of(recommended):+.3f}"
            f"   (Δ R@1 {delta:+.3f} vs production)"
        )
        out("               ↑ highest accuracy among configs with a USABLE threshold")
    else:
        out("  ⚠️  No configuration has a positive noise gap. Every threshold either")
        out("      drops correct hits or admits off-topic queries. Fix the data first.")

    worst_cat = min(
        ((c, min(r.get(f"cat_{c}", 1) for r in results)) for c in categories), key=lambda kv: kv[1]
    )
    best_of_worst = max(r.get(f"cat_{worst_cat[0]}", 0) for r in results)
    out(
        f"\n  weakest category: {worst_cat[0]} — best any config achieves is {best_of_worst:.2f}"
    )
    if best_of_worst < 0.8:
        out("  → no lexical strategy fixes this. Test --with-openai; it is a semantic gap.")
    out("=" * 78)

    _save(args, out, results, categories, all_failures)
    return 0


def _save(args, out: "Tee", results: List[dict], categories: List[str], failures: List[dict]):
    if args.no_save:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.txt").write_text(out.text(), encoding="utf-8")
    (args.out_dir / "latest_summary.txt").write_text(out.text(), encoding="utf-8")

    if results:
        for target in (run_dir / "results.csv", args.out_dir / "latest_results.csv"):
            with open(target, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                w.writerows(results)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                w.writerows(results)

    if failures:
        for target in (run_dir / "failures.csv", args.out_dir / "latest_failures.csv"):
            with open(target, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["config", "category", "query", "expected", "got", "rank"]
                )
                w.writeheader()
                w.writerows(failures)

    if results:
        for target in (run_dir / "report.html", args.out_dir / "latest_report.html"):
            write_html_report(target, results, categories, out.text(), failures)

    print(f"\n📁 Saved to {run_dir.relative_to(BASE_DIR)}/")
    if results:
        report = (args.out_dir / "latest_report.html").relative_to(BASE_DIR)
        print(f"   open {report}   ← start here")
    else:
        print(f"   {(args.out_dir / 'latest_summary.txt').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    raise SystemExit(main())
