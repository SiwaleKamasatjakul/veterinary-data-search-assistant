# Reading the benchmark — a beginner's guide

This explains every column in `benchmark_results/latest_report.html`, what the numbers
mean, and what a "ground truth dataset" is. No prior knowledge assumed.

---

## 1. The big idea: it's a report card

We are testing a **search engine**, not the chatbot. The question is always the same:

> A cat owner types some symptoms. Does the search find the right disease record?

To grade that, you need to already know the right answer for each question. That list of
"question → right answer" pairs is called the **ground truth**.

```
┌──────────────────────┐     ┌─────────────────┐     ┌──────────┐
│  GROUND TRUTH        │     │  THE SEARCH     │     │  GRADE   │
│  question + answer   │ ──► │  (the student)  │ ──► │  ✅ / ❌ │
│  data/testset.csv    │     │                 │     │          │
└──────────────────────┘     └─────────────────┘     └──────────┘
```

Each row of the table is **one student taking the same exam**. Same 167 questions every
time, so the scores are comparable.

---

## 2. What is "ground truth"?

Ground truth = **the answer key.** It is the thing you compare against, and it is assumed
correct by definition. In our project it lives in one file:

```csv
query,expected_data_id,category,note
ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน,6,exact,โรคเชื้อราในแมว
แมวขนหลุด เป็นอะไร,6,colloquial,from: ขนร่วง
วันนี้อากาศดีมาก,-1,negative,nothing should be retrieved
```

- **`query`** — what the user types
- **`expected_data_id`** — the row in `vet_doc` that *should* come back. **This is the
  ground truth.** `-1` means "nothing should come back."
- **`category`** — how hard/what kind of question (explained in §5)

**Where does the answer key come from?** The generator reads row 6 of your database and
writes questions *from* it. Because it built the question out of row 6, it already knows
the answer is 6. No manual labelling needed to get started.

**Why is a fixed answer key essential?** Without it, "did search improve?" is an opinion.
With it, it's a number. You can:

- compare 15 different designs fairly (same exam)
- catch regressions (a change that quietly breaks an old query)
- decide whether to spend money (do OpenAI embeddings actually score better?)

> ⚠️ Ground truth is only as good as its labels. Ours has a known flaw — see §7.

---

## 3. The two left columns: which "student" is being tested

```
chunking              retriever
row_combined          tfidf
```

**`chunking`** — how a database row is cut up before indexing.

| value | meaning |
|---|---|
| `row_symptom_only` | 1 searchable unit per row: just the symptom list |
| `row_combined` | 1 per row: symptoms + disease name — **what your app uses now** |
| `row_full` | 1 per row: every field including treatment |
| `per_symptom` | ~6 per row: each symptom phrase separately |
| `per_symptom_expanded` | ~10 per row: each phrase, plus colloquial rewordings |

**`retriever`** — how text is matched.

| value | meaning |
|---|---|
| `bm25` | keyword-style matching. Classic search-engine algorithm |
| `tfidf` | vector matching on character patterns — **what your app uses now** |
| `hybrid(bm25+tfidf)` | run both, merge the two ranked lists |

15 rows = 5 chunkings × 3 retrievers. Every combination takes the same exam.

---

## 4. The score columns

### `R@1` — Recall at 1 · **higher is better** · 0.000 to 1.000

**Out of all questions, how often was the correct record the #1 result?**

`0.739` means 73.9% of the time the right answer came first. This is the number that
matches what a user experiences, because your app shows the top result.

```
0.739 = "about 3 out of 4 questions got the right answer first"
```

### `R@3` — Recall at 3 · **higher is better** · 0.000 to 1.000

**How often was the correct record somewhere in the top 3?**

`0.866` means the right answer was in the top 3 on 86.6% of questions.

R@3 is always ≥ R@1 (being #1 means being in the top 3). The **gap between them is
headroom**: `per_symptom_expanded` has 0.752 → 0.866, so on 11% of questions the right
answer was found but ranked 2nd or 3rd. A reranker could recover those.

### `MRR` — Mean Reciprocal Rank · **higher is better** · 0.000 to 1.000

**How close to the top was the right answer, on average?**

You score `1 / position`:

| right answer at position | you score |
|---|---|
| 1st | 1.00 |
| 2nd | 0.50 |
| 3rd | 0.33 |
| not found | 0.00 |

Average those across all questions = MRR.

Why bother when you have R@1? Because R@1 can't tell a near-miss from a disaster. Two
systems both scoring 0.700 on R@1 look identical — but if one puts its misses at position
2 and the other at position 20, the first is far better. MRR sees that; R@1 doesn't.

### `gap` — noise gap · **higher is better** · positive is what you want

This is the most important column and the least obvious.

Your app has a cutoff (`min_similarity`). If nothing scores above it, the app returns
nothing and says "I'm not sure." That's correct behaviour for an off-topic question.

The gap asks: **is there any cutoff value that works?**

```
gap = (worst score among correct answers) − (best score among off-topic questions)
```

- **gap positive** ✅ — a clean line exists. Set the cutoff between them and you keep every
  correct answer while rejecting every off-topic question.
- **gap negative** ❌ — the ranges overlap. *No cutoff works.* Set it high and you throw
  away correct answers; set it low and nonsense gets through.

```
POSITIVE GAP (good)              NEGATIVE GAP (bad)

off-topic  ██                    off-topic     ████
correct         ████             correct     ████
           ↑ cutoff here                   ↑ nowhere clean
```

> ⚠️ **Only the sign is comparable across retrievers.** BM25 scores are unbounded, so its
> gap is on a huge scale (`-8.338`). TF-IDF scores live in −1..1, so its gap is small
> (`-0.093`). `-8.338` is **not** "worse than" `-0.093` — they're different units. Both
> just mean "negative = broken." Compare magnitudes only within the same retriever.

### `ms` — milliseconds · **lower is better**

Median time for one search. `0.16` = 0.16 thousandths of a second.

All values here are tiny and none should drive your decision — the LLM call that follows
takes 500–2000 ms, so retrieval speed is invisible by comparison. This column only starts
to matter at hundreds of thousands of records.

---

## 5. The right-hand columns: score by question type

Every question is tagged, so you get a separate score per type. **This is the diagnostic
part** — it tells you *what kind* of problem you have.

| category | what the question looks like | example |
|---|---|---|
| `exact` | the record's own text, copied | `น้ำมูกไหล ไอ จาม ตาแดง...` |
| `partial` | a few symptoms from the record, shuffled | `แมวน้ำตาไหล เบื่ออาหาร น้ำมูกไหล` |
| `single` | just one symptom | `แมวเบื่ออาหาร เป็นอะไร` |
| `colloquial` | the owner's everyday word instead of the vet's | `ฉี่` instead of `ปัสสาวะ` |
| `typo` | misspelled | `แมวขนร่ววง` |

### How to read them

```
exact 1.00, partial 1.00   → your DATA is fine. Every record can be found.
colloquial 0.43            → when the owner uses a different word, search fails
                             more than half the time.
```

**This split is the whole point.** It separates two very different problems:

- **`exact` low** → a **data problem**. A record can't even find itself, or two records
  are too similar to tell apart. Changing the search algorithm will not help; fix the
  records.
- **`exact` high but `colloquial` low** → an **algorithm problem**. The records are fine;
  character matching can't connect `ฉี่` to `ปัสสาวะ` because they share no letters. This
  is what semantic embeddings fix.

Your table shows the second pattern clearly.

### Which categories actually rank the configs?

Measured spread across all 15 rows:

| category | min | max | spread | what it's for |
|---|---|---|---|---|
| `typo` | 0.60 | 0.80 | 0.20 | ranks configs |
| `colloquial` | 0.41 | 0.54 | 0.14 | ranks configs |
| `single` | 0.70 | 0.78 | 0.08 | weak signal |
| `exact` | 1.00 | 1.00 | 0.00 | **data health alarm** |
| `partial` | 1.00 | 1.00 | 0.00 | **data health alarm** |

`exact` and `partial` are identical everywhere, so they never help you pick a winner. They
have a different job: **if they ever drop below 1.00, stop tuning and go fix your data.**

---

## 6. Reading your actual results

```
row_combined          tfidf    0.739 0.809 0.788  -0.026  ← your app today
per_symptom_expanded  bm25     0.752 0.866 0.820  +0.974  ← recommended
row_symptom_only      tfidf    0.764 0.828 0.806  -0.093  ← highest R@1
```

The trap: `row_symptom_only` has the best R@1 (0.764) — but its gap is **negative**. High
accuracy with no usable cutoff means you can't safely reject nonsense questions. Not worth
it for +0.025 R@1.

`per_symptom_expanded` scores slightly lower on R@1 but has the best R@3 (0.866), best MRR
(0.820), the best `colloquial` (0.54), and a **positive gap**. That's why the benchmark
recommends it — the script deliberately refuses to pick on R@1 alone.

**The headline finding:** `colloquial` never rises above 0.54 in *any* of the 15 rows. No
amount of chunking or hybrid search fixes it, because every retriever here matches
characters, and these queries share almost no characters with their record. That is the
measured argument for semantic embeddings — run `--with-openai` to see whether they
deliver.

---

## 7. Known flaw in our ground truth (be honest about this)

**14 of the ~41 failures are mislabelled.** Example:

```
query: "ปัสสาวะบ่อย เป็นอะไร"   (frequent urination)
  answer key says : Diabetes
  search returned : CKD (chronic kidney disease)
  → 'ปัสสาวะบ่อย' is listed in BOTH records
```

The search is marked wrong, but CKD is a medically reasonable answer. The generator picked
one arbitrarily because it built the question from that row.

**So real R@1 is closer to 0.83, not 0.739.** Treat absolute numbers as pessimistic by
roughly 9 points. Comparisons *between* rows are still valid, because every config suffers
the same mislabels equally.

The fix is multi-label ground truth — allow several acceptable answers per question:

```csv
query,acceptable_data_ids,category
ปัสสาวะบ่อย เป็นอะไร,"4,19",single
```

Other known limits:

- The `colloquial` synonyms are *our guesses* about how Thai owners speak, not observed
  language.
- Only 10 off-topic questions — the gap rests on a thin sample.
- Questions are generated from the records, so this measures "can search find the record
  it came from," not "can search answer real users."

**The strongest upgrade available:** collect real user questions from `chat_history.db`,
have someone label the correct disease for each, save as CSV, and run
`benchmark.py --testset real_queries.csv`. Fifty real questions beat 167 generated ones.

---

## 8. Cheat sheet

| column | higher or lower? | range | means |
|---|---|---|---|
| `R@1` | **higher** | 0–1 | right answer was #1 |
| `R@3` | **higher** | 0–1 | right answer was in top 3 |
| `MRR` | **higher** | 0–1 | how near the top, on average |
| `gap` | **higher; must be > 0** | varies | whether any cutoff works |
| `ms` | **lower** | ms | search speed (rarely matters) |
| `exact`, `partial` | **must stay 1.00** | 0–1 | data health alarm |
| `colloquial`, `typo` | **higher** | 0–1 | what actually ranks configs |

Commands:

```bash
python scripts/make_testset.py    # rebuild the answer key from vet_doc
python scripts/benchmark.py       # take the exam, save reports
open benchmark_results/latest_report.html
```
