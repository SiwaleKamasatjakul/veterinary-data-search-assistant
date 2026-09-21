# Veterinary Cat Store Chatbot — macOS

A macOS-ready rebuild of `ChatbotCatStoreLinux`. Same architecture, same API contract,
same knowledge base — with the portability blockers and several latent bugs fixed, plus a
pluggable embedding backend and a retrieval benchmark.

```
user question
   └─> embedder   TF-IDF char n-grams (default)  ·  OpenAI text-embedding-3-small (opt-in)
        └─> FAISS IndexIDMap2(IndexFlatIP) over the vet_doc knowledge base
             └─> unit vectors, so the score IS cosine similarity (-1..1, higher is better)
                  └─> top-k above min_similarity → matched disease records
                       └─> injected into the system prompt as clinical reference
                            └─> LLM answer (+ saved to chat history)
```

---

> **New to this codebase?** Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it maps the
> request lifecycle, explains what every file does, and documents the embedding backends.
> For the benchmark numbers, read [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).

## Setup

Requires **Python 3.11**. macOS ships 3.9, which has no arm64 wheels for faiss/numpy/scipy.

```bash
brew install python@3.11          # skip if you already have it

cd veterinary-data-search-assistant
/opt/homebrew/bin/python3.11 -m venv .venv    # Intel Mac: /usr/local/bin/python3.11
source .venv/bin/activate
python --version                  # must say 3.11.x

pip install -r requirements.txt
cp .env.example .env
python scripts/build_index.py     # seeds vet_doc + builds the FAISS index
```

## Run

```bash
./run.sh
```

or `uvicorn src.LLM_Process_API:app --reload` from the project root.

| URL | |
|---|---|
| http://127.0.0.1:8000/docs | **Swagger UI** — the request example is pre-filled |
| http://127.0.0.1:8000/health | mode, record count, index status |

**Offline mode.** Leave `OPENAI_API_KEY` empty in `.env` and the app runs with a stub LLM.
Retrieval, chat history and the response envelope all work — no key, no cost. Fill the key in
and restart to use the real model; nothing else changes.

## Endpoints

| Method | Path | |
|---|---|---|
| POST | `/message_handler/` | The chat contract — unchanged request shape from the original |
| GET | `/message_handler/search?q=...` | Retrieval only, no LLM call. Tune here for free |
| GET | `/health` | Status |

Verified response (offline mode):

```json
{
  "id": "f30416b1-…",
  "object": "chat.completion",
  "choices": [{ "message": { "role": "assistant", "content": "…", "refdata": [...] } }],
  "vet_doc": [{ "disease": "โรคเชื้อราในแมว (Ringworm)", "score": 0.677 }],
  "mock": true,
  "latency_ms": 6
}
```

---

## Storage layers

Seven stores, each with one job. Only the first two are committed; everything else is a
build artifact, a cache, or runtime state, and all of it is reproducible from
`scripts/build_index.py`.

| layer | path | holds | lifecycle |
|---|---|---|---|
| **Knowledge base** | `data/vet_doc.db` | SQLite, **20 disease records** — the only source of truth for retrieval | committed; seeded from `seed_vet_doc.json` + `more_vet_doc.json` |
| **Synonyms** | `data/synonyms.json` | colloquial → clinical rewordings (`ฉี่` → `ปัสสาวะ`) | committed; feeds `per_symptom_expanded` chunking and testset generation |
| **Vector index** | `src/vectordb/vet_doc.index` | FAISS `IndexIDMap2(IndexFlatIP)`, vectors keyed by `data_id` — not by position | build artifact, gitignored |
| **Embedder state** | `src/vectordb/vet_doc_embedder.joblib` | the fitted TF-IDF vectorizer (the OpenAI backend writes only `{model, dim}`) | build artifact, gitignored |
| **Index metadata** | `src/vectordb/index_meta.json` | `{backend, embedder, dim, records}` — startup compares this against config and **rebuilds automatically on mismatch** | build artifact, gitignored |
| **Embedding cache** | `data/embedding_cache.db` | SQLite keyed `(model, sha256(text))` — OpenAI vectors only, so a rebuild re-sends only text that actually changed | gitignored; survives rebuilds, which is the whole point |
| **Chat history** | `data/chat_history.db` | conversation turns per session, replayed into the next request | runtime, gitignored |

TF-IDF is never cached in `embedding_cache.db`: its vectors depend on the whole corpus, so
adding one record changes every vector and a rebuild must refit from scratch. OpenAI vectors
are corpus-independent, so they cache. That single property — `is_corpus_coupled` in
`src/tools/embedders.py` — is what `build_index()` branches on.

### Which layers the benchmark touches

The benchmark builds every index **in memory** and never touches the production stores.
It is safe to run against a live app.

| layer | benchmark |
|---|---|
| `data/vet_doc.db` | **read** — the corpus under test |
| `data/synonyms.json` | **read** — for `per_symptom_expanded` |
| `data/testset.csv` | **read** — the answer key (gitignored; regenerate, don't hand-edit) |
| `benchmark_results/` | **written** — the only thing it writes |
| `vet_doc.index`, `vet_doc_embedder.joblib`, `index_meta.json` | untouched |
| `data/embedding_cache.db` | untouched — see the cost warning below |
| `data/chat_history.db` | untouched |

> ⚠️ `--with-openai` does **not** go through the embedding cache. Every run re-embeds all
> 167 queries and all chunks from scratch, and pays for it. Runs take ~200 ms per query
> against the API versus ~0.2 ms locally, so budget minutes, not seconds.

---

## Benchmark

`scripts/benchmark.py` grades the **retrieval layer only** — not the chatbot. One question:
a cat owner types some symptoms; does search find the right disease record?

```bash
python scripts/make_testset.py       # rebuild the answer key from vet_doc
python scripts/benchmark.py          # local retrievers only — free, seconds
python scripts/benchmark.py --with-openai   # add the two OpenAI rows ($, minutes)
python scripts/benchmark.py --data-health   # diagnostics only, no matrix
open benchmark_results/latest_report.html
```

Every run writes `latest_report.html`, `latest_summary.txt`, `latest_results.csv` and
`latest_failures.csv`, plus a timestamped `run_YYYYMMDD_HHMMSS/` copy of all four.

### What it measures

**Ground truth** is `data/testset.csv` — 167 `query → expected_data_id` pairs, generated
from the records themselves and tagged by category (`exact`, `partial`, `single`,
`colloquial`, `typo`, plus 10 off-topic `negative` cases that should return nothing).

The matrix is **5 chunking strategies × 5 retrievers = 25 configs**, all taking the same exam:

- chunking — `row_symptom_only`, `row_combined` (production), `row_full`, `per_symptom`, `per_symptom_expanded`
- retriever — `bm25`, `tfidf` (production), `hybrid(bm25+tfidf)`, `openai`, `hybrid(bm25+openai)`

| metric | direction | means |
|---|---|---|
| `R@1` | higher | correct record was the #1 hit — what the user experiences at `top_k=1` |
| `R@3` | higher | correct record was in the top 3 — headroom a reranker could recover |
| `MRR` | higher | 1/rank of the correct record, averaged — sees near-misses that `R@1` hides |
| `gap` | **must be > 0** | (lowest correct score) − (best off-topic score). Positive means a `min_similarity` exists that keeps every hit and rejects every off-topic query. Negative means **no threshold works** |
| `exact`, `partial` | must stay 1.00 | data-health alarm — if these drop, stop tuning and fix the records |
| `colloquial`, `typo` | higher | the categories that actually separate the configs |

[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) explains every column from scratch, including the
trap where the best `R@1` has an unusable threshold.

### Latest results — 167 cases, 20 records, 25 configs

Selected rows (full matrix in `benchmark_results/latest_summary.txt`):

| chunking | retriever | R@1 | R@3 | MRR | gap | ms | colloq | |
|---|---|---|---|---|---|---|---|---|
| `row_combined` | `tfidf` | 0.739 | 0.809 | 0.788 | −0.026 | 0.21 | 0.43 | ← **production today** |
| `row_symptom_only` | `tfidf` | **0.764** | 0.828 | 0.806 | −0.093 | 0.20 | 0.46 | ← best R@1, unusable threshold |
| `per_symptom_expanded` | `bm25` | 0.752 | 0.866 | 0.820 | **+0.974** | 0.40 | 0.54 | ← **recommended** |
| `row_combined` | `openai` | 0.529 | 0.662 | 0.602 | −0.233 | 203.37 | 0.14 | |
| `per_symptom_expanded` | `openai` | 0.707 | 0.866 | 0.795 | −0.037 | 203.90 | 0.49 | |
| `per_symptom_expanded` | `hybrid(bm25+openai)` | 0.752 | 0.866 | **0.823** | −0.002 | 205.59 | **0.59** | ← best colloquial of all 25 |

Three findings worth carrying forward:

1. **The best `R@1` is a trap.** `row_symptom_only + tfidf` wins on accuracy but its gap is
   negative — no cutoff separates real hits from nonsense, so the app cannot safely say
   "I'm not sure." The script deliberately refuses to recommend on `R@1` alone.
2. **`per_symptom_expanded + bm25` is the recommendation**: +0.013 `R@1` over production and
   a clean +0.974 gap, at zero cost and 0.4 ms.
3. **`colloquial` is the ceiling.** No lexical config passes 0.54, because `ฉี่` and
   `ปัสสาวะ` share no characters. Only the OpenAI hybrid breaks through, at 0.59 — and it
   costs 500× the latency for it.

### Known flaws in the ground truth

- **~14 of ~41 failures are mislabelled.** The generator picks one arbitrary record when a
  symptom appears in several (frequent urination is listed under both Diabetes and CKD), so
  a medically reasonable answer is scored wrong. **Real `R@1` is closer to 0.83 than 0.739** —
  treat absolute numbers as pessimistic by roughly 9 points. Comparisons *between* rows stay
  valid; every config suffers the same mislabels.
- The `colloquial` synonyms are guesses about how Thai owners speak, not observed language.
- Only 10 off-topic cases, so the `gap` rests on a thin sample.
- Questions are generated *from* the records, so this measures "can search find the record it
  came from," not "can search answer real users."

The strongest upgrade available: label real questions out of `data/chat_history.db` and run
`benchmark.py --testset real_queries.csv`. Fifty real questions beat 167 generated ones.

---

## Why the retrieval backend is TF-IDF today

**Retrieval started on OpenAI embeddings.** The original design embedded the user's question
with `text-embedding-3-small`, searched FAISS, and injected the best-matching record — chosen
because semantic vectors are the obvious answer to a user who writes `ฉี่` when the record
says `ปัสสาวะ`.

Character n-gram TF-IDF came later, as the local, free, no-network alternative for Thai text.
The benchmark is what settled which one ships: **on this 20-record corpus, OpenAI embeddings
alone score materially lower than plain character matching** — `R@1` 0.529 vs 0.739 on the
production chunking, and 0.14 vs 0.43 on the colloquial cases the embeddings were supposed to
win. Twenty short records give the dense vectors very little to separate, and Thai clinical
phrasing is where lexical matching is strongest.

The measured picture is narrower than "OpenAI loses":

- **alone, on coarse chunks** — clearly worse (0.529)
- **alone, on fine chunks** — competitive (0.707 on `per_symptom_expanded`)
- **hybrid with BM25 on fine chunks** — ties the best `R@1` (0.752), takes the best `MRR`
  (0.823), and is the **only** config that pushes `colloquial` past 0.54, to 0.59

So the semantic argument holds where it was always going to hold — paraphrases — and nowhere
else, at ~200 ms and a per-query API call. TF-IDF stays the default; OpenAI stays one config
line away, and becomes the right call as the corpus grows.

### Switching

```bash
# .env
OPENAI_API_KEY=sk-...
```

```jsonc
// config/llm_config.json
"retrieval": {
  "backend": "openai",                          // "tfidf" | "openai"  — the EMBEDDINGS
  "embedding_model": "text-embedding-3-small",
  "min_similarity": { "tfidf": 0.19, "openai": 0.35 }
}
```

```bash
python scripts/build_index.py                   # re-embeds once, then caches
python scripts/check_retrieval.py --tune --backend openai
```

> Two keys named `backend` live in `llm_config.json` and they are unrelated:
> `retrieval.backend` chooses the **embedding** model, the top-level `backend` chooses the
> **chat** model. OpenAI is the chat model from day one; only the embedding side is pluggable.

`RETRIEVAL_BACKEND=openai` as an env var overrides the config for a single run. The index
records which backend built it, so a config/index mismatch rebuilds on startup rather than
serving vectors from the wrong space.

**Thresholds are per backend** — the scales differ, so one number cannot serve both. The
tfidf value of 0.19 is measured (correct hits 0.2098–0.7409, off-topic ≤ 0.1794). **The
openai value of 0.35 is a starting estimate**; run `--tune --backend openai` and replace it.

## Tests

```bash
pytest -q        # no API key required
```

---

## What changed from ChatbotCatStoreLinux, and why

### The four things that made it unrunnable on macOS

**1. The FAISS index was locked to x86.** `vet_doc_faiss_index.pkl` was written with
`pickle.dump(faiss_index)`, and the pickle embedded the SWIG module name
`faiss.swigfaiss_avx2` — the x86-64 AVX2 build. Apple Silicon FAISS has no such module,
so loading it raised `ModuleNotFoundError: No module named 'faiss.swigfaiss_avx2'`.
This project uses `faiss.write_index` / `faiss.read_index`, which produce an
architecture-independent file. The index is now a build artifact, rebuilt by
`scripts/build_index.py` or automatically on first startup.

**2. Hardcoded home directory.** `config_loader.load_config()` defaulted to
`/home/siwale/Documents/Chatbot/ChatbotCatStoreLinux/config/llm_config.json`, and
`llm_config.json` held three more absolute paths. Since config loads at import time,
the app died before FastAPI was even constructed. Everything now resolves from
`src/tools/paths.py::BASE_DIR`, derived from the file's own location — the project runs
from any folder on any machine.

**3. The bundled `env/` was a Linux venv** (Python 3.8.10, `home = /usr/bin`, 478 MB of
x86-64 binaries). Replaced by `requirements.txt` pinned to versions that all have prebuilt
arm64 wheels for Python 3.11 — nothing compiles from source.

**4. Python 3.8 is a dead end on Apple Silicon.** numpy 1.24.4, scipy 1.10.1 and
scikit-learn 1.3.2 have no arm64 wheels for 3.8; pip falls back to building from source
and fails. Now 3.11.

### Bugs that were broken on Linux too

**The chat endpoint returned nothing.** `message_handler` called `extract_parameters`, set
`prompt_type`, and left the `StreamingResponse(...)` line commented out — so every request
answered `null`. The route now runs the full pipeline and returns the formatted response.

**The retrieved context never reached the model.** `generate_response` built a generic
system prompt and passed the whole `input_data` dict as the human message, so the FAISS
results were formatted and then effectively discarded. They are now injected into the
system prompt as a clinical reference block — which is the entire point of the pipeline.

**Chat history was loaded and dropped.** History was read from SQLite, converted to
`HumanMessage`/`AIMessage`, assigned to `input_data["history"]`, and never used. It is now
passed to the model as real conversation turns.

**The distance filter was backwards and had a precedence bug.** The result comprehension
read `if i < len(records) and j < len(D[0]) or D[0][j] >= similarity_threshold` — `and`
binds tighter than `or`, so a FAISS miss (`-1`) could slip past the bounds check and raise
IndexError. Worse, `IndexFlatL2` returns *distances* (smaller is better) but they were
compared against a threshold as if larger were better.

> The index is now `IndexFlatIP` over L2-normalised vectors, so the score **is** cosine
> similarity: −1..1, **higher is better**, thresholded by `min_similarity`. Any older note
> saying "lower is better" is stale. For unit vectors the two relate as `L2² = 2(1 − cos)`.

**`chat_history.db` was a relative path.** `sqlite3.connect("chat_history.db")` created a
stray empty database in whatever directory you launched from. Now absolute, from config.

**The error handler raised its own error.** `JSONMessage.error_response` read `e.message`,
`e.param` and `e.code`, none of which exist on a plain Exception — so the except branch
raised AttributeError and masked the real failure. Now uses `getattr` fallbacks.

**`get_general_config` read the wrong key.** It looked up `general` but the JSON key is
`general_setting`, so host and port always silently fell through to the defaults.

**The index mapped by position, not by ID.** Vectors are now stored in `IndexIDMap2` against
`data_id`, so a search returns database IDs directly. Verified by deleting a row without
rebuilding: the query still returns the correct record instead of silently shifting.

**Other:** deprecated `langchain.chat_models` import → `langchain_openai`;
`datetime.utcnow()` → timezone-aware; `on_event("startup")` → lifespan handler;
formatters return dicts instead of JSON strings that FastAPI then double-encoded;
`execution_time.log` and `logs/` anchored to the project root instead of the cwd.

### Security

The original had a live OpenAI API key hardcoded in `src/models/llm.py`
(`os.environ["OPENAI_API_KEY"] = "sk-proj-..."`), plus a second one commented out above it.
**Both should be revoked** at platform.openai.com — they are in your git history and were
copied between machines. This project reads the key from `.env` only, and `.env` is
gitignored.

### Retrieval quality

`TfidfVectorizer()` defaults to word-level tokenisation, which is a poor fit for Thai —
it has no spaces between words, so each space-delimited phrase becomes a single token and
matching depends on the user reusing your exact phrasing. This project uses
`analyzer="char_wb", ngram_range=(2,4)`, which matches on character n-grams instead.

That choice is now measured rather than argued — see [Benchmark](#benchmark) above. Ad-hoc
spot checks live in `scripts/check_retrieval.py`; the 25-config matrix is the real answer.

### A note on the data

In `vet_doc`, the `symptoms` column actually holds the **disease name**, `cause` holds the
**symptom list**, and `diagnosis` holds the **cause**. The original indexed column 2
(`cause`), which is the correct text to search on — so the behaviour was right even though
the labels are shifted. That indexing behaviour is preserved, and search results expose the
fields under honest names (`disease`, `symptoms`, `cause`, `treatment`). If you ever fix the
column names in the schema, `_search_text()` in `ImportDB2Faiss.py` is the one place to update.

## Layout

```
veterinary-data-search-assistant/
├── run.sh                          start the API from anywhere
├── requirements.txt                Python 3.11, arm64 wheels only
├── .env.example                    copy to .env
├── ARCHITECTURE.md                 request lifecycle, file reference, backends
├── BENCHMARK_GUIDE.md              every benchmark column, explained from scratch
├── config/llm_config.json          paths, retrieval backend, thresholds
├── data/
│   ├── vet_doc.db                  knowledge base (20 records)
│   ├── seed_vet_doc.json           source of truth for rebuilding the DB
│   ├── more_vet_doc.json           additional records
│   ├── synonyms.json               colloquial → clinical rewordings
│   ├── testset.csv                 benchmark ground truth (generated, gitignored)
│   ├── retrieval_tests.csv         ad-hoc spot checks
│   ├── embedding_cache.db          OpenAI vectors, keyed by (model, hash)
│   └── chat_history.db             created at runtime
├── scripts/
│   ├── build_index.py              seed + rebuild the FAISS index
│   ├── benchmark.py                25-config retrieval matrix + data health
│   ├── make_testset.py             regenerate the answer key from vet_doc
│   ├── check_retrieval.py          spot checks and threshold tuning (--tune)
│   └── add_vet_doc.py              add records to the knowledge base
├── benchmark_results/              reports per run (gitignored)
├── src/
│   ├── LLM_Process_API.py          FastAPI app
│   ├── core/                       request models, extraction, generation, logging
│   ├── Formatter/                  response envelopes
│   ├── models/llm.py               LLM factory (real / stub)
│   ├── prompts/prompts.py          system prompt
│   ├── routers/                    chat + search endpoints
│   ├── tools/
│   │   ├── paths.py                BASE_DIR, .env loading
│   │   ├── config_loader.py        config access
│   │   ├── database_process.py     sqlite
│   │   ├── chunking.py             the 5 chunking strategies
│   │   ├── embedders.py            TfidfEmbedder | OpenAIEmbedder
│   │   ├── embedding_cache.py      sqlite vector cache
│   │   ├── retrievers.py           BM25 | Dense | Hybrid
│   │   ├── ImportDB2Faiss.py       index build
│   │   └── SearchVecDoc.py         index search
│   └── vectordb/                   generated index + metadata (gitignored)
└── tests/test_api.py
```

## Possible next steps

- **Adopt `per_symptom_expanded` chunking.** The benchmark's standing recommendation: better
  `R@3`, better `MRR`, better `colloquial`, and the only production-viable config with a
  positive noise gap. Costs 222 chunks instead of 20.
- **Multi-label ground truth.** Replace `expected_data_id` with `acceptable_data_ids` so
  overlapping symptoms stop scoring correct answers as wrong, and the ~9-point pessimism
  in every number goes away.
- **Real-query testset.** Label questions out of `chat_history.db` and benchmark against
  those instead of generated ones.
- **Rerank the top 3.** `R@3` sits ~0.11 above `R@1` on the best configs — that is measured
  headroom a reranker could convert.
- **Streaming.** `parameters.stream` is accepted but ignored. Wire it to `StreamingResponse`
  plus `llm.stream()`; `ResponseFormatter._format_stream_response` is already written.
- **Product recommendation.** Add a `products` table keyed by disease, look it up alongside
  the retrieval hit, and extend the context block in `generate_response.build_messages`.
