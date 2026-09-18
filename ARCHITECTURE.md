# ARCHITECTURE — read this first

You're picking up a RAG chatbot for a cat store. A pet owner describes symptoms in Thai,
the service finds matching disease records in a small curated knowledge base, injects them
into the prompt, and the LLM answers using *those records* rather than its own guesses.

This document is the map. Read it top to bottom once (~15 min), then use the
[Where do I change X?](#where-do-i-change-x) table as your day-to-day reference.

---

## 1. Why this exists

The original version sent the user's question straight to the LLM. That failed on a specific
class of input: colloquial Thai symptom words that look like ordinary words. The model read
them literally and answered confidently about the wrong thing.

The fix is retrieval. A curated table of disease records is the authority on what those words
mean clinically. Before the model sees the question, we look up the closest records and hand
them over as reference material. The model's job shifts from *recall* to *reading
comprehension*, which it is far more reliable at.

**The consequence for you:** retrieval quality is the product. If the right record isn't
retrieved, no amount of prompt engineering saves the answer. When something is wrong, check
retrieval first — that's why `GET /message_handler/search` exists.

---

## 2. The request lifecycle

One request, every file it touches, in order. This is the spine of the codebase — if you
understand this sequence, you understand the project.

```
POST /message_handler/
  │
  │  { sessionid, modelid, messages:[{role:"user", content:"แมวขนร่วงเป็นวง ๆ..."}], parameters }
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ routers/message_handler.py        message_handler()             │
│   HTTP boundary. Validates the body into a RequestBody,          │
│   calls the two steps below, converts exceptions to HTTP 500.    │
│   Contains no business logic — by design.                        │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ core/extract_parameters.py        ExtractParameters.extract_…() │
│   1. Splits messages by role → system prompt / turns / profile  │
│   2. Last user turn becomes `questions`                          │
│   3. ★ Calls VetDocumentSearch.search_documents(questions)       │
│   4. Packs everything into a Chatbot_Messages object             │
└─────────────────────────────────────────────────────────────────┘
  │                                    │
  │                                    ▼  (the retrieval detour)
  │            ┌──────────────────────────────────────────────────┐
  │            │ tools/SearchVecDoc.py     (thin facade, logs)     │
  │            │   └─> tools/ImportDB2Faiss.py  VetFAISS.search…() │
  │            │         a. load FAISS index + TF-IDF vectorizer   │
  │            │         b. vectorize the query                    │
  │            │         c. index.search() → distances + positions │
  │            │         d. map positions → SQLite rows            │
  │            │         e. drop anything past max_distance        │
  │            └──────────────────────────────────────────────────┘
  │                                    │  [{disease, symptoms, cause, treatment, distance}]
  ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ core/generate_response.py         generate_response()           │
│   1. build_messages():                                           │
│        SystemMessage(prompt + retrieved records)                 │
│        + history turns from SQLite                               │
│        + HumanMessage(question)                                  │
│   2. models/llm.py → real ChatOpenAI or the offline stub         │
│   3. llm.invoke(messages)                                        │
│   4. Save the exchange to chat_history                           │
│   5. Formatter/response_formatter.py wraps it OpenAI-style       │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
  { id, object:"chat.completion", choices:[{message:{content, refdata}}],
    vet_doc:[...], mock:false, latency_ms:842 }
```

**Trace it yourself.** Set a breakpoint on `ExtractParameters.extract_parameters` and send the
Swagger example. Ten minutes there is worth more than re-reading this section.

---

## 3. Layers and the dependency rule

```
                    ┌──────────────┐
   HTTP boundary    │   routers/   │   knows about core/, nothing below
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
   Orchestration    │    core/     │   knows about tools/, models/, Formatter/
                    └──────┬───────┘
                           ▼
        ┌──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌───────────┐  ┌──────────┐
   │ tools/  │   │ models/  │   │ prompts/  │  │Formatter/│
   │ (I/O)   │   │  (LLM)   │   │ (strings) │  │ (shapes) │
   └────┬────┘   └──────────┘   └───────────┘  └──────────┘
        ▼
   ┌─────────────────────────────┐
   │ tools/paths.py              │  depends on nothing. The anchor.
   └─────────────────────────────┘
```

**The rule: dependencies point downward only.** `tools/` must never import from `core/` or
`routers/`. If you find yourself wanting to, the logic belongs in `core/`, not in `tools/`.

This is what makes the retrieval layer testable without FastAPI, and swappable without
touching the API. You can `from tools.ImportDB2Faiss import VetFAISS` in a plain script —
`scripts/build_index.py` does exactly that.

---

## 4. File reference

### The entry points

| File | Responsibility | You'll touch it when… |
|---|---|---|
| `src/LLM_Process_API.py` | FastAPI app, CORS, router mounting, `lifespan` startup (builds the index if missing), `/health` | Adding a router, changing startup behaviour, editing the Swagger description |
| `run.sh` | Starts uvicorn from the project root regardless of your cwd | Rarely |
| `scripts/build_index.py` | Seeds `vet_doc` from JSON, rebuilds the FAISS index, runs a smoke query | After changing data or the vectorizer |

### The request path

| File | Responsibility | You'll touch it when… |
|---|---|---|
| `src/routers/message_handler.py` | HTTP endpoints. `POST /` (chat) and `GET /search` (retrieval only) | Adding an endpoint, changing status codes |
| `src/core/request_templates.py` | Pydantic models = the **API contract**. `RequestBody` in, `Chatbot_Messages` internally | Changing the request/response shape. **Breaking change — coordinate with API consumers** |
| `src/core/extract_parameters.py` | Request → internal object. Where retrieval is triggered | Changing how roles are parsed, or what gets retrieved |
| `src/core/generate_response.py` | Prompt assembly + LLM call + history persistence | Changing what the model sees. **The highest-leverage file for answer quality** |
| `src/models/llm.py` | LLM factory. Real `ChatOpenAI` or offline stub, chosen by `use_mock()` | Swapping providers, changing model parameters |
| `src/prompts/prompts.py` | The Thai system prompt and its rules | Tuning tone, adding safety rules |
| `src/Formatter/response_formatter.py` | OpenAI-shaped response envelope | Changing the response JSON |
| `src/Formatter/json_formatter.py` | Error envelope, streaming chunk shapes | Adding streaming |

### The retrieval engine

| File | Responsibility | You'll touch it when… |
|---|---|---|
| `src/tools/ImportDB2Faiss.py` | **The heart.** `VetDB` (SQLite CRUD) + `VetFAISS` (build/search). Both vector operations live here | Changing the embedding method, index type, or ranking |
| `src/tools/SearchVecDoc.py` | Thin facade over `VetFAISS`. Exists for naming compatibility with the original | Almost never — it's 12 lines |

### Infrastructure

| File | Responsibility | You'll touch it when… |
|---|---|---|
| `src/tools/paths.py` | `BASE_DIR` derived from `__file__`. **Every path in the project resolves through here** | Never, ideally |
| `src/tools/config_loader.py` | Reads `config/llm_config.json`, returns absolute paths and tuned values | Adding a config key |
| `src/tools/database_process.py` | Chat history CRUD | Changing history behaviour (e.g. a retention window) |
| `src/core/logger.py` | Rotating file + console logger | Changing log format or level |
| `src/tools/measurellm.py` | Execution timing → `logs/execution_time.log` | Adding metrics |

### Data and config

| Path | What it is |
|---|---|
| `config/llm_config.json` | Host, port, **all paths (relative)**, `top_k`, `max_distance` |
| `data/vet_doc.db` | The knowledge base. 8 records. Source of truth at runtime |
| `data/seed_vet_doc.json` | Human-editable copy of the same data, for rebuilding from scratch |
| `data/chat_history.db` | Created at runtime. Per-`sessionid` conversation log |
| `src/vectordb/*.index`, `*.joblib` | **Build artifacts.** Gitignored. Delete them freely — they regenerate |
| `.env` | Your API key. Gitignored. Never commit |
| `tests/test_api.py` | 4 smoke tests, run without an API key |

---

## 5. Patterns worth copying

These are deliberate. Follow them when you add code.

**Paths resolve from `__file__`, never from cwd or `$HOME`.**
`paths.py` computes `BASE_DIR = Path(__file__).resolve().parents[2]` and everything hangs off
that. This is why the project runs from any folder on any machine. The previous version
hardcoded `/home/siwale/...` and died on a different laptop. *If you ever write a bare
relative path like `open("data/foo.json")`, you've reintroduced the bug* — it resolves against
wherever the process was launched.

**Config holds values; code holds behaviour.**
`top_k` and `max_distance` live in `llm_config.json` because they're tuned, not designed.
Anything you'll want to change without a code review belongs there too.

**One boundary object between layers.**
`Chatbot_Messages` is the only thing that crosses from extraction to generation. Neither side
knows about the other's internals. To pass something new through, add a field there — don't
thread an extra argument through three functions.

**Dependency injection through config, not imports.**
`LLM.model()` returns either a real client or a stub, decided at call time by `use_mock()`.
Nothing downstream knows which it got. This is what makes the whole pipeline testable with no
API key, and it's the pattern to reuse if you add a second provider.

**The facade is allowed to be trivial.**
`SearchVecDoc.py` just calls `VetFAISS` and logs. It stays because callers import that name.
Not every file needs to justify itself by size.

**Escape hatches for debugging are first-class.**
`GET /search` runs retrieval with no LLM call. The chat response carries `vet_doc` and
`latency_ms`. When retrieval is the failure mode, you need to see retrieval in isolation —
build that door on day one, not after the first incident.

---

## 6. Two landmines

Read these before you touch the retrieval code.

### 6.1 The index maps by *position*, not by ID

`build_index()` calls `index.add(vectors)` — no IDs. FAISS returns positions `0..n-1`, and
`search_vet_doc()` maps them with `records[record_pos]`, where `records` came from
`SELECT * FROM vet_doc ORDER BY data_id`.

**This silently breaks if a row is deleted without rebuilding the index.** Positions shift, and
the search starts returning *the wrong record with a confident-looking distance*. No error, no
warning — just wrong answers.

Today this is contained because `lifespan` builds the index at startup and there is no delete
endpoint. If you add one, fix the mapping first:

```python
# in build_index()
index = faiss.IndexIDMap2(faiss.IndexFlatL2(vectors.shape[1]))
index.add_with_ids(vectors, np.array([r["data_id"] for r in records], dtype="int64"))

# in search_vet_doc() — the returned id IS data_id, look it up directly
```

That also removes the need to load every record on every search (see §7).

### 6.2 The column names are shifted

In `vet_doc`, the columns do not mean what they say:

| Column | Actually contains | Example |
|---|---|---|
| `symptoms` | the **disease name** | `โรคเชื้อราในแมว (Ringworm)` |
| `cause` | the **symptom list** | `ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน` |
| `diagnosis` | the **cause** | `เชื้อรากลุ่ม Dermatophytes` |
| `treatment` | the treatment ✓ | |

The original code indexed column 2 (`cause`), which *is* the right text to search — so
behaviour was correct by accident. That's preserved, and search results rename the fields
honestly (`disease`, `symptoms`, `cause`, `treatment`).

`_search_text()` in `ImportDB2Faiss.py` is the single place that decides what gets embedded.
If you ever fix the schema, that's the one function to update.

---

## 7. Optimization: measured, not guessed

Profiled on the current 8-record dataset:

| Operation | Time |
|---|---|
| `build_index()` (8 records) | 7.6 ms |
| **`search_vet_doc()` end to end** | **2.73 ms** |
| ↳ `joblib.load` (vectorizer) | 1.70 ms ← **62% of it** |
| ↳ `tfidf.transform` | 0.36 ms |
| ↳ `SELECT * FROM vet_doc` | 0.26 ms |
| ↳ `faiss.read_index` | 0.03 ms |
| ↳ `index.search` | 0.02 ms |

**The actual FAISS search is 0.7% of the time.** Everything else is reloading state from disk
on every single request. The index and vectorizer are immutable between rebuilds — there is no
reason to read them more than once.

### Fix #1 — cache the index and vectorizer (measured 13× faster)

```python
# in ImportDB2Faiss.py
_CACHE: dict = {}

@staticmethod
def _load(db_name, index_file, vectorizer_file):
    key = (str(index_file), index_file.stat().st_mtime)   # mtime invalidates on rebuild
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = (faiss.read_index(str(index_file)), joblib.load(vectorizer_file))
    return _CACHE[key]
```

Measured: **2.73 ms → 0.213 ms.** Keying on mtime means a rebuild invalidates the cache
automatically, so you don't get stale vectors after running `build_index.py`.

### Fix #2 — stop loading every record on every search

`search_vet_doc()` calls `get_all_vet_doc()` to map positions back to rows. That's `SELECT *`
over the whole table to retrieve 2 records:

| Rows in `vet_doc` | `SELECT *` (today) | `SELECT … WHERE data_id IN (…)` |
|---|---|---|
| 8 | 0.02 ms | 0.006 ms |
| 1,000 | 2.13 ms | 0.006 ms |
| 10,000 | **21.42 ms** | **0.006 ms** |

Linear growth versus flat. Fixing landmine §6.1 fixes this at the same time: with
`IndexIDMap2`, FAISS hands you `data_id` directly, so you fetch only the rows you need.

**Do these two together.** They're the same 20-line change, and they take the service from
"fine at 8 records" to "fine at 100,000".

### What not to optimize

`build_index()` at 7.6 ms and `fit_transform` at 386 ms for 5,000 documents are offline,
one-time costs. Leave them alone. And once the LLM call is real, it dominates everything —
a live `gpt-4o` round trip is 500–2000 ms, so retrieval is noise by comparison. **Optimize
retrieval for scale, not for latency.**

---

## 8. Improving answer quality

In priority order:

**1. Add more records.** Eight diseases is a demo. This is by far the highest-leverage change
and needs no code — edit `data/seed_vet_doc.json`, then:

```bash
python scripts/build_index.py --reseed
```

**2. Re-tune `max_distance` after adding records.** It's currently `1.5`, chosen from measured
spread: correct matches land at 0.50–1.22, wrong ones at 1.70+. More records narrow that gap.
Check with `GET /search?q=...&threshold=-1` to see scores that are normally filtered out.
Too high → irrelevant records poison the prompt. Too low → the model gets nothing and says
"I don't know".

**3. Move from TF-IDF to embeddings.** TF-IDF matches *characters*. A user writing
`ขนหลุดเป็นหย่อม` when the record says `ขนร่วงเป็นวง ๆ` shares few n-grams and may miss
entirely — same meaning, different characters. Semantic embeddings (OpenAI
`text-embedding-3-small`) fix that class of miss. Only `build_index()` and `search_vet_doc()`
change; everything above them is untouched. Note the trade-offs: a network call per query,
cost per token, and switch `IndexFlatL2` to `IndexFlatIP` over normalized vectors so the score
is cosine similarity.

**4. Tune the prompt.** `prompts.py` and `build_messages()`. Cheapest to try, smallest effect —
do it after retrieval is solid.

---

## 9. Where do I change X?

| I want to… | Edit |
|---|---|
| Add disease records | `data/seed_vet_doc.json` → `python scripts/build_index.py --reseed` |
| Change how many results are retrieved | `config/llm_config.json` → `retrieval.top_k` |
| Make retrieval stricter or looser | `config/llm_config.json` → `retrieval.max_distance` |
| Change what text gets embedded | `ImportDB2Faiss.py` → `_search_text()` |
| Change the embedding method | `ImportDB2Faiss.py` → `build_index()` **and** `search_vet_doc()` |
| Change the system prompt | `prompts.py` → `DEFAULT_SYSTEM_PROMPT` |
| Change what the model actually receives | `generate_response.py` → `build_messages()` |
| Change the response JSON | `Formatter/response_formatter.py` |
| Change the request shape | `core/request_templates.py` (**breaking change**) |
| Add an endpoint | `routers/message_handler.py` |
| Swap LLM provider | `models/llm.py` → `LLM.model()` |
| Change host/port | `config/llm_config.json` → `general_setting` |
| Add a config value | `config/llm_config.json` + a getter in `config_loader.py` |
| Add product recommendations | New table + lookup in `extract_parameters.py`, render in `build_messages()` |
| Enable streaming | `message_handler.py` → `StreamingResponse`; the chunk formatters already exist |

---

## 10. Working on it

```bash
source .venv/bin/activate         # Python 3.11 — 3.9 will not work
pytest -q                         # 4 tests, no API key needed
./run.sh                          # http://127.0.0.1:8000/docs
```

**The loop that catches most mistakes:**

```bash
# 1. retrieval alone — free, instant, no LLM
curl -sG http://127.0.0.1:8000/message_handler/search \
  --data-urlencode "q=แมวขนร่วงเป็นวง ๆ ผิวหนังแดง คัน" | python3 -m json.tool

# 2. full pipeline with the stub LLM (USE_MOCK_LLM=1 in .env)
#    verifies wiring and the response contract without spending tokens

# 3. only then, with a real key
```

`pytest` before every commit. The tests run in mock mode, so they're fast and free — there's
no excuse to skip them.

**Debugging by symptom:**

| Symptom | Look at |
|---|---|
| Answer is wrong or generic | `GET /search` first. If the right record isn't retrieved, it's a retrieval problem, not a prompt problem |
| `vet_doc: []` in the response | Nothing cleared `max_distance`. Re-check with `threshold=-1` |
| Retrieval returns the wrong record confidently | §6.1 — rebuild the index |
| `ModuleNotFoundError: faiss` | Wrong venv, or Python 3.9 |
| Path-related `FileNotFoundError` | Someone wrote a relative path instead of going through `paths.py` |

---

## 11. Known debt

Honest list, roughly by priority:

1. **Position-based index mapping** (§6.1) — a correctness bug waiting for a delete endpoint
2. **State reloaded from disk on every search** (§7) — 13× slower than it needs to be
3. **`SELECT *` on every search** (§7) — linear in table size
4. **`parameters.stream` is accepted and ignored** — the formatters exist, the wiring doesn't
5. **No auth** — anyone who can reach the port can spend your OpenAI budget
6. **Chat history grows unbounded** — no retention policy, no cleanup
7. **`refdata` is overloaded** — caller-supplied on input, retrieved records on output
8. **Shifted column names** (§6.2) — works, but every new reader loses ten minutes to it
9. **Only 4 tests** — no coverage of retrieval quality; a regression in ranking is invisible

Number 9 is the one that will bite you silently. A small CSV of *query → expected disease*
asserted in CI would turn retrieval quality from a vibe into a number.
