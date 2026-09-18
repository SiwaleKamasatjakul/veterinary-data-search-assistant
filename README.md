# Veterinary Cat Store Chatbot — macOS

A macOS-ready rebuild of `ChatbotCatStoreLinux`. Same architecture, same API contract,
same knowledge base — with the portability blockers and several latent bugs fixed.

```
user question
   └─> TF-IDF vector (char n-grams, Thai-friendly)
        └─> FAISS IndexFlatL2 over the vet_doc knowledge base
             └─> matched disease records
                  └─> injected into the system prompt as clinical reference
                       └─> LLM answer (+ saved to chat history)
```

---

> **New to this codebase?** Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it maps the
> request lifecycle, explains what every file does, and lists the two landmines in the
> retrieval layer.

## Setup

Requires **Python 3.11**. macOS ships 3.9, which has no arm64 wheels for faiss/numpy/scipy.

```bash
brew install python@3.11          # skip if you already have it

cd ChatbotCatStoreMac
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
| GET | `/message_handler/search?q=...` | FAISS retrieval only, no LLM call. Tune here for free |
| GET | `/health` | Status |

Verified response (offline mode):

```json
{
  "id": "f30416b1-…",
  "object": "chat.completion",
  "choices": [{ "message": { "role": "assistant", "content": "…", "refdata": [...] } }],
  "vet_doc": [{ "disease": "โรคเชื้อราในแมว (Ringworm)", "distance": 0.677 }],
  "mock": true,
  "latency_ms": 6
}
```

## Tests

```bash
pytest -q        # 4 passed — no API key required
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
compared against a threshold as if larger were better. Now: bounds checked first,
`distance <= max_distance` second.

**`chat_history.db` was a relative path.** `sqlite3.connect("chat_history.db")` created a
stray empty database in whatever directory you launched from. Now absolute, from config.

**The error handler raised its own error.** `JSONMessage.error_response` read `e.message`,
`e.param` and `e.code`, none of which exist on a plain Exception — so the except branch
raised AttributeError and masked the real failure. Now uses `getattr` fallbacks.

**`get_general_config` read the wrong key.** It looked up `general` but the JSON key is
`general_setting`, so host and port always silently fell through to the defaults.

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

Measured on the 8 seeded records, top-1 is correct on every test query, and the gap between
a real match and the next candidate is wide enough to threshold on:

| query | top hit | distance | runner-up |
|---|---|---|---|
| แมวน้ำมูกไหล จาม ตาแดง มีไข้ | โรคหวัดแมว | 0.50 | 1.89 |
| แมวขนร่วงเป็นวง ๆ ผิวหนังแดง คัน | Ringworm | 0.89 | 1.77 |
| แมวอาเจียน ถ่ายเหลว ท้องอืด | พยาธิในทางเดินอาหาร | 0.98 | 1.70 |
| แมวกินน้ำเยอะ ฉี่บ่อย น้ำหนักลดแต่กินเยอะ | เบาหวานในแมว | 1.09 | 1.74 |
| แมวไม่กินข้าว ตัวเหลือง อาเจียน | Hepatic Lipidosis | 1.22 | 1.70 |

`max_distance` in `config/llm_config.json` is set to **1.5** from that spread. Re-check it
if you add records; the runner-up column is what it has to stay below.

### A note on the data

In `vet_doc`, the `symptoms` column actually holds the **disease name**, `cause` holds the
**symptom list**, and `diagnosis` holds the **cause**. The original indexed column 2
(`cause`), which is the correct text to search on — so the behaviour was right even though
the labels are shifted. That indexing behaviour is preserved, and search results expose the
fields under honest names (`disease`, `symptoms`, `cause`, `treatment`). If you ever fix the
column names in the schema, `_search_text()` in `ImportDB2Faiss.py` is the one place to update.

## Layout

```
ChatbotCatStoreMac/
├── run.sh                      start the API from anywhere
├── requirements.txt            Python 3.11, arm64 wheels only
├── .env.example                copy to .env
├── config/llm_config.json      all paths relative to the project root
├── data/
│   ├── vet_doc.db              knowledge base (8 records)
│   ├── seed_vet_doc.json       source of truth for rebuilding the DB
│   └── chat_history.db         created at runtime
├── scripts/build_index.py      seed + rebuild FAISS index
├── src/
│   ├── LLM_Process_API.py      FastAPI app
│   ├── core/                   request models, extraction, generation, logging
│   ├── Formatter/              response envelopes
│   ├── models/llm.py           LLM factory (real / stub)
│   ├── prompts/prompts.py      system prompt
│   ├── routers/                chat + search endpoints
│   ├── tools/                  paths, config, sqlite, TF-IDF + FAISS
│   └── vectordb/               generated index (gitignored)
└── tests/test_api.py
```

## Possible next steps

- **Swap TF-IDF for OpenAI embeddings.** `ImportDB2Faiss.build_index` and `search_vet_doc`
  are the only two places that touch vectors. Semantic embeddings would match paraphrases
  that share no characters with the stored text — the current ceiling.
- **Streaming.** `parameters.stream` is accepted but ignored. Wire it to `StreamingResponse`
  plus `llm.stream()`; `ResponseFormatter._format_stream_response` is already written.
- **Product recommendation.** Add a `products` table keyed by disease, look it up alongside
  the FAISS hit, and extend the context block in `generate_response.build_messages`.
