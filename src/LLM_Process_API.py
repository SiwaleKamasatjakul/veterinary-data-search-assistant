"""Veterinary cat-store chatbot API — macOS port.

Run from the project root:

    uvicorn src.LLM_Process_API:app --reload --app-dir .

or simply:

    ./run.sh

Swagger UI: http://127.0.0.1:8000/docs
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make `src` importable whether the app is started from the project root or from
# inside src/ — the original only worked when launched from src/.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from core.logger import CustomLogger  # noqa: E402
from models.llm import use_mock  # noqa: E402
from routers import message_handler  # noqa: E402
from tools.config_loader import ConfigManager  # noqa: E402
from tools.ImportDB2Faiss import VetDB, VetFAISS  # noqa: E402

config = ConfigManager.load_config()
host, port = ConfigManager.get_general_config(config)

logger = CustomLogger(__name__).get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ConfigManager.get_config_database()
    VetDB.create_tables(cfg["DB_NAME"])
    if not cfg["FAISS_INDEX_FILE"].exists():
        logger.info("FAISS index not found — building it now")
        VetFAISS.build_index(cfg["DB_NAME"])
    records = len(VetDB.get_all_vet_doc(cfg["DB_NAME"]))
    logger.info("Ready — mock_llm=%s vet_doc_records=%d", use_mock(), records)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Veterinary Cat Store Chatbot",
    description=(
        "Symptom question → TF-IDF vector → FAISS similarity search over the vet_doc "
        "knowledge base → matched records injected into the prompt → LLM answer.\n\n"
        "Leave `OPENAI_API_KEY` unset (or set `USE_MOCK_LLM=1`) to run offline with a "
        "stub model — retrieval and the response contract stay fully testable."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(message_handler.router, prefix="/message_handler", tags=["chat"])


@app.get("/health", tags=["system"])
def health() -> dict:
    cfg = ConfigManager.get_config_database()
    return {
        "status": "ok",
        "mock_llm": use_mock(),
        "vet_doc_records": len(VetDB.get_all_vet_doc(cfg["DB_NAME"])),
        "index_exists": cfg["FAISS_INDEX_FILE"].exists(),
        "base_dir": str(ConfigManager.get_base_dir()),
    }


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "veterinary-cat-store-chatbot", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)
