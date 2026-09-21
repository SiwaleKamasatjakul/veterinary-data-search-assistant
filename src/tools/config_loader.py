"""Configuration loading.

Changes from the Linux original:
* No hardcoded ``/home/siwale/...`` default — the config path is derived from
  the project root, so the app starts wherever the folder lives.
* ``get_general_config`` reads the ``general_setting`` key that actually exists
  in llm_config.json (the original read ``general`` and silently fell through to
  its defaults).
* All path values are returned absolute, resolved against the project root.
"""

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from tools.paths import BASE_DIR, CONFIG_DIR, resolve

DEFAULT_CONFIG_PATH = CONFIG_DIR / "llm_config.json"


class ConfigManager:

    @staticmethod
    @lru_cache(maxsize=4)
    def _read(config_path: str) -> dict:
        path = Path(config_path)
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            print(f"Configuration file {path} not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the configuration file {path}.")
            raise

    @staticmethod
    def load_config(config_path=None) -> dict:
        """Load llm_config.json. Override with the CHATBOT_CONFIG env var if needed."""
        config_path = config_path or os.environ.get("CHATBOT_CONFIG") or DEFAULT_CONFIG_PATH
        return ConfigManager._read(str(config_path))

    @staticmethod
    def get_general_config(config=None):
        """Return (host, port) from the general_setting block."""
        config = config if config is not None else ConfigManager.load_config()
        general_config = config.get("general_setting", {})
        host = general_config.get("host", "127.0.0.1")
        port = int(general_config.get("port", 8000))
        return host, port

    @staticmethod
    def get_log_dir() -> Path:
        config = ConfigManager.load_config()
        log_dir = config.get("paths", {}).get("log_file_path", "logs")
        return resolve(log_dir)

    @staticmethod
    def get_log_file_path() -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return ConfigManager.get_log_dir() / f"LLM_Processing_Manager_APIs_{date_str}.log"

    @staticmethod
    def get_config_database() -> dict:
        """Absolute paths for every data artifact the app reads or writes."""
        config = ConfigManager.load_config()
        return {
            "DB_NAME": resolve(config.get("DB_NAME", "data/vet_doc.db")),
            "CHAT_HISTORY_DB": resolve(config.get("CHAT_HISTORY_DB", "data/chat_history.db")),
            "EMBEDDING_CACHE": resolve(
                config.get("EMBEDDING_CACHE", "data/embedding_cache.db")
            ),
            "FAISS_INDEX_FILE": resolve(
                config.get("FAISS_INDEX_FILE", "src/vectordb/vet_doc.index")
            ),
            "EMBEDDER_FILE": resolve(
                config.get("EMBEDDER_FILE", "src/vectordb/vet_doc_embedder.joblib")
            ),
            "INDEX_META_FILE": resolve(
                config.get("INDEX_META_FILE", "src/vectordb/index_meta.json")
            ),
        }

    @staticmethod
    def get_retrieval_config() -> dict:
        """Retrieval settings.

        ``min_similarity`` may be a single number or a per-backend mapping; the
        value for the active backend is resolved here so callers get a float.
        Scores are cosine similarity in -1..1 — **higher is better**.

        The RETRIEVAL_BACKEND env var overrides the config file, which makes it
        easy to A/B the two backends without editing anything.
        """
        import os

        config = ConfigManager.load_config()
        block = config.get("retrieval", {})

        backend = (os.environ.get("RETRIEVAL_BACKEND") or block.get("backend") or "tfidf").lower()

        threshold = block.get("min_similarity", 0.2)
        if isinstance(threshold, dict):
            threshold = threshold.get(backend, 0.2)

        return {
            "backend": backend,
            "embedding_model": block.get("embedding_model", "text-embedding-3-small"),
            "top_k": int(block.get("top_k", 2)),
            "min_similarity": float(threshold),
        }

    @staticmethod
    def get_base_dir() -> Path:
        return BASE_DIR
