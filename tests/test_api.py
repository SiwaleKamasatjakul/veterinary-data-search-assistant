"""Smoke tests — run with `pytest -q`. No OpenAI key needed (stub LLM)."""

import os
import sys
from pathlib import Path

os.environ["USE_MOCK_LLM"] = "1"

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from LLM_Process_API import app  # noqa: E402

PAYLOAD = {
    "requestid": "",
    "sessionid": "test-001",
    "modelid": "gpt-4o",
    "messages": [
        {"role": "user", "content": "แมวขนร่วงเป็นวง ๆ ผิวหนังแดง คัน มีสะเก็ดเป็นขุย"}
    ],
    "parameters": {"stream": False, "temperature": 0.1, "max_tokens": 4096, "top_p": 0.1},
    "refdata": [],
}


def test_health():
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["vet_doc_records"] > 0
        assert body["index_exists"] is True


def test_search_finds_ringworm():
    with TestClient(app) as client:
        res = client.get(
            "/message_handler/search",
            params={"q": "ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน", "top_k": 3},
        ).json()
        assert res["results"], "FAISS returned nothing"
        assert "Ringworm" in res["results"][0]["disease"]


def test_chat_returns_openai_shaped_response():
    with TestClient(app) as client:
        res = client.post("/message_handler/", json=PAYLOAD)
        assert res.status_code == 200
        body = res.json()
        assert body["object"] == "chat.completion"
        assert body["id"], "blank requestid should be filled in"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"]
        assert body["vet_doc"], "retrieved records should be attached"


def test_chat_history_is_persisted():
    with TestClient(app) as client:
        client.post("/message_handler/", json={**PAYLOAD, "sessionid": "test-history"})
        client.post("/message_handler/", json={**PAYLOAD, "sessionid": "test-history"})

    from tools.database_process import DatabaseProcess

    history = DatabaseProcess.get_chat_history("test-history")
    assert len(history) >= 2


def test_search_returns_cosine_scores():
    """Scores are cosine similarity in -1..1, higher is better."""
    with TestClient(app) as client:
        res = client.get(
            "/message_handler/search",
            params={"q": "ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน", "top_k": 3},
        ).json()
        assert res["results"]
        scores = [r["score"] for r in res["results"]]
        assert all(-1.0 <= s <= 1.0 for s in scores), scores
        assert scores == sorted(scores, reverse=True), "results must be best-first"


def test_index_maps_by_data_id_not_position():
    """Regression guard for ARCHITECTURE.md 6.1."""
    from tools.ImportDB2Faiss import VetFAISS

    hits = VetFAISS.search_vet_doc("ขนร่วงเป็นวง ๆ ผิวหนังแดง คัน", top_k=1, min_similarity=-1)
    assert hits and "Ringworm" in hits[0]["disease"]
    # data_id comes straight from FAISS, so it must be a real row id
    assert hits[0]["data_id"] > 0


def test_backend_is_reported():
    from tools.ImportDB2Faiss import VetFAISS

    info = VetFAISS.index_info()
    assert info.get("backend") in {"tfidf", "openai"}
    assert info.get("dim", 0) > 0


def test_openai_backend_fails_loudly_without_key(monkeypatch):
    """Switching backend with no key must raise a clear error, not a stack trace."""
    from tools.embedders import OpenAIEmbedder

    monkeypatch.setenv("OPENAI_API_KEY", "")
    embedder = OpenAIEmbedder()
    try:
        _ = embedder.client
    except RuntimeError as e:
        assert "OPENAI_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError when no key is set")


def test_env_loads_from_project_root_not_cwd(tmp_path, monkeypatch):
    """Regression: scripts run standalone must still see .env.

    load_dotenv() was only called in models/llm.py, which scripts/ never
    imports — so `check_retrieval.py --backend openai` failed with
    "OPENAI_API_KEY is not set" even though the key was in .env.
    """
    from tools.paths import BASE_DIR, ENV_FILE, load_env

    assert ENV_FILE == BASE_DIR / ".env", "env file must be anchored to the project root"

    env_existed = ENV_FILE.exists()
    original = ENV_FILE.read_text(encoding="utf-8") if env_existed else None
    try:
        ENV_FILE.write_text("PROJECT_ROOT_ENV_PROBE=loaded\n", encoding="utf-8")
        monkeypatch.delenv("PROJECT_ROOT_ENV_PROBE", raising=False)
        monkeypatch.chdir(tmp_path)          # cwd far from the project
        assert load_env() is True
        assert os.environ.get("PROJECT_ROOT_ENV_PROBE") == "loaded"
    finally:
        if original is not None:
            ENV_FILE.write_text(original, encoding="utf-8")
        elif ENV_FILE.exists():
            ENV_FILE.unlink()


def test_debug_flag_exposes_full_prompt():
    """?debug=true must show system + history turns + new question, in order."""
    with TestClient(app) as client:
        sid = "test-debug-prompt"
        first = {**PAYLOAD, "sessionid": sid}
        client.post("/message_handler/", json=first)          # seed one turn
        res = client.post("/message_handler/?debug=true", json=first).json()

        assert "prompt_sent" in res
        roles = [m["role"] for m in res["prompt_sent"]]
        assert roles[0] == "system", roles
        assert roles[-1] == "user", roles
        assert "assistant" in roles, "history turns must be injected"
        assert res["history_turns"] >= 1


def test_preview_does_not_call_llm_or_save():
    """/preview shows the prompt without spending a call or writing history."""
    from tools.database_process import DatabaseProcess

    with TestClient(app) as client:
        sid = "test-preview-no-write"
        before = len(DatabaseProcess.get_chat_history(sid))
        res = client.post("/message_handler/preview", json={**PAYLOAD, "sessionid": sid})
        assert res.status_code == 200
        body = res.json()
        assert body["prompt_sent"][0]["role"] == "system"
        assert body["total_chars"] > 0
        assert "choices" not in body, "preview must not produce an answer"
        assert len(DatabaseProcess.get_chat_history(sid)) == before, "preview wrote history"


def test_history_endpoint_matches_what_gets_injected():
    with TestClient(app) as client:
        sid = "test-history-endpoint"
        client.post("/message_handler/", json={**PAYLOAD, "sessionid": sid})
        hist = client.get(f"/message_handler/history/{sid}").json()
        preview = client.post(
            "/message_handler/preview", json={**PAYLOAD, "sessionid": sid}
        ).json()
        assert hist["turns"] == preview["history_turns"]
