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
