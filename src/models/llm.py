"""LLM factory.

Two changes from the original:

* **No API key in source.** The original hardcoded a live key with
  ``os.environ["OPENAI_API_KEY"] = "sk-proj-..."``. The key is now read from the
  environment / .env only.
* ``from langchain_openai import ChatOpenAI`` instead of the deprecated
  ``langchain.chat_models`` path, which is removed in langchain 0.3.

Set ``USE_MOCK_LLM=1`` to run the whole pipeline without an OpenAI key — useful
for testing retrieval and the response contract offline.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_ID = "gpt-4o"


class _MockResponse:
    def __init__(self, content: str):
        self.content = content


class _MockChatModel:
    """Offline stand-in that echoes the retrieved context back."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, messages):
        payload = messages[-1].content if messages else ""
        return _MockResponse(
            "[MOCK LLM — ไม่ได้เรียก OpenAI จริง]\n"
            f"model={self.model_name}\n"
            f"{payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)}"
        )


def use_mock() -> bool:
    if os.environ.get("USE_MOCK_LLM", "").strip() in {"1", "true", "True"}:
        return True
    return not os.environ.get("OPENAI_API_KEY", "").strip()


class LLM:
    @staticmethod
    def model(parameter=None, model_name: str = DEFAULT_MODEL_ID):
        model_name = model_name or DEFAULT_MODEL_ID

        if use_mock():
            return _MockChatModel(model_name)

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=getattr(parameter, "temperature", 0.1),
            top_p=getattr(parameter, "top_p", 0.1),
            max_tokens=getattr(parameter, "max_tokens", 4096),
        )
