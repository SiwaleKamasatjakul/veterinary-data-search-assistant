"""Chat endpoint.

The original never returned anything: it called ``extract_parameters``, assigned
``prompt_type``, and left the ``StreamingResponse(...)`` line commented out, so
``POST /message_handler/`` answered ``null`` on every request. That is fixed here.
"""

import logging

from fastapi import APIRouter, HTTPException

from core.extract_parameters import ExtractParameters
from core.generate_response import generate_response
from core.request_templates import RequestBody
from Formatter.json_formatter import JSONMessage
from tools.SearchVecDoc import VetDocumentSearch

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", summary="Ask about cat symptoms (FAISS retrieval + LLM)")
async def message_handler(requestBody: RequestBody):
    try:
        chatbot_messages = ExtractParameters.extract_parameters(requestBody)
        return await generate_response(chatbot_messages, prompt_type="chatbot")
    except Exception as e:
        logger.exception("Error in /message_handler/")
        raise HTTPException(status_code=500, detail=JSONMessage.error_response(e)) from e


@router.get("/search", summary="Run FAISS retrieval only (no LLM call)")
async def search_only(q: str, top_k: int = 3):
    """Cheap way to inspect what the retriever returns before spending tokens."""
    return {"query": q, "results": VetDocumentSearch.search_documents(q, top_k=top_k)}
