"""Chat endpoints.

The original never returned anything: it called ``extract_parameters``, assigned
``prompt_type``, and left the ``StreamingResponse(...)`` line commented out, so
``POST /message_handler/`` answered ``null`` on every request. That is fixed here.

Three endpoints, in increasing cost:

    GET  /search    retrieval only            free, no LLM call
    POST /preview   full assembled prompt     free, no LLM call
    POST /          the real thing            costs an LLM call
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from core.extract_parameters import ExtractParameters
from core.generate_response import build_messages, describe_messages, generate_response
from core.request_templates import RequestBody
from Formatter.json_formatter import JSONMessage
from tools.SearchVecDoc import VetDocumentSearch

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", summary="Ask about cat symptoms (retrieval + history + LLM)")
async def message_handler(
    requestBody: RequestBody,
    debug: bool = Query(
        False,
        description=(
            "Add `prompt_sent` to the response: every message handed to the LLM, "
            "in order — system prompt with the retrieved records, then the chat "
            "history for this sessionid, then the new question."
        ),
    ),
):
    try:
        chatbot_messages = ExtractParameters.extract_parameters(requestBody)
        return await generate_response(chatbot_messages, prompt_type="chatbot", debug=debug)
    except Exception as e:
        logger.exception("Error in /message_handler/")
        raise HTTPException(status_code=500, detail=JSONMessage.error_response(e)) from e


@router.post(
    "/preview",
    summary="See the exact prompt WITHOUT calling the LLM (free)",
    description=(
        "Runs retrieval and assembles the full prompt — system message with the "
        "injected vet_doc records, the stored chat history for this `sessionid`, "
        "and the new question — then returns it instead of sending it.\n\n"
        "Use this to verify history injection and prompt assembly without spending "
        "tokens. Nothing is written to chat_history."
    ),
)
async def preview_prompt(requestBody: RequestBody):
    try:
        chatbot_messages = ExtractParameters.extract_parameters(requestBody)
        messages = build_messages(chatbot_messages)
        described = describe_messages(messages)
        return {
            "sessionid": requestBody.sessionid,
            "history_turns": (len(messages) - 2) // 2,
            "vet_doc_injected": len(chatbot_messages.vet_doc),
            "total_chars": sum(m["chars"] for m in described),
            "prompt_sent": described,
            "note": (
                "Nothing was sent to the LLM and nothing was saved. "
                "history_turns counts stored exchanges for this sessionid."
            ),
        }
    except Exception as e:
        logger.exception("Error in /message_handler/preview")
        raise HTTPException(status_code=500, detail=JSONMessage.error_response(e)) from e


@router.get("/search", summary="Run retrieval only (no LLM call)")
async def search_only(q: str, top_k: int = 3):
    """Cheap way to inspect what the retriever returns before spending tokens."""
    return {"query": q, "results": VetDocumentSearch.search_documents(q, top_k=top_k)}


@router.get("/history/{session_id}", summary="Inspect stored chat history for a session")
async def get_history(session_id: str, limit: int = 20):
    """What build_messages() will inject as conversation turns for this session."""
    from tools.database_process import DatabaseProcess

    turns = DatabaseProcess.get_chat_history(session_id, limit=limit)
    return {
        "sessionid": session_id,
        "turns": len(turns),
        "history": [{"user": u, "assistant": a} for u, a in turns],
    }
