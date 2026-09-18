"""Prompt assembly + LLM call.

Changes from the original:
* The prompt now actually carries the retrieved context. The original built a
  generic system prompt and passed the whole ``input_data`` dict as the human
  message, so the FAISS results never reached the model in usable form.
* Chat history is loaded from SQLite and fed to the model as real turns instead
  of being assembled and then dropped.
* It is a plain coroutine returning a dict, not an async generator. The router
  never consumed the generator, so nothing was ever sent to the client.
* ``uuid4`` fills in a blank requestid so every response is traceable.
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.request_templates import Chatbot_Messages
from Formatter.response_formatter import ResponseFormatter
from models.llm import LLM, use_mock
from prompts.prompts import DEFAULT_SYSTEM_PROMPT
from tools.database_process import DatabaseProcess
from tools.measurellm import MeasureTimePerformance

logger = logging.getLogger(__name__)

NO_MATCH = "(ไม่พบข้อมูลอ้างอิงที่ตรงกับอาการนี้ในฐานข้อมูล — ให้ถามข้อมูลเพิ่มเติมจากผู้ใช้)"


def build_messages(message: Chatbot_Messages):
    """System prompt (+ retrieved context) → history turns → the new question."""
    system_prompt = (message.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    context = (message.context or "").strip() or NO_MATCH
    system_content = f"{system_prompt}\n\nข้อมูลอ้างอิงทางคลินิก:\n{context}"

    messages = [SystemMessage(content=system_content)]

    for user_msg, ai_msg in DatabaseProcess.get_chat_history(message.session_id):
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))

    messages.append(HumanMessage(content=message.questions.strip()))
    return messages


async def generate_response(message: Chatbot_Messages, prompt_type: str = "chatbot") -> dict:
    timer = MeasureTimePerformance().begin_process_time()

    DatabaseProcess.init_database()
    llm_model = LLM.model(message.parameters, message.model_name)
    messages = build_messages(message)

    logger.debug("Prompt: %s", messages[0].content[:400])
    response = llm_model.invoke(messages)
    answer = response.content

    DatabaseProcess.save_message(
        session_id=message.session_id,
        user_message=message.questions,
        ai_response=answer,
    )

    timer.end_process_time()

    payload = ResponseFormatter.format_response(
        id=message.request_id or str(uuid.uuid4()),
        model_name=message.model_name,
        chunk=answer,
        refdata=message.refdata or message.vet_doc,
        stream=False,
    )
    payload["mock"] = use_mock()
    payload["latency_ms"] = int(timer.get_execution_time() * 1000)
    payload["vet_doc"] = message.vet_doc
    return payload
