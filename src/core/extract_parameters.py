"""Turn an incoming RequestBody into the Chatbot_Messages the generator consumes.

Same shape as the original, minus the debug print() storm (it now logs), and the
retrieved vet documents are carried on their own field as well as being folded
into the context string.
"""

import json
import logging
from typing import List

from core.request_templates import Chatbot_Messages, RequestBody
from tools.SearchVecDoc import VetDocumentSearch

logger = logging.getLogger(__name__)


class ExtractParameters:
    @staticmethod
    def _process_messages(messages):
        allowed_roles = ["user", "assistant"]
        system_prompt = ""
        request_messages_list = []
        context = []

        for message in messages or []:
            if message.role == "userprofile":
                context.append(json.dumps(message.detail, indent=4, ensure_ascii=False))
            elif message.role in allowed_roles:
                request_messages_list.append({message.role: message.content})
            elif message.role == "system":
                system_prompt += f"\n{message.content}"

        return system_prompt, request_messages_list, context

    @staticmethod
    def _extract_questions_and_history(request_messages_list: List[dict]):
        if not request_messages_list:
            return "", ""
        questions = request_messages_list[-1].get("user", "")
        history = "\n".join(str(request) for request in request_messages_list[:-1])
        return questions, history

    @staticmethod
    def extract_parameters(body: RequestBody) -> Chatbot_Messages:
        parameters = body.parameters
        system_prompt, request_messages_list, context = ExtractParameters._process_messages(
            body.messages
        )
        questions, history = ExtractParameters._extract_questions_and_history(
            request_messages_list
        )

        # FAISS retrieval: the whole point of the pipeline.
        vet_doc = VetDocumentSearch.search_documents(questions)
        if vet_doc:
            context.append(json.dumps(vet_doc, indent=2, ensure_ascii=False))

        logger.info("question=%r history_turns=%d vet_hits=%d",
                    questions, len(request_messages_list) - 1, len(vet_doc))

        return Chatbot_Messages(
            system_prompt=system_prompt,
            stream=bool(parameters.stream),
            questions=questions,
            parameters=parameters,
            model_name=body.modelid or "gpt-4o",
            request_id=body.requestid,
            session_id=body.sessionid,
            history=history,
            context="\n".join(context),
            refdata=body.refdata or [],
            vet_doc=vet_doc,
        )
