"""Error / chunk envelopes.

Fix from the original: ``error_response`` read ``e.message``, ``e.param`` and
``e.code``, none of which exist on a plain Exception — so the except branch that
was supposed to report an error raised AttributeError instead, masking the real
failure. It now falls back safely via getattr.
"""

import time


class JSONMessage:
    @staticmethod
    def error_response(e: Exception) -> dict:
        return {
            "error": {
                "message": getattr(e, "message", None) or str(e),
                "type": type(e).__name__,
                "param": getattr(e, "param", None),
                "code": getattr(e, "code", None),
            }
        }

    @staticmethod
    def chunk_response(id, model_name: str, chunk: str) -> dict:
        return {
            "id": id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": chunk},
                    "finish_reason": None,
                }
            ],
        }

    @staticmethod
    def chunk_completion(id, model_name: str) -> dict:
        return {
            "id": id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
