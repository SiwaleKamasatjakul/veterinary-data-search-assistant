"""OpenAI-shaped response envelope — identical contract to the original.

The formatters now return dicts; the router serialises once. The original
returned a JSON string that FastAPI then re-encoded, producing a quoted string
instead of a JSON object.
"""

import time


class ResponseFormatter:
    @staticmethod
    def format_response(id, model_name, refdata=None, chunk="", first_chunk=False, stream=False):
        if stream:
            return ResponseFormatter._format_stream_response(
                id, model_name, refdata, chunk, first_chunk
            )
        return ResponseFormatter._format_non_stream_response(id, model_name, refdata, chunk)

    @staticmethod
    def _format_stream_response(id, model_name, refdata, chunk, first_chunk):
        delta = (
            {"content": "", "refdata": refdata or []}
            if first_chunk
            else {"role": "assistant", "content": chunk}
        )
        return {
            "id": id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }

    @staticmethod
    def _format_non_stream_response(id, model_name, refdata, chunk):
        return {
            "id": id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": chunk,
                        "refdata": refdata or [],
                    },
                    "finish_reason": "stop",
                }
            ],
        }
