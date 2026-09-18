"""Request / response models — the wire contract is unchanged from the original."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MessageContent(BaseModel):
    role: Optional[str] = ""
    type: Optional[str] = ""
    content: Optional[str] = ""
    detail: Optional[dict] = {}
    image_url: Optional[dict] = {}


class Parameters(BaseModel):
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = 0.1
    frequency_penalty: Optional[float] = 1
    presence_penalty: Optional[float] = 0
    other_model_specific_parameters: Optional[str] = ""


class RequestBody(BaseModel):
    requestid: str = ""
    sessionid: str = ""
    modelid: Optional[str] = "gpt-4o"
    messages: Optional[List[MessageContent]] = []
    parameters: Optional[Parameters] = Parameters()
    refdata: Optional[list] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "requestid": "",
                    "sessionid": "001",
                    "modelid": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": "แมวขนร่วงเป็นวง ๆ ผิวหนังแดง คัน มีสะเก็ดเป็นขุย",
                        }
                    ],
                    "parameters": {
                        "stream": False,
                        "temperature": 0.1,
                        "max_tokens": 4096,
                        "top_p": 0.1,
                        "frequency_penalty": 1,
                        "presence_penalty": 0,
                        "other_model_specific_parameters": "",
                    },
                    "refdata": [],
                }
            ]
        }
    }


class Chatbot_Messages(BaseModel):
    system_prompt: str
    questions: str
    stream: bool
    parameters: Parameters
    model_name: str
    history: str
    request_id: str
    session_id: str
    context: str = ""
    refdata: list = []
    vet_doc: List[Dict[str, Any]] = []
