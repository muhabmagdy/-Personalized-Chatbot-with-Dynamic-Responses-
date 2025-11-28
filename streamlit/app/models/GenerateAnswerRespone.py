# app/models/GenerateAnswerRespone.py
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateAnswerRespone(BaseModel):
    signal: str
    answer: Optional[str]
    full_prompt: Optional[str]
    chat_history: Optional[List[ChatMessage]] = []
