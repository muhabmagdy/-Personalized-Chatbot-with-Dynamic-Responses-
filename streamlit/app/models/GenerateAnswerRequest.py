# app/models/GenerateAnswerRequest.py
from pydantic import BaseModel
from typing import Optional

class GenerateAnswerRequest(BaseModel):
    text: str
    limit: Optional[int] = 5
