# app/api/client.py
import requests
from ..models.GenerateAnswerRequest import GenerateAnswerRequest
from ..models.GenerateAnswerRespone import GenerateAnswerRespone

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def answer_rag(self, project_id: int, text: str, limit: int = 5) -> GenerateAnswerRespone:
        url = f"{self.base_url}/api/v1/nlp/index/answer/{project_id}"
        payload = GenerateAnswerRequest(text=text, limit=limit).dict()
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return GenerateAnswerRespone(**response.json())
