import logging
from typing import List, Optional, Union
import ollama
from ..LLMInterface import LLMInterface

class OllamaProvider(LLMInterface):

    def __init__(self,
                 host: str,
                 default_input_max_characters: int = 2000,
                 default_generation_max_output_tokens: int = 500,
                 default_generation_temperature: float = 0.1):

        self.generation_model_id = None
        self.embedding_model_id = None
        
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.embedding_size = None  # Ollama returns dynamic vector sizes depending on model
        
        self.host = host
        
        self.client = ollama.Client(host=self.host)

        self.logger = logging.getLogger("uvicorn")
        self.logger.info(f"OllamaProvider initialized with host: {self.host}, client {self.client}")

    # ---------------------------------------------------
    # HELPER
    # ---------------------------------------------------
    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    # ---------------------------------------------------
    # PROMPT BUILDER
    # ---------------------------------------------------
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }

    # ---------------------------------------------------
    # GENERATION MODEL
    # ---------------------------------------------------
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    # ---------------------------------------------------
    # GENERATION
    # ---------------------------------------------------
    def generate_text(self,
                      prompt: str,
                      chat_history: list = [],
                      max_output_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> Optional[str]:

        if not self.generation_model_id:
            self.logger.error("Ollama generation model not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        try:
            messages = []

            # Use last 4 messages for context
            for msg in chat_history[-4:]:
                if msg.get("content"):
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg["content"]
                    })

            # Add new user prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            response = self.client.chat(
                model=self.generation_model_id,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_output_tokens
                }
            )

            return response.get("message", {}).get("content", None)

        except Exception as e:
            self.logger.error(f"Ollama generate_text error: {e}")
            return None

    # ---------------------------------------------------
    # EMBEDDING MODEL
    # ---------------------------------------------------
    def set_embedding_model(self, model_id: str, embedding_size: int = None):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    # ---------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------
    def embed_text(self, text: Union[str, List[str]], document_type: Optional[str] = None):

        if not self.embedding_model_id:
            self.logger.error("Ollama embedding model not set")
            return None

        if isinstance(text, str):
            text = [text]

        try:
            embeddings = []
            for item in text:
                result = self.client.embeddings(
                    model=self.embedding_model_id,
                    prompt=item
                )
                embeddings.append(result.get("embedding"))

            return embeddings if len(embeddings) > 1 else embeddings[0]

        except Exception as e:
            self.logger.error(f"Ollama embed_text error: {e}")
            return None
