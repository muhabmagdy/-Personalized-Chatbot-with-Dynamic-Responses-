import logging
from typing import List, Optional, Union
import google.generativeai as genai
from ..LLMInterface import LLMInterface 
from ..LLMEnums import GeminiEnums


class GeminiProvider(LLMInterface):

    def __init__(self,
                 api_key: str,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):

        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        genai.configure(api_key=self.api_key)
        self.enums = GeminiEnums
        self.client = genai
        self.logger = logging.getLogger("uvicorn")

    # -----------------------------------------
    # MODEL SELECTION
    # -----------------------------------------
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    # -----------------------------------------
    # HELPER
    # -----------------------------------------
    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    # -----------------------------------------
    # PROMPT BUILDER (SIMILAR TO OPENAIPROVIDER)
    # -----------------------------------------
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }

    # -----------------------------------------
    # GENERATION
    # -----------------------------------------
    def generate_text(self,
                      prompt: str,
                      chat_history: list = [],
                      max_output_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> Optional[str]:

        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        try:
            model = self.client.GenerativeModel(self.generation_model_id)
            
            # Build conversation history
            conversation = []
            for message in chat_history[-4:]:
                if message.get("content"):
                    conversation.append({
                        "role": message.get("role", "user"),
                        "parts": [message["content"]]
                    })
            
            # Add current prompt
            conversation.append({
                "role": "user",
                "parts": [prompt]
            })

            response = model.generate_content(
                contents=conversation,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens
                }
            )

            self.logger.info("Response from Gemini generate_content: {}".format(response))


        except Exception as e:
            self.logger.error(f"API Error during generate_content: {e}")
            return None
        
        # Check response safely without FinishReason import
        if not hasattr(response, 'candidates') or not response.candidates:
            # Check if prompt was blocked
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                self.logger.error(f"Input prompt blocked: {getattr(response.prompt_feedback, 'block_reason', 'Unknown')}")
                return "Error: Your input was blocked by safety filters."
            
            self.logger.error("Error while generating text with Gemini: No candidates returned.")
            return None

        # Check if content was blocked using string comparison
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', None)
        
        # Use string comparison for safety check
        if finish_reason and hasattr(finish_reason, 'name') and finish_reason.name == 'SAFETY':
            self.logger.error(f"Generated content was blocked: {getattr(candidate, 'safety_ratings', 'Unknown')}")
            return "Error: Generated content was blocked by safety filters. Please rephrase."

        # If we get here, return the text
        if hasattr(response, 'text'):
            return response.text
        
        self.logger.error("No text in response")
        return None

    # -----------------------------------------
    # EMBEDDING
    # -----------------------------------------
    def embed_text(self, text: Union[str, List[str]], document_type: Optional[str] = None):

        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        response = self.client.embed_content(
            model=self.embedding_model_id,
            content=text,
            output_dimensionality=self.embedding_size
        )
        self.logger.info("Response from Gemini embed_content: {}".format(response.keys()))

        # Check if response is None/empty (always a good idea)
        if not response:
            self.logger.error("Error while embedding text with Gemini: Response is empty")
            return None

        # CCheck if the key 'embeddings' exists in the dictionary
        if 'embedding' not in response:
            self.logger.error("Error while embedding text with Gemini: 'embedding' key not found in response dictionary.")
            return None

        # Access the data using dictionary key notation []
        # Gemini returns a list of dictionaries with 'values' inside the 'embeddings' list.
        # We now iterate over the LIST of embeddings within the response dictionary.
        return response['embedding']
        # NOTE: The exact key inside `vec` might be 'values' or 'embedding', check the raw response structure.
        # The common structure for a list of embeddings is:
        # response['embeddings'] = [ { 'values': [...] }, { 'values': [...] } ]
