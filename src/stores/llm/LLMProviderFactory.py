
from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider, GeminiProvider
from helpers.config import Settings
from typing import cast

class LLMProviderFactory:
    def __init__(self, config: Settings):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key = cast(str,self.config.OPENAI_API_KEY),
                api_url = self.config.OPENAI_API_URL,
                default_input_max_characters=cast(int,self.config.INPUT_DAFAULT_MAX_CHARACTERS),
                default_generation_max_output_tokens=cast(int, self.config.GENERATION_DAFAULT_MAX_TOKENS),
                default_generation_temperature=cast(float, self.config.GENERATION_DAFAULT_TEMPERATURE)
            )

        if provider == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key = cast(str,self.config.COHERE_API_KEY),
                default_input_max_characters=cast(int, self.config.INPUT_DAFAULT_MAX_CHARACTERS),
                default_generation_max_output_tokens=cast(int, self.config.GENERATION_DAFAULT_MAX_TOKENS),
                default_generation_temperature=cast(float, self.config.GENERATION_DAFAULT_TEMPERATURE)
            )

        if provider == LLMEnums.GEMINI.value:
            return GeminiProvider(
                api_key = cast(str,self.config.GEMINI_API_KEY),
                default_input_max_characters=cast(int,self.config.INPUT_DAFAULT_MAX_CHARACTERS),
                default_generation_max_output_tokens=cast(int,self.config.GENERATION_DAFAULT_MAX_TOKENS),
                default_generation_temperature=cast(float,self.config.GENERATION_DAFAULT_TEMPERATURE)
            )
        
        return None
