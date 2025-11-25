import logging
from typing import Optional, Dict, Any, cast

# Import necessary files
from .EmbedderInterface import EmbedderInterface
from .providers import HuggingFaceEmbeddingsProvider
from .EmbedderEnums import EmbedderEnums
from helpers.config import Settings

class EmbedderProviderFactory:
    """
    Factory class to create instances of embedding providers based on a configuration 
    passed in the constructor. This uses conditional logic (if/else) for simplicity.
    """
    
    def __init__(self, config: Settings):
        """Initializes the factory with the application configuration."""
        self.config = config
        self._logger = logging.getLogger("uvicorn")

    def create(self, provider_name: str) -> Optional[EmbedderInterface]:
        """
        Creates and returns an instance of the specified embedding provider.

        :param provider_name: The name of the embedding provider (e.g., "HuggingFace").
        :return: An initialized EmbedderInterface instance, or None if unsupported.
        """
        
        if provider_name == EmbedderEnums.HUGGINGFACE.value:
            # Create HuggingFace Provider using configuration settings
            try:
                self._logger.info(f"Creating HuggingFace Provider with model ID: {self.config.EMBEDDING_MODEL_ID}")
                return HuggingFaceEmbeddingsProvider(
                    model_id=cast(str, self.config.EMBEDDING_MODEL_ID),
                    embedding_size=cast(int, self.config.EMBEDDING_MODEL_SIZE),
                    model_kwargs={"device": cast(str, self.config.HUGGINGFACE_DEVICE)}
                )
            except Exception as e:
                self._logger.critical(f"Failed to instantiate HuggingFace Provider: {e}")
                raise RuntimeError(f"Factory failed to create {provider_name} provider.") from e
        
        # Add more providers here (e.g., if provider_name == EmbedderEnums.OPENAI.value: ...)
        
        self._logger.warning(f"Unsupported embedding provider requested: {provider_name}")
        return None