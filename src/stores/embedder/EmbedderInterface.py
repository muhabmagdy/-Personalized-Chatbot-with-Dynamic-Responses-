from abc import ABC, abstractmethod
from typing import Union, List, Optional

# --- Interface Definition ---
class EmbedderInterface(ABC):
    """Abstract Base Class for all embedding providers."""
    
    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """Sets the model ID and expected embedding size."""
        pass

    @abstractmethod
    def embed_text(self, text: Union[str, List[str]], document_type: Optional[str] = None) -> List[List[float]]:
        """
        Embeds text (single string or list of strings) and returns a list 
        of embedding vectors (List[List[float]]).
        """
        pass