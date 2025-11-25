from ..EmbedderInterface import EmbedderInterface
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from typing import List, Union, Optional

class HuggingFaceEmbeddingsProvider(EmbedderInterface):
    """
        Implements EmbedderInterface using a local HuggingFace Sentence Transformer model
        via LangChain's HuggingFaceEmbeddings.
    """
    def __init__(self, model_id: str, embedding_size: int, model_kwargs: Optional[dict] = None):
        """
        Initializes the provider with a specified Sentence Transformer model.
        """
        
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger = logging.getLogger("uvicorn")

        if model_kwargs is None:
            # Default to CPU if no kwargs are provided
            model_kwargs = {"device": "cpu"}

        try:
            # 1. Initialize the LangChain HuggingFaceEmbeddings object
            self.embedder = HuggingFaceEmbeddings(
                model_name=self.embedding_model_id,
                model_kwargs=model_kwargs
            )
            self.logger.info(f"Embedding model '{model_id}' loaded successfully with kwargs: {model_kwargs}")

        except Exception as e:
            self.logger.error(f"Failed to load embedding model {model_id}: {e}")
            self.embedder = None
            self.embedding_size = None

    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        Sets the embedding model ID and size. 
        Note: For HuggingFaceEmbeddings, changing the model requires re-initialization, 
        but we fulfill the interface requirement by storing the values.
        """
        if self.embedding_model_id != model_id:
             self.logger.warning(
                 f"Model change requested from '{self.embedding_model_id}' to '{model_id}'. "
                 "The underlying LangChain model has NOT been re-initialized. "
                 "Create a new provider instance for a different model."
            )
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Model metadata set: ID='{model_id}', Size='{embedding_size}'")

        
    def embed_text(self, text: Union[str, List[str]], document_type: Optional[str] = None) -> List[List[float]]:
        """
        Embeds a single string or a list of strings using the HuggingFace model.
        
        :param text: A single string or a list of strings to embed.
        :param document_type: Ignored for HuggingFace embeddings, kept for interface compatibility.
        :return: A list of embedding vectors (List[List[float]]). Returns empty list on failure.
        """
        if not self.embedder:
            self.logger.error("HuggingFace Embedder was not initialized correctly.")
            return []

        # 1. Ensure input is a list of strings
        if isinstance(text, str):
            input_texts = [text]
        elif isinstance(text, list):
            input_texts = text
        else:
            self.logger.error(f"Invalid input type: expected str or List[str], got {type(text)}")
            return []

        try:
            # 2. Use LangChain's embed_documents method for batch embedding
            # This method consistently returns List[List[float]].
            embeddings = self.embedder.embed_documents(input_texts)
            
            # Optional: Check output size for consistency
            if self.embedding_size and embeddings and len(embeddings[0]) != self.embedding_size:
                 self.logger.warning(
                     f"Embedding dimension ({len(embeddings[0])}) does not match expected size ({self.embedding_size})."
                )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error while embedding text: {e}")
            return []
    


    

