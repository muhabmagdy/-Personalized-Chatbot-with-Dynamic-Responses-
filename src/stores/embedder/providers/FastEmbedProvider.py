from ..EmbedderInterface import EmbedderInterface
from fastembed import TextEmbedding 
import logging
from typing import List, Union, Optional, cast # 0.7.3 is compatible with standard Python typing

class FastEmbedProvider(EmbedderInterface):
    """
    Implements EmbedderInterface using the fastembed library (v0.7.3), which leverages
    ONNX for highly optimized, fast, and lightweight embedding generation.
    """
    def __init__(self, model_id: str, embedding_size: int, **kwargs):
        """
        Initializes the provider with a specified FastEmbed compatible model.
        
        :param model_id: The name of the model (e.g., 'BAAI/bge-small-en-v1.5').
        :param embedding_size: The expected dimension of the embedding vectors.
        :param kwargs: Additional arguments for TextEmbedding (e.g., cache_dir, threads).
        """
        
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger = logging.getLogger("uvicorn")
        self.embedder: Optional[TextEmbedding] = None

        try:
            # 1. Initialize FastEmbed's TextEmbedding model
            # FastEmbed handles model download, caching, and device selection (CPU/GPU) internally.
            self.embedder = TextEmbedding(
                model_name=self.embedding_model_id,
                **kwargs
            )
            
            # The TextEmbedding object has a 'model_dimensions' property we can use for verification.
            if self.embedder.model_dimensions != self.embedding_size:
                 self.logger.warning(
                    f"Model dimensions detected by FastEmbed ({self.embedder.model_dimensions}) "
                    f"do not match expected size ({self.embedding_size}). Using detected size."
                )
                 self.embedding_size = self.embedder.model_dimensions
                 
            self.logger.info(f"FastEmbed model '{model_id}' loaded successfully. Dimension: {self.embedding_size}")

        except Exception as e:
            self.logger.error(f"Failed to load FastEmbed model {model_id}: {e}")
            self.embedder = None
            self.embedding_size = None

    # --- Interface Methods ---

    def set_embedding_model(self, model_id: str, embedding_size: int):
        """Sets the model metadata. Underlying FastEmbed model is NOT re-initialized."""
        if self.embedding_model_id != model_id:
            self.logger.warning(
                f"Model change requested from '{self.embedding_model_id}' to '{model_id}'. "
                "The underlying FastEmbed model has NOT been re-initialized. "
                "Create a new provider instance for a different model."
            )
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Model metadata set: ID='{model_id}', Size='{embedding_size}'")

        
    def embed_text(self, text: Union[str, List[str]], document_type: Optional[str] = None) -> List[List[float]]:
        """
        Embeds text using the FastEmbed model. Applies 'query' or 'passage' prefixing.
        
        :param text: A single string or a list of strings to embed.
        :param document_type: Should be 'query' or 'passage' for optimized retrieval.
        :return: A list of embedding vectors (List[List[float]]).
        """
        if not self.embedder:
            self.logger.error("FastEmbed Embedder was not initialized correctly.")
            return []

        # 1. Standardize input to List[str]
        input_texts = [text] if isinstance(text, str) else cast(List[str], text)
        if not isinstance(input_texts, list):
            self.logger.error(f"Invalid input type: expected str or List[str], got {type(text)}")
            return []

        # 2. Determine embedding type for prefixing optimization 
        if document_type == 'query':
            embed_type = 'query'
        elif document_type in ('passage', None):
            embed_type = 'passage'
        else:
            self.logger.warning(f"Unknown document_type '{document_type}'. Defaulting to 'passage'.")
            embed_type = 'passage'
        
        try:
            # 3. Use FastEmbed's highly optimized embed method
            # The 'embed' method returns an iterator of NumPy arrays.
            embeddings_generator = self.embedder.embed(
                input_texts, 
                batch_size=256, 
                # The 'type' parameter adds the necessary prefix ('query:' or 'passage:')
                type=embed_type 
            )
            
            # Convert NumPy arrays to lists of floats to match the interface definition
            embeddings = [emb.tolist() for emb in embeddings_generator]
                 
            return embeddings

        except Exception as e:
            self.logger.error(f"Error while embedding text with FastEmbed: {e}")
            return []