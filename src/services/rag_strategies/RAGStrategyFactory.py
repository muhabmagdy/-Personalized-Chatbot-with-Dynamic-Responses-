from .RAGStrategyInterface import RAGStrategyInterface
from .BasicRAGStrategy import BasicRAGStrategy
from .FusionRAGStrategy import FusionRAGStrategy
from .ReRankRAGStrategy import ReRankRAGStrategy
from models.enums.RAGTypeEnum import RAGTypeEnum
from typing import Optional
import logging

class RAGStrategyFactory:
    """
    Factory for creating RAG strategy instances.
    
    Follows Factory Pattern and Open/Closed Principle:
    - Adding new RAG strategies requires only adding a new class and factory case
    - No modification to existing strategies needed
    - Centralized strategy creation logic
    """
    
    def __init__(
        self,
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser
    ):
        """
        Initialize factory with shared dependencies.
        
        Args:
            vectordb_client: Vector database client
            generation_client: LLM client
            embedding_client: Embedding client
            template_parser: Template parser
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.logger = logging.getLogger("uvicorn")
    
    def create_strategy(
        self, 
        rag_type: str,
        **kwargs
    ) -> Optional[RAGStrategyInterface]:
        """
        Create a RAG strategy instance based on type.
        
        Args:
            rag_type: Type of RAG strategy (from RAGTypeEnum)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            RAGStrategyInterface instance or None if type unsupported
            
        Examples:
            # Basic RAG
            strategy = factory.create_strategy(RAGTypeEnum.BASIC.value)
            
            # Fusion RAG with custom query count
            strategy = factory.create_strategy(
                RAGTypeEnum.FUSION.value,
                num_queries=5
            )
            
            # ReRank RAG with custom multiplier
            strategy = factory.create_strategy(
                RAGTypeEnum.RERANK.value,
                initial_retrieve_multiplier=4
            )
        """
        # Common dependencies for all strategies
        common_deps = {
            "vectordb_client": self.vectordb_client,
            "generation_client": self.generation_client,
            "embedding_client": self.embedding_client,
            "template_parser": self.template_parser,
        }
        
        # Create strategy based on type
        if rag_type == RAGTypeEnum.BASIC.value:
            self.logger.info("Creating Basic RAG strategy")
            return BasicRAGStrategy(**common_deps)
        
        elif rag_type == RAGTypeEnum.FUSION.value:
            self.logger.info("Creating Fusion RAG strategy")
            fusion_params = {**common_deps}
            
            # Add Fusion-specific parameters
            if "num_queries" in kwargs:
                fusion_params["num_queries"] = kwargs["num_queries"]
            
            return FusionRAGStrategy(**fusion_params)
        
        elif rag_type == RAGTypeEnum.RERANK.value:
            self.logger.info("Creating ReRank RAG strategy")
            rerank_params = {**common_deps}
            
            # Add ReRank-specific parameters
            if "initial_retrieve_multiplier" in kwargs:
                rerank_params["initial_retrieve_multiplier"] = kwargs["initial_retrieve_multiplier"]
            
            return ReRankRAGStrategy(**rerank_params)
        
        # Add more strategies here as they are implemented
        # elif rag_type == RAGTypeEnum.HYDE.value:
        #     return HyDERAGStrategy(**common_deps)
        
        else:
            self.logger.warning(
                f"Unsupported RAG type: {rag_type}. Falling back to Basic RAG."
            )
            return BasicRAGStrategy(**common_deps)
    
    def get_available_strategies(self) -> list[str]:
        """
        Get list of available RAG strategy types.
        
        Returns:
            List of strategy type strings
        """
        return [strategy.value for strategy in RAGTypeEnum]