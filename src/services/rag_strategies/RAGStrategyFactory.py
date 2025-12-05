from .RAGStrategyInterface import RAGStrategyInterface
from .BasicRAGStrategy import BasicRAGStrategy
from .FusionRAGStrategy import FusionRAGStrategy
from .ReRankRAGStrategy import ReRankRAGStrategy
from .SentenceWindowRAGStrategy import SentenceWindowRAGStrategy
from .AutoMergingRAGStrategy import AutoMergingRAGStrategy
from .WebSearchRAGStrategy import WebSearchRAGStrategy
from models.enums.RAGTypeEnum import RAGTypeEnum
from typing import Optional, List, Dict
from helpers.config import get_settings
import logging


class RAGStrategyFactory:
    """
    Factory for creating RAG strategy instances.

    Follows Factory Pattern and Open/Closed Principle:
    - Adding new RAG strategies requires only adding a new class and factory case
    - No modification to existing strategies needed
    - Centralized strategy creation logic
    
    Design Patterns Applied:
    - Factory Pattern: Centralized object creation
    - Dependency Injection: Dependencies injected at construction
    - Open/Closed Principle: Open for extension (new strategies), closed for modification
    - Single Responsibility: Only responsible for strategy creation
    
    Usage:
        factory = RAGStrategyFactory(vectordb, llm, embeddings, templates)
        strategy = factory.create_strategy("sentence_window", window_size=3)
    """
    
    def __init__(
        self,
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser,
        config: Optional[Dict] = None
    ):
        """
        Initialize factory with shared dependencies.
        
        Args:
            vectordb_client: Vector database client
            generation_client: LLM client
            embedding_client: Embedding client
            template_parser: Template parser
            config: Optional configuration dict (for API keys, etc.)
        """
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.config = config or {}
        self.logger = logging.getLogger("uvicorn")
        self.settings = get_settings()
    
    def create_strategy(
        self, 
        rag_type: str,
        **kwargs
    ) -> Optional[RAGStrategyInterface]:
        """
        Create a RAG strategy instance based on type.
        
        Args:
            rag_type: Type of RAG strategy (from RAGTypeEnum)
            **kwargs: Strategy-specific parameters
            
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
            
            # Sentence Window RAG
            strategy = factory.create_strategy(
                RAGTypeEnum.SENTENCE_WINDOW.value,
                window_size=3,
                apply_reranking=True
            )
            
            # Web Search RAG
            strategy = factory.create_strategy(
                RAGTypeEnum.WEB_SEARCH.value,
                tavily_api_key="your-api-key"
            )
        """
        # Common dependencies for all strategies
        common_deps = {
            "vectordb_client": self.vectordb_client,
            "generation_client": self.generation_client,
            "embedding_client": self.embedding_client,
            "template_parser": self.template_parser,
        }
        
        try:
            # Basic RAG
            if rag_type == RAGTypeEnum.BASIC.value:
                self.logger.info("Creating Basic RAG strategy")
                return BasicRAGStrategy(**common_deps)
            
            # Fusion RAG
            elif rag_type == RAGTypeEnum.FUSION.value:
                self.logger.info("Creating Fusion RAG strategy")
                fusion_params = {**common_deps}
                
                if "num_queries" in kwargs:
                    fusion_params["num_queries"] = kwargs["num_queries"]
                
                return FusionRAGStrategy(**fusion_params)
            
            # ReRank RAG
            elif rag_type == RAGTypeEnum.RERANK.value:
                self.logger.info("Creating ReRank RAG strategy")
                rerank_params = {**common_deps}
                
                if "initial_retrieve_multiplier" in kwargs:
                    rerank_params["initial_retrieve_multiplier"] = kwargs["initial_retrieve_multiplier"]
                
                return ReRankRAGStrategy(**rerank_params)
            
            # Sentence Window RAG (NEW)
            elif rag_type == RAGTypeEnum.SENTENCE_WINDOW.value:
                self.logger.info("Creating Sentence Window RAG strategy")
                window_params = {**common_deps}
                
                if "window_size" in kwargs:
                    window_params["window_size"] = kwargs["window_size"]
                
                if "apply_reranking" in kwargs:
                    window_params["apply_reranking"] = kwargs["apply_reranking"]
                
                return SentenceWindowRAGStrategy(**window_params)
            
            # Auto-Merging RAG (NEW)
            elif rag_type == RAGTypeEnum.AUTO_MERGING.value:
                self.logger.info("Creating Auto-Merging RAG strategy")
                merging_params = {**common_deps}
                
                if "merge_threshold" in kwargs:
                    merging_params["merge_threshold"] = kwargs["merge_threshold"]
                
                if "similarity_threshold" in kwargs:
                    merging_params["similarity_threshold"] = kwargs["similarity_threshold"]
                
                return AutoMergingRAGStrategy(**merging_params)
            
            # Web Search RAG (NEW)
            elif rag_type == RAGTypeEnum.WEB_SEARCH.value:
                self.logger.info("Creating Web Search RAG strategy")
                web_params = {**common_deps}
                
                # Get API key from kwargs or config
                tavily_key = kwargs.get("tavily_api_key") or self.config.get("tavily_api_key") or self.settings.TAVILY_API_KEY
                
                if not tavily_key:
                    self.logger.warning(
                        "Tavily API key not provided. Web search will be skipped. "
                        "Get free API key at https://tavily.com (1000 searches/month)"
                    )
                
                web_params["tavily_api_key"] = tavily_key
                
                if "web_search_count" in kwargs:
                    web_params["web_search_count"] = kwargs["web_search_count"]
                
                if "vector_search_weight" in kwargs:
                    web_params["vector_search_weight"] = kwargs["vector_search_weight"]
                
                if "web_search_weight" in kwargs:
                    web_params["web_search_weight"] = kwargs["web_search_weight"]
                
                return WebSearchRAGStrategy(**web_params)
            
            # Fallback to Basic RAG
            else:
                self.logger.warning(
                    f"Unsupported RAG type: {rag_type}. Falling back to Basic RAG. "
                    f"Available types: {self.get_available_strategies()}"
                )
                return BasicRAGStrategy(**common_deps)
        
        except Exception as e:
            self.logger.error(f"Error creating RAG strategy '{rag_type}': {e}")
            self.logger.info("Falling back to Basic RAG")
            return BasicRAGStrategy(**common_deps)
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available RAG strategy types.
        
        Returns:
            List of strategy type strings
        """
        return [strategy.value for strategy in RAGTypeEnum]
    
    def get_strategy_info(self, rag_type: str) -> Dict:
        """
        Get detailed information about a strategy.
        
        Returns:
            Dictionary with description, use cases, and requirements
        """
        return {
            "type": rag_type,
            "description": RAGTypeEnum.get_description(rag_type),
            "use_cases": RAGTypeEnum.get_use_cases(rag_type),
            "requirements": RAGTypeEnum.get_requirements(rag_type)
        }
    
    def validate_strategy_config(self, rag_type: str, config: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate configuration for a strategy.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        requirements = RAGTypeEnum.get_requirements(rag_type)
        
        # Check API keys
        required_keys = requirements.get("api_keys", [])
        for key in required_keys:
            if key not in config and key not in self.config:
                return False, f"Missing required API key: {key}"
        
        return True, None