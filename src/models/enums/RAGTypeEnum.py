from enum import Enum
from typing import List, Dict

class RAGTypeEnum(Enum):
    """
    Enumeration of available RAG strategy types.
    
    Each strategy offers different trade-offs between:
    - Precision vs Recall
    - Speed vs Accuracy
    - Internal vs External knowledge
    - Simplicity vs Sophistication
    """
    
    # Basic Strategies
    BASIC = "basic"  # Simple vector search
    """
    Basic RAG: Single query → Vector search → Generate answer
    Best for: Simple Q&A, single-hop reasoning
    """
    
    # Advanced Retrieval Strategies
    FUSION = "fusion"  # Query expansion + Reciprocal Rank Fusion
    """
    Fusion RAG: Query expansion → Multiple searches → Reciprocal Rank Fusion → Generate
    Best for: Complex queries needing multiple perspectives
    """

    RERANK = "rerank"  # Two-stage retrieval with reranking
    """
    ReRank RAG: Retrieve many candidates → Cross-encoder reranking → Generate
    Best for: High precision requirements, complex documents
    """

    SENTENCE_WINDOW = "sentence_window"  # Sentence-level retrieval with context
    AUTO_MERGING = "auto_merging"  # Hierarchical retrieval with auto-merging
    
    # Hybrid Strategies
    WEB_SEARCH = "web_search"  # Vector DB + Web search (Tavily)

    # Future RAG types can be added here:
    # HYDE = "hyde"  # Hypothetical Document Embeddings
    # RAPTOR = "raptor"  # Recursive abstractive processing
    # SELF_RAG = "self_rag"  # Self-reflective RAG
    
    @classmethod
    def get_description(cls, rag_type: str) -> str:
        """Get human-readable description of RAG strategy."""
        descriptions = {
            cls.BASIC.value: "Simple vector similarity search - Fast and reliable",
            cls.FUSION.value: "Query expansion with result fusion - Better for complex queries",
            cls.RERANK.value: "Two-stage retrieval with reranking - High precision",
            cls.SENTENCE_WINDOW.value: "Sentence-level retrieval with context window - Focused and precise",
            cls.AUTO_MERGING.value: "Hierarchical retrieval with smart merging - Maintains document structure",
            cls.WEB_SEARCH.value: "Hybrid vector + web search - Access to real-time information"
        }
        return descriptions.get(rag_type, "Unknown strategy")
    
    @classmethod
    def get_use_cases(cls, rag_type: str) -> List[str]:
        """Get recommended use cases for each strategy."""
        use_cases = {
            cls.BASIC.value: [
                "Simple Q&A",
                "Well-defined queries",
                "Fast responses needed"
            ],
            cls.FUSION.value: [
                "Complex queries",
                "Multi-perspective questions",
                "When query intent is ambiguous"
            ],
            cls.RERANK.value: [
                "High precision required",
                "Complex technical documentation",
                "When accuracy > speed"
            ],
            cls.SENTENCE_WINDOW.value: [
                "Precise factual queries",
                "Detailed documentation",
                "When focus > breadth"
            ],
            cls.AUTO_MERGING.value: [
                "Structured content",
                "Technical documentation",
                "When context structure matters"
            ],
            cls.WEB_SEARCH.value: [
                "Current events",
                "Recent information",
                "Fact verification",
                "Evolving topics"
            ]
        }
        return use_cases.get(rag_type, [])
    
    @classmethod
    def get_requirements(cls, rag_type: str) -> Dict[str, any]:
        """Get requirements and configuration for each strategy."""
        requirements = {
            cls.BASIC.value: {
                "api_keys": [],
                "complexity": "low",
                "avg_latency_ms": 200,
                "cost_per_query": "low"
            },
            cls.FUSION.value: {
                "api_keys": [],
                "complexity": "medium",
                "avg_latency_ms": 500,
                "cost_per_query": "medium"
            },
            cls.RERANK.value: {
                "api_keys": [],
                "complexity": "medium",
                "avg_latency_ms": 600,
                "cost_per_query": "medium-high"
            },
            cls.SENTENCE_WINDOW.value: {
                "api_keys": [],
                "complexity": "medium",
                "avg_latency_ms": 400,
                "cost_per_query": "medium"
            },
            cls.AUTO_MERGING.value: {
                "api_keys": [],
                "complexity": "high",
                "avg_latency_ms": 450,
                "cost_per_query": "medium"
            },
            cls.WEB_SEARCH.value: {
                "api_keys": ["tavily_api_key"],
                "complexity": "medium",
                "avg_latency_ms": 800,
                "cost_per_query": "medium-high",
                "external_dependencies": ["Tavily API (1000 free searches/month)"]
            }
        }
        return requirements.get(rag_type, {})