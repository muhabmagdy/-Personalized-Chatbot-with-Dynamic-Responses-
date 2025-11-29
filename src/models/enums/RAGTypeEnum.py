from enum import Enum

class RAGTypeEnum(Enum):
    """
    Enumeration of supported RAG strategies.
    Easy to extend with new RAG types.
    """
    
    BASIC = "basic"
    """
    Basic RAG: Single query → Vector search → Generate answer
    Best for: Simple Q&A, single-hop reasoning
    """
    
    FUSION = "fusion"
    """
    Fusion RAG: Query expansion → Multiple searches → Reciprocal Rank Fusion → Generate
    Best for: Complex queries needing multiple perspectives
    """
    
    RERANK = "rerank"
    """
    ReRank RAG: Retrieve many candidates → Cross-encoder reranking → Generate
    Best for: High precision requirements, complex documents
    """
    
    # Future RAG types can be added here:
    # HYDE = "hyde"  # Hypothetical Document Embeddings
    # RAPTOR = "raptor"  # Recursive abstractive processing
    # SELF_RAG = "self_rag"  # Self-reflective RAG