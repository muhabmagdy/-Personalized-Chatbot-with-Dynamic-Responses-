from enum import Enum

class EvaluationMethodEnum(Enum):
    """Methods for evaluating RAG metrics."""
    LLM_FEEDBACK = "llm_feedback"
    SIMILARITY_COMPARISON = "similarity_comparison"
    HUMAN_FEEDBACK = "human_feedback"
    HYBRID = "hybrid"