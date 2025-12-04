from dataclasses import dataclass
from typing import List, Optional
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum

@dataclass
class RAGEvaluationReport:
    """
    Comprehensive RAG evaluation report.
    
    Contains all evaluation metrics and overall assessment.
    """
    query: str
    answer: str
    retrieved_documents: List[str]
    
    # Core metrics
    answer_relevance: Optional[EvaluationMethodEnum] = None
    context_relevance: Optional[EvaluationMethodEnum] = None
    groundedness: Optional[EvaluationMethodEnum] = None
    
    # Advanced metrics
    context_precision: Optional[EvaluationMethodEnum] = None
    context_recall: Optional[EvaluationMethodEnum] = None
    answer_correctness: Optional[EvaluationMethodEnum] = None
    answer_similarity: Optional[EvaluationMethodEnum] = None
    
    # Overall assessment
    overall_score: float = 0.0
    timestamp: Optional[str] = None
    strategy_name: Optional[str] = None