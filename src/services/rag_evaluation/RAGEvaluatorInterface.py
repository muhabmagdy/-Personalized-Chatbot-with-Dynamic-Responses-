from abc import ABC, abstractmethod
from typing import List, Optional
from models.dtos.EvaluationResult import EvaluationResult
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum

class RAGEvaluatorInterface(ABC):
    """
    Abstract interface for RAG evaluation metrics.
    
    Each concrete evaluator implements a specific metric using
    one or more evaluation methods (LLM feedback, similarity, human).
    
    Design Patterns:
    - Strategy Pattern: Different evaluation strategies
    - Template Method: Common evaluation flow
    - Single Responsibility: Each evaluator handles one metric
    """
    
    @abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_documents: List[str],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate RAG system output.
        
        Args:
            query: User's question
            answer: Generated answer
            retrieved_documents: List of retrieved document texts
            ground_truth: Optional reference answer for correctness metrics
            **kwargs: Additional metric-specific parameters
            
        Returns:
            EvaluationResult with score and explanation
        """
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """Return the name of the metric being evaluated."""
        pass
    
    @abstractmethod
    def get_evaluation_method(self) -> EvaluationMethodEnum:
        """Return the evaluation method used."""
        pass
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize score to [0, 1] range."""
        if max_val == min_val:
            return 0.0
        return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))