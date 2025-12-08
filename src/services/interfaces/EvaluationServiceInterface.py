# project/src/services/interfaces/EvaluationServiceInterface.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from models.dtos.RAGASEvaluationDTO import RAGASEvaluationReport, BatchEvaluationResult


class EvaluationServiceInterface(ABC):
    """
    Abstract interface for RAG evaluation services.
    
    Follows:
    - Interface Segregation Principle (ISP)
    - Dependency Inversion Principle (DIP)
    - Open/Closed Principle (OCP)
    
    Allows different evaluation implementations (RAGAs, custom metrics, etc.)
    """
    
    @abstractmethod
    async def evaluate_single(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        strategy_name: Optional[str] = None,
        enabled_metrics: Optional[List[str]] = None
    ) -> RAGASEvaluationReport:
        """
        Evaluate a single RAG response.
        
        Args:
            query: User's question
            answer: Generated answer
            retrieved_contexts: List of retrieved document texts
            strategy_name: Name of RAG strategy used
            enabled_metrics: Specific metrics to compute
            
        Returns:
            Evaluation report with metrics and metadata
        """
        pass
    
    @abstractmethod
    async def evaluate_batch(
        self,
        items: List[Dict],
        enabled_metrics: Optional[List[str]] = None,
        parallel: bool = False
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple RAG responses in batch.
        
        Args:
            items: List of evaluation items (query, answer, contexts)
            enabled_metrics: Metrics to compute
            parallel: Enable parallel processing
            
        Returns:
            Batch evaluation result with aggregated metrics
        """
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available evaluation metrics.
        
        Returns:
            List of metric names
        """
        pass
    
    @abstractmethod
    def validate_metrics(self, metrics: List[str]) -> bool:
        """
        Validate if requested metrics are available.
        
        Args:
            metrics: List of metric names to validate
            
        Returns:
            True if all metrics are available
        """
        pass