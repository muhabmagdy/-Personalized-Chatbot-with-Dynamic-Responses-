from typing import List, Optional, Dict
from datetime import datetime
import logging
from models.dtos.RAGEvaluationReport import RAGEvaluationReport
from .RAGEvaluatorInterface import RAGEvaluatorInterface
from .AnswerRelevanceEvaluator import AnswerRelevanceEvaluator
from .ContextRelevanceEvaluator import ContextRelevanceEvaluator
from .GroundednessEvaluator import GroundednessEvaluator
from .ContextPrecisionEvaluator import ContextPrecisionEvaluator
from .ContextRecallEvaluator import ContextRecallEvaluator
from .AnswerCorrectnessEvaluator import AnswerCorrectnessEvaluator

class RAGEvaluationService:
    """
    RAG Evaluation Service: Orchestrates comprehensive RAG evaluation.
    
    Design Patterns:
    - Facade Pattern: Provides simple interface to complex evaluation system
    - Strategy Pattern: Uses multiple evaluator strategies
    - Dependency Injection: Evaluators injected for flexibility
    
    Features:
    - Run all metrics or selective evaluation
    - Async evaluation for performance
    - Comprehensive reporting
    - Integration-ready (UI, MLflow, logging)
    
    Usage:
        service = RAGEvaluationService(llm_client, embedding_client)
        report = await service.evaluate_rag_response(
            query="What is Python?",
            answer="Python is a programming language...",
            retrieved_documents=[...],
            strategy_name="Basic RAG"
        )
    """
    
    def __init__(
        self,
        llm_client,
        embedding_client,
        enable_all_metrics: bool = True
    ):
        """
        Initialize evaluation service.
        
        Args:
            llm_client: LLM for feedback-based evaluation
            embedding_client: For similarity-based evaluation
            enable_all_metrics: Enable all metrics by default (default: True)
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.logger = logging.getLogger("uvicorn")
        
        # Initialize evaluators
        self.evaluators: Dict[str, RAGEvaluatorInterface] = {}
        
        if enable_all_metrics:
            self._initialize_all_evaluators()
    
    def _initialize_all_evaluators(self):
        """Initialize all available evaluators."""
        # Core metrics
        self.evaluators["answer_relevance"] = AnswerRelevanceEvaluator(
            llm_client=self.llm_client,
            embedding_client=self.embedding_client,
            use_hybrid=True
        )
        
        self.evaluators["context_relevance"] = ContextRelevanceEvaluator(
            embedding_client=self.embedding_client,
            llm_client=self.llm_client,
            use_llm_verification=False  # Set True for more accuracy but slower
        )
        
        self.evaluators["groundedness"] = GroundednessEvaluator(
            llm_client=self.llm_client
        )
        
        # Advanced metrics
        self.evaluators["context_precision"] = ContextPrecisionEvaluator(
            llm_client=self.llm_client,
            embedding_client=self.embedding_client
        )
        
        self.evaluators["context_recall"] = ContextRecallEvaluator(
            llm_client=self.llm_client
        )
        
        self.evaluators["answer_correctness"] = AnswerCorrectnessEvaluator(
            llm_client=self.llm_client,
            embedding_client=self.embedding_client
        )
        
        self.logger.info(f"Initialized {len(self.evaluators)} evaluators")
    
    def add_evaluator(self, name: str, evaluator: RAGEvaluatorInterface):
        """Add custom evaluator."""
        self.evaluators[name] = evaluator
        self.logger.info(f"Added custom evaluator: {name}")
    
    def remove_evaluator(self, name: str):
        """Remove evaluator."""
        if name in self.evaluators:
            del self.evaluators[name]
            self.logger.info(f"Removed evaluator: {name}")
    
    async def evaluate_rag_response(
        self,
        query: str,
        answer: str,
        retrieved_documents: List[str],
        strategy_name: Optional[str] = None,
        ground_truth: Optional[str] = None,
        metrics_to_evaluate: Optional[List[str]] = None
    ) -> RAGEvaluationReport:
        """
        Comprehensive RAG evaluation.
        
        Args:
            query: User's question
            answer: Generated answer
            retrieved_documents: Retrieved document texts
            strategy_name: RAG strategy used (for reporting)
            ground_truth: Optional reference answer
            metrics_to_evaluate: Specific metrics to run (None = all)
            
        Returns:
            RAGEvaluationReport with all evaluation results
        """
        self.logger.info(
            f"Starting RAG evaluation: {len(self.evaluators)} metrics, "
            f"{len(retrieved_documents)} documents"
        )
        
        # Create report
        report = RAGEvaluationReport(
            query=query,
            answer=answer,
            retrieved_documents=retrieved_documents,
            strategy_name=strategy_name,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Determine which metrics to evaluate
        metrics = metrics_to_evaluate or list(self.evaluators.keys())
        
        # Run evaluations
        evaluation_results = {}
        
        for metric_name in metrics:
            if metric_name not in self.evaluators:
                self.logger.warning(f"Evaluator not found: {metric_name}")
                continue
            
            try:
                evaluator = self.evaluators[metric_name]
                
                result = await evaluator.evaluate(
                    query=query,
                    answer=answer,
                    retrieved_documents=retrieved_documents,
                    ground_truth=ground_truth
                )
                
                evaluation_results[metric_name] = result
                
                self.logger.info(
                    f"{metric_name}: {result.score:.2f} - {result.explanation}"
                )
                
            except Exception as e:
                self.logger.error(f"Error evaluating {metric_name}: {e}")
        
        # Assign results to report
        report.answer_relevance = evaluation_results.get("answer_relevance")
        report.context_relevance = evaluation_results.get("context_relevance")
        report.groundedness = evaluation_results.get("groundedness")
        report.context_precision = evaluation_results.get("context_precision")
        report.context_recall = evaluation_results.get("context_recall")
        report.answer_correctness = evaluation_results.get("answer_correctness")
        
        # Calculate overall score
        report.overall_score = self._calculate_overall_score(evaluation_results)
        
        self.logger.info(f"Evaluation complete. Overall score: {report.overall_score:.2f}")
        
        return report
    
    def _calculate_overall_score(
        self,
        evaluation_results: Dict[str, any]
    ) -> float:
        """
        Calculate weighted overall score.
        
        Weights:
        - Answer Relevance: 25%
        - Context Relevance: 20%
        - Groundedness: 30% (most critical - prevents hallucinations)
        - Context Precision: 10%
        - Context Recall: 10%
        - Answer Correctness: 5% (only if ground truth available)
        """
        weights = {
            "answer_relevance": 0.25,
            "context_relevance": 0.20,
            "groundedness": 0.30,
            "context_precision": 0.10,
            "context_recall": 0.10,
            "answer_correctness": 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, result in evaluation_results.items():
            if result and metric_name in weights:
                total_score += result.score * weights[metric_name]
                total_weight += weights[metric_name]
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def format_report_for_ui(self, report: RAGEvaluationReport) -> Dict:
        """
        Format report for UI display.
        
        Returns:
            Dictionary with UI-friendly structure
        """
        return {
            "query": report.query,
            "strategy": report.strategy_name,
            "overall_score": round(report.overall_score, 2),
            "timestamp": report.timestamp,
            "metrics": {
                "answer_relevance": self._format_metric(report.answer_relevance),
                "context_relevance": self._format_metric(report.context_relevance),
                "groundedness": self._format_metric(report.groundedness),
                "context_precision": self._format_metric(report.context_precision),
                "context_recall": self._format_metric(report.context_recall),
                "answer_correctness": self._format_metric(report.answer_correctness)
            },
            "summary": self._generate_summary(report)
        }
    
    def format_report_for_mlflow(self, report: RAGEvaluationReport) -> Dict:
        """
        Format report for MLflow logging.
        
        Returns:
            Dictionary with metrics suitable for MLflow
        """
        metrics = {}
        
        if report.answer_relevance:
            metrics["answer_relevance_score"] = report.answer_relevance.score
        
        if report.context_relevance:
            metrics["context_relevance_score"] = report.context_relevance.score
        
        if report.groundedness:
            metrics["groundedness_score"] = report.groundedness.score
        
        if report.context_precision:
            metrics["context_precision_score"] = report.context_precision.score
        
        if report.context_recall:
            metrics["context_recall_score"] = report.context_recall.score
        
        if report.answer_correctness:
            metrics["answer_correctness_score"] = report.answer_correctness.score
        
        metrics["overall_rag_score"] = report.overall_score
        metrics["num_retrieved_documents"] = len(report.retrieved_documents)
        
        return {
            "metrics": metrics,
            "params": {
                "rag_strategy": report.strategy_name or "unknown",
                "query_length": len(report.query),
                "answer_length": len(report.answer)
            },
            "tags": {
                "evaluation_timestamp": report.timestamp,
                "component": "rag_evaluation"
            }
        }
    
    def _format_metric(self, result) -> Optional[Dict]:
        """Format single metric for UI."""
        if not result:
            return None
        
        return {
            "score": round(result.score, 2),
            "explanation": result.explanation,
            "method": result.method.value,
            "status": self._get_status_label(result.score)
        }
    
    def _get_status_label(self, score: float) -> str:
        """Get human-readable status label."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _generate_summary(self, report: RAGEvaluationReport) -> str:
        """Generate human-readable summary."""
        summary_parts = []
        
        if report.overall_score >= 0.8:
            summary_parts.append("Excellent RAG performance overall.")
        elif report.overall_score >= 0.6:
            summary_parts.append("Good RAG performance with room for improvement.")
        else:
            summary_parts.append("RAG performance needs improvement.")
        
        # Identify weakest metric
        metrics = {
            "Answer Relevance": report.answer_relevance,
            "Context Relevance": report.context_relevance,
            "Groundedness": report.groundedness
        }
        
        weak_metrics = [
            name for name, result in metrics.items()
            if result and result.score < 0.6
        ]
        
        if weak_metrics:
            summary_parts.append(f"Focus on improving: {', '.join(weak_metrics)}")
        
        return " ".join(summary_parts)