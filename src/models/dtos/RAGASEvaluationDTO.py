# project/src/models/dtos/RAGASEvaluationDTO.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

class EvaluationStatus(str, Enum):
    """Evaluation execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"

class MetricScore(BaseModel):
    """Individual metric score with details."""
    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0-1)")
    raw_value: Optional[float] = Field(None, description="Raw metric value before normalization")
    status: EvaluationStatus = Field(..., description="Metric evaluation status")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    execution_time_ms: Optional[float] = Field(None, description="Time taken to compute metric")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "faithfulness",
                "score": 0.85,
                "raw_value": 0.85,
                "status": "success",
                "error_message": None,
                "execution_time_ms": 245.3
            }
        }

class RAGASEvaluationReport(BaseModel):
    """
    Comprehensive RAGAs evaluation report.
    
    Contains all metrics, metadata, and analysis results.
    Follows Single Responsibility Principle - only holds evaluation data.
    """
    # Identifiers
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    timestamp: str = Field(default_factory=datetime.utcnow().isoformat(), description="Evaluation timestamp")
    
    # Input data
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Generated answer")
    retrieved_contexts: List[str] = Field(..., description="Retrieved document texts")
    strategy_name: Optional[str] = Field(None, description="RAG strategy used")
    
    # Evaluation results
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted average of all metrics")
    overall_status: EvaluationStatus = Field(..., description="Overall evaluation status")
    
    # Individual metrics
    faithfulness: Optional[MetricScore] = Field(None, description="Faithfulness score")
    answer_relevancy: Optional[MetricScore] = Field(None, description="Answer relevancy score")
    context_precision: Optional[MetricScore] = Field(None, description="Context precision score")
    
    # Metadata
    total_execution_time_ms: float = Field(..., description="Total evaluation time")
    metrics_computed: int = Field(..., description="Number of metrics successfully computed")
    metrics_failed: int = Field(..., description="Number of metrics that failed")
    
    # Additional info
    configuration: Dict = Field(default_factory=dict, description="Evaluation configuration used")
    warnings: List[str] = Field(default_factory=list, description="Warnings during evaluation")
    
    class Config:
        schema_extra = {
            "example": {
                "evaluation_id": "eval_abc123",
                "timestamp": "2024-12-08T10:30:00Z",
                "query": "What is Python?",
                "answer": "Python is a high-level programming language...",
                "retrieved_contexts": ["Python is...", "Features include..."],
                "strategy_name": "basic",
                "overall_score": 0.82,
                "overall_status": "success",
                "faithfulness": {
                    "name": "faithfulness",
                    "score": 0.90,
                    "status": "success",
                    "execution_time_ms": 200.5
                },
                "answer_relevancy": {
                    "name": "answer_relevancy",
                    "score": 0.85,
                    "status": "success",
                    "execution_time_ms": 180.3
                },
                "context_precision": {
                    "name": "context_precision",
                    "score": 0.70,
                    "status": "success",
                    "execution_time_ms": 150.2
                },
                "total_execution_time_ms": 531.0,
                "metrics_computed": 3,
                "metrics_failed": 0,
                "configuration": {
                    "batch_size": 10,
                    "timeout_seconds": 60
                },
                "warnings": []
            }
        }
    
    def get_metric_scores(self) -> Dict[str, float]:
        """Get dictionary of metric names to scores."""
        scores = {}
        if self.faithfulness:
            scores["faithfulness"] = self.faithfulness.score
        if self.answer_relevancy:
            scores["answer_relevancy"] = self.answer_relevancy.score
        if self.context_precision:
            scores["context_precision"] = self.context_precision.score
        return scores
    
    def get_successful_metrics(self) -> List[str]:
        """Get list of successfully computed metric names."""
        metrics = []
        if self.faithfulness and self.faithfulness.status == EvaluationStatus.SUCCESS:
            metrics.append("faithfulness")
        if self.answer_relevancy and self.answer_relevancy.status == EvaluationStatus.SUCCESS:
            metrics.append("answer_relevancy")
        if self.context_precision and self.context_precision.status == EvaluationStatus.SUCCESS:
            metrics.append("context_precision")
        return metrics
    
    def get_failed_metrics(self) -> List[str]:
        """Get list of failed metric names."""
        metrics = []
        if self.faithfulness and self.faithfulness.status != EvaluationStatus.SUCCESS:
            metrics.append("faithfulness")
        if self.answer_relevancy and self.answer_relevancy.status != EvaluationStatus.SUCCESS:
            metrics.append("answer_relevancy")
        if self.context_precision and self.context_precision.status != EvaluationStatus.SUCCESS:
            metrics.append("context_precision")
        return metrics
    
    def to_mlflow_metrics(self) -> Dict[str, float]:
        """
        Convert evaluation report to MLflow-compatible metrics dict.
        
        Returns:
            Dictionary with metric names as keys and scores as values.
        """
        metrics = {
            "overall_score": self.overall_score,
            "metrics_computed": float(self.metrics_computed),
            "metrics_failed": float(self.metrics_failed),
            "execution_time_ms": self.total_execution_time_ms
        }
        
        # Add individual metric scores
        if self.faithfulness and self.faithfulness.status == EvaluationStatus.SUCCESS:
            metrics["faithfulness"] = self.faithfulness.score
            metrics["faithfulness_time_ms"] = self.faithfulness.execution_time_ms or 0.0
        
        if self.answer_relevancy and self.answer_relevancy.status == EvaluationStatus.SUCCESS:
            metrics["answer_relevancy"] = self.answer_relevancy.score
            metrics["answer_relevancy_time_ms"] = self.answer_relevancy.execution_time_ms or 0.0
        
        if self.context_precision and self.context_precision.status == EvaluationStatus.SUCCESS:
            metrics["context_precision"] = self.context_precision.score
            metrics["context_precision_time_ms"] = self.context_precision.execution_time_ms or 0.0
        
        return metrics
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Evaluation Report ({self.evaluation_id})",
            f"Strategy: {self.strategy_name or 'Unknown'}",
            f"Overall Score: {self.overall_score:.2f} ({self.overall_status.value})",
            f"",
            "Metrics:",
        ]
        
        if self.faithfulness:
            status_icon = "✓" if self.faithfulness.status == EvaluationStatus.SUCCESS else "✗"
            lines.append(f"  {status_icon} Faithfulness: {self.faithfulness.score:.2f}")
        
        if self.answer_relevancy:
            status_icon = "✓" if self.answer_relevancy.status == EvaluationStatus.SUCCESS else "✗"
            lines.append(f"  {status_icon} Answer Relevancy: {self.answer_relevancy.score:.2f}")
        
        if self.context_precision:
            status_icon = "✓" if self.context_precision.status == EvaluationStatus.SUCCESS else "✗"
            lines.append(f"  {status_icon} Context Precision: {self.context_precision.score:.2f}")
        
        lines.append(f"")
        lines.append(f"Execution Time: {self.total_execution_time_ms:.0f}ms")
        
        if self.warnings:
            lines.append(f"")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)


class BatchEvaluationResult(BaseModel):
    """Results from batch evaluation of multiple queries."""
    batch_id: str = Field(..., description="Batch identifier")
    timestamp: str = Field(default_factory=datetime.utcnow().isoformat(), description="Batch evaluation timestamp")
    total_queries: int = Field(..., description="Total number of queries evaluated")
    successful_evaluations: int = Field(..., description="Number of successful evaluations")
    failed_evaluations: int = Field(..., description="Number of failed evaluations")
    
    # Aggregate metrics
    average_overall_score: float = Field(..., ge=0.0, le=1.0)
    average_faithfulness: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_answer_relevancy: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_context_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Individual reports
    reports: List[RAGASEvaluationReport] = Field(..., description="Individual evaluation reports")
    
    # Timing
    total_execution_time_ms: float = Field(..., description="Total batch execution time")
    average_execution_time_ms: float = Field(..., description="Average time per evaluation")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_xyz789",
                "timestamp": "2024-12-08T10:30:00Z",
                "total_queries": 10,
                "successful_evaluations": 9,
                "failed_evaluations": 1,
                "average_overall_score": 0.81,
                "average_faithfulness": 0.87,
                "average_answer_relevancy": 0.83,
                "average_context_precision": 0.73,
                "reports": [],
                "total_execution_time_ms": 5234.5,
                "average_execution_time_ms": 523.5
            }
        }