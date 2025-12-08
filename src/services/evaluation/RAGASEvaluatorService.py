# project/src/services/evaluation/RAGASEvaluatorService.py

import asyncio
import time
import uuid
from typing import List, Optional, Dict
from datetime import datetime
import logging

from services.interfaces.EvaluationServiceInterface import EvaluationServiceInterface
from services.evaluation.metrics.FaithfulnessMetric import FaithfulnessMetric
from services.evaluation.metrics.AnswerRelevancyMetric import AnswerRelevancyMetric
from services.evaluation.metrics.ContextPrecisionMetric import ContextPrecisionMetric
from models.dtos.RAGASEvaluationDTO import (
    RAGASEvaluationReport,
    MetricScore,
    EvaluationStatus,
    BatchEvaluationResult
)
from helpers.config import Settings


class RAGASEvaluatorService(EvaluationServiceInterface):
    """
    RAGAs-based evaluation service for RAG systems.
    
    Implements three core metrics without ground truth:
    - Faithfulness: Answer grounded in retrieved contexts
    - Answer Relevancy: Answer relevance to the query
    - Context Precision: Quality of retrieved contexts
    
    SOLID Principles:
    - Single Responsibility: Only evaluates RAG responses
    - Open/Closed: Extensible with new metrics
    - Liskov Substitution: Implements EvaluationServiceInterface
    - Interface Segregation: Focused evaluation interface
    - Dependency Inversion: Depends on abstractions (LLM client, embeddings)
    
    Design Patterns:
    - Strategy Pattern: Different metrics are interchangeable
    - Factory Pattern: Creates metric instances
    - Template Method: Common evaluation workflow
    """
    
    AVAILABLE_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
    
    def __init__(
        self,
        llm_client,
        embedding_client,
        settings: Settings
    ):
        """
        Initialize RAGAs evaluator service.
        
        Args:
            llm_client: LLM client for evaluation (Ollama)
            embedding_client: Embedding client for similarity
            settings: Application settings
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.settings = settings
        self.logger = logging.getLogger("uvicorn")
        
        # Initialize metrics
        self._initialize_metrics()
        
        self.logger.info(
            f"RAGAs evaluator initialized with metrics: "
            f"{', '.join(self.AVAILABLE_METRICS)}"
        )
    
    def _initialize_metrics(self) -> None:
        """Initialize metric calculators based on configuration."""
        self.metrics = {}
        
        if self.settings.RAGAS_ENABLE_FAITHFULNESS:
            self.metrics["faithfulness"] = FaithfulnessMetric(
                llm_client=self.llm_client
            )
        
        if self.settings.RAGAS_ENABLE_ANSWER_RELEVANCY:
            self.metrics["answer_relevancy"] = AnswerRelevancyMetric(
                llm_client=self.llm_client,
                embedding_client=self.embedding_client
            )
        
        if self.settings.RAGAS_ENABLE_CONTEXT_PRECISION:
            self.metrics["context_precision"] = ContextPrecisionMetric(
                llm_client=self.llm_client
            )
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        return list(self.metrics.keys())
    
    def validate_metrics(self, metrics: List[str]) -> bool:
        """Validate if requested metrics are available."""
        return all(metric in self.metrics for metric in metrics)
    
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
            enabled_metrics: Specific metrics to compute (None = all)
            
        Returns:
            Comprehensive evaluation report
        """
        evaluation_id = f"eval_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        self.logger.info(f"Starting evaluation {evaluation_id}")
        
        # Determine which metrics to compute
        metrics_to_compute = enabled_metrics or list(self.metrics.keys())
        
        # Validate metrics
        if not self.validate_metrics(metrics_to_compute):
            invalid = [m for m in metrics_to_compute if m not in self.metrics]
            self.logger.error(f"Invalid metrics requested: {invalid}")
            raise ValueError(f"Invalid metrics: {invalid}")
        
        # Compute metrics
        metric_results = {}
        warnings = []
        
        for metric_name in metrics_to_compute:
            try:
                metric_calculator = self.metrics[metric_name]
                
                self.logger.debug(f"Computing {metric_name}...")
                metric_start = time.time()
                
                # Compute with timeout
                score = await asyncio.wait_for(
                    metric_calculator.compute(
                        query=query,
                        answer=answer,
                        contexts=retrieved_contexts
                    ),
                    timeout=self.settings.RAGAS_TIMEOUT_SECONDS
                )
                
                metric_time = (time.time() - metric_start) * 1000  # ms
                
                metric_results[metric_name] = MetricScore(
                    name=metric_name,
                    score=score,
                    raw_value=score,
                    status=EvaluationStatus.SUCCESS,
                    execution_time_ms=metric_time
                )
                
                self.logger.debug(f"{metric_name} = {score:.3f} ({metric_time:.0f}ms)")
                
            except asyncio.TimeoutError:
                self.logger.warning(f"{metric_name} timed out")
                metric_results[metric_name] = MetricScore(
                    name=metric_name,
                    score=0.0,
                    status=EvaluationStatus.TIMEOUT,
                    error_message=f"Timeout after {self.settings.RAGAS_TIMEOUT_SECONDS}s"
                )
                warnings.append(f"{metric_name} computation timed out")
                
            except Exception as e:
                self.logger.error(f"{metric_name} failed: {e}")
                metric_results[metric_name] = MetricScore(
                    name=metric_name,
                    score=0.0,
                    status=EvaluationStatus.FAILED,
                    error_message=str(e)
                )
                warnings.append(f"{metric_name} computation failed: {str(e)}")
        
        # Calculate overall score (average of successful metrics)
        successful_scores = [
            result.score 
            for result in metric_results.values() 
            if result.status == EvaluationStatus.SUCCESS
        ]
        
        overall_score = (
            sum(successful_scores) / len(successful_scores)
            if successful_scores else 0.0
        )
        
        # Determine overall status
        failed_count = sum(
            1 for r in metric_results.values() 
            if r.status != EvaluationStatus.SUCCESS
        )
        
        if failed_count == 0:
            overall_status = EvaluationStatus.SUCCESS
        elif failed_count == len(metric_results):
            overall_status = EvaluationStatus.FAILED
        else:
            overall_status = EvaluationStatus.PARTIAL
        
        # Calculate execution time
        total_time = (time.time() - start_time) * 1000  # ms
        
        # Build report
        report = RAGASEvaluationReport(
            evaluation_id=evaluation_id,
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            strategy_name=strategy_name,
            overall_score=overall_score,
            overall_status=overall_status,
            faithfulness=metric_results.get("faithfulness"),
            answer_relevancy=metric_results.get("answer_relevancy"),
            context_precision=metric_results.get("context_precision"),
            total_execution_time_ms=total_time,
            metrics_computed=len(successful_scores),
            metrics_failed=failed_count,
            configuration={
                "enabled_metrics": metrics_to_compute,
                "timeout_seconds": self.settings.RAGAS_TIMEOUT_SECONDS,
                "batch_size": self.settings.RAGAS_BATCH_SIZE
            },
            warnings=warnings
        )
        
        self.logger.info(
            f"Evaluation {evaluation_id} complete: "
            f"score={overall_score:.3f}, "
            f"status={overall_status.value}, "
            f"time={total_time:.0f}ms"
        )
        
        return report
    
    async def evaluate_batch(
        self,
        items: List[Dict],
        enabled_metrics: Optional[List[str]] = None,
        parallel: bool = False
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple RAG responses in batch.
        
        Args:
            items: List of dicts with keys: query, answer, retrieved_contexts, strategy_name
            enabled_metrics: Metrics to compute for all items
            parallel: Enable parallel evaluation (experimental)
            
        Returns:
            Batch evaluation result with aggregated metrics
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        self.logger.info(
            f"Starting batch evaluation {batch_id} "
            f"({len(items)} items, parallel={parallel})"
        )
        
        # Evaluate items
        if parallel:
            # Parallel evaluation (use with caution - may overwhelm LLM)
            tasks = [
                self.evaluate_single(
                    query=item["query"],
                    answer=item["answer"],
                    retrieved_contexts=item["retrieved_contexts"],
                    strategy_name=item.get("strategy_name"),
                    enabled_metrics=enabled_metrics
                )
                for item in items
            ]
            reports = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            reports = [
                r for r in reports 
                if isinstance(r, RAGASEvaluationReport)
            ]
        else:
            # Sequential evaluation (recommended)
            reports = []
            for i, item in enumerate(items, 1):
                try:
                    report = await self.evaluate_single(
                        query=item["query"],
                        answer=item["answer"],
                        retrieved_contexts=item["retrieved_contexts"],
                        strategy_name=item.get("strategy_name"),
                        enabled_metrics=enabled_metrics
                    )
                    reports.append(report)
                    
                    self.logger.info(f"Batch progress: {i}/{len(items)}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to evaluate item {i}: {e}")
                    continue
        
        # Calculate aggregate metrics
        successful_reports = [
            r for r in reports 
            if r.overall_status != EvaluationStatus.FAILED
        ]
        
        def average_metric(metric_name: str) -> Optional[float]:
            """Calculate average for a specific metric."""
            scores = [
                getattr(r, metric_name).score
                for r in successful_reports
                if getattr(r, metric_name) and 
                   getattr(r, metric_name).status == EvaluationStatus.SUCCESS
            ]
            return sum(scores) / len(scores) if scores else None
        
        # Build batch result
        total_time = (time.time() - start_time) * 1000  # ms
        
        result = BatchEvaluationResult(
            batch_id=batch_id,
            timestamp=datetime.utcnow().isoformat(),
            total_queries=len(items),
            successful_evaluations=len(successful_reports),
            failed_evaluations=len(items) - len(successful_reports),
            average_overall_score=(
                sum(r.overall_score for r in successful_reports) / len(successful_reports)
                if successful_reports else 0.0
            ),
            average_faithfulness=average_metric("faithfulness"),
            average_answer_relevancy=average_metric("answer_relevancy"),
            average_context_precision=average_metric("context_precision"),
            reports=reports,
            total_execution_time_ms=total_time,
            average_execution_time_ms=(
                total_time / len(reports) if reports else 0.0
            )
        )
        
        self.logger.info(
            f"Batch evaluation {batch_id} complete: "
            f"{result.successful_evaluations}/{result.total_queries} successful, "
            f"avg_score={result.average_overall_score:.3f}, "
            f"time={total_time:.0f}ms"
        )
        
        return result