# project/src/routes/evaluation.py

from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.evaluation import (
    EvaluateRAGResponseRequest,
    EvaluateRAGResponseResponse,
    BatchEvaluateRequest,
    BatchEvaluateResponse
)
from services.evaluation.RAGASEvaluatorService import RAGASEvaluatorService
from services.tracking.MLflowTrackingService import MLflowTrackingService
from models import ResponseSignal
import logging
import time

logger = logging.getLogger('uvicorn.error')

evaluation_router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["api_v1", "evaluation"],
)


def create_evaluation_service(request: Request) -> RAGASEvaluatorService:
    """Factory function to create evaluation service."""
    return RAGASEvaluatorService(
        llm_client=request.app.generation_client_for_evaluation,
        embedding_client=request.app.embedding_client,
        settings=request.app.settings
    )


def create_tracking_service(request: Request) -> MLflowTrackingService:
    """Factory function to create tracking service."""
    return MLflowTrackingService(settings=request.app.settings)


# ==========================================
# SINGLE EVALUATION ENDPOINT
# ==========================================
@evaluation_router.post("/evaluate")
async def evaluate_rag_response(
    request: Request,
    eval_request: EvaluateRAGResponseRequest
):
    """
    Evaluate a single RAG response.
    
    This endpoint accepts an already-generated answer and evaluates it
    using RAGAs metrics (faithfulness, answer_relevancy, context_precision).
    
    Workflow:
    1. Receive query, answer, and retrieved contexts
    2. Compute selected metrics
    3. Optionally log to MLflow
    4. Return evaluation report
    
    Request Body:
        {
            "query": "What is Python?",
            "answer": "Python is a programming language...",
            "retrieved_contexts": ["context1", "context2"],
            "strategy_name": "basic",
            "enabled_metrics": ["faithfulness", "answer_relevancy"],
            "log_to_mlflow": true,
            "mlflow_run_name": "eval-001"
        }
    
    Response:
        {
            "signal": "EVALUATION_SUCCESS",
            "evaluation_report": {...},
            "mlflow_run_id": "abc123",
            "mlflow_run_url": "https://dagshub.com/..."
        }
    """
    start_time = time.time()
    
    try:
        # Create evaluation service
        evaluator = create_evaluation_service(request)
        
        logger.info(
            f"Starting evaluation for query: '{eval_request.query[:50]}...'"
        )
        
        # Perform evaluation
        report = await evaluator.evaluate_single(
            query=eval_request.query,
            answer=eval_request.answer,
            retrieved_contexts=eval_request.retrieved_contexts,
            strategy_name=eval_request.strategy_name,
            enabled_metrics=eval_request.enabled_metrics
        )
        
        # Log to MLflow if requested
        mlflow_run_id = None
        mlflow_run_url = None
        
        if eval_request.log_to_mlflow:
            try:
                tracker = create_tracking_service(request)
                
                if tracker.is_tracking_enabled():
                    # Start run
                    run_name = (
                        eval_request.mlflow_run_name or 
                        f"eval_{report.evaluation_id}"
                    )
                    
                    mlflow_run_id = tracker.start_run(
                        run_name=run_name,
                        tags={
                            "evaluation_type": "single",
                            "strategy": eval_request.strategy_name or "unknown",
                            "evaluation_id": report.evaluation_id
                        }
                    )
                    
                    # Log metrics
                    tracker.log_metrics(report.to_mlflow_metrics())
                    
                    # Log parameters
                    tracker.log_params({
                        "query": eval_request.query[:100],  # Truncate
                        "strategy": eval_request.strategy_name or "unknown",
                        "num_contexts": len(eval_request.retrieved_contexts)
                    })
                    
                    # Log full report
                    tracker.log_evaluation_report(report.dict())
                    
                    # End run
                    tracker.end_run()
                    
                    # Get run URL
                    mlflow_run_url = tracker.get_run_url(mlflow_run_id)
                    
                    logger.info(f"Logged to MLflow: run_id={mlflow_run_id}")
                else:
                    logger.warning("MLflow tracking not enabled")
                    
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {e}")
                # Continue anyway - evaluation succeeded
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Evaluation complete: "
            f"score={report.overall_score:.3f}, "
            f"status={report.overall_status.value}, "
            f"time={execution_time:.0f}ms"
        )
        
        return JSONResponse(
            content={
                "signal": ResponseSignal.EVALUATION_SUCCESS.value,
                "evaluation_report": report.dict(),
                "mlflow_run_id": mlflow_run_id,
                "mlflow_run_url": mlflow_run_url,
                "execution_time_ms": execution_time
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "EVALUATION_VALIDATION_ERROR",
                "message": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "EVALUATION_FAILED",
                "message": str(e)
            }
        )


# ==========================================
# BATCH EVALUATION ENDPOINT
# ==========================================
@evaluation_router.post("/batch-evaluate")
async def batch_evaluate_rag_responses(
    request: Request,
    batch_request: BatchEvaluateRequest
):
    """
    Evaluate multiple RAG responses in batch.
    
    Useful for:
    - Testing multiple strategies on the same queries
    - Evaluating a test dataset
    - Comparing model versions
    
    Request Body:
        {
            "items": [
                {
                    "query": "What is Python?",
                    "answer": "Python is...",
                    "retrieved_contexts": ["..."],
                    "strategy_name": "basic"
                },
                ...
            ],
            "enabled_metrics": ["faithfulness", "answer_relevancy"],
            "log_to_mlflow": true,
            "mlflow_experiment_name": "batch-eval-001",
            "parallel": false
        }
    
    Response:
        {
            "signal": "BATCH_EVALUATION_SUCCESS",
            "batch_id": "batch_xyz789",
            "total_queries": 10,
            "successful_evaluations": 9,
            "failed_evaluations": 1,
            "average_overall_score": 0.81,
            "aggregate_metrics": {...},
            "reports": [...],
            "mlflow_experiment_url": "https://dagshub.com/..."
        }
    """
    start_time = time.time()
    
    try:
        # Create evaluation service
        evaluator = create_evaluation_service(request)
        
        logger.info(
            f"Starting batch evaluation: {len(batch_request.items)} items, "
            f"parallel={batch_request.parallel}"
        )
        
        # Perform batch evaluation
        batch_result = await evaluator.evaluate_batch(
            items=[item.dict() for item in batch_request.items],
            enabled_metrics=batch_request.enabled_metrics,
            parallel=batch_request.parallel
        )
        
        # Log to MLflow if requested
        mlflow_experiment_url = None
        
        if batch_request.log_to_mlflow:
            try:
                tracker = create_tracking_service(request)
                
                if tracker.is_tracking_enabled():
                    experiment_name = (
                        batch_request.mlflow_experiment_name or
                        request.app.settings.MLFLOW_EXPERIMENT_NAME
                    )
                    
                    # Log each evaluation as a separate run
                    for i, report in enumerate(batch_result.reports, 1):
                        run_name = f"batch_{batch_result.batch_id}_item_{i}"
                        
                        run_id = tracker.start_run(
                            run_name=run_name,
                            experiment_name=experiment_name,
                            tags={
                                "evaluation_type": "batch",
                                "batch_id": batch_result.batch_id,
                                "item_index": str(i)
                            }
                        )
                        
                        tracker.log_metrics(report.to_mlflow_metrics())
                        tracker.log_params({
                            "query": report.query[:100],
                            "strategy": report.strategy_name or "unknown"
                        })
                        
                        tracker.end_run()
                    
                    # Get experiment URL
                    mlflow_experiment_url = tracker.get_experiment_url(experiment_name)
                    
                    logger.info(
                        f"Logged {len(batch_result.reports)} runs to MLflow "
                        f"experiment: {experiment_name}"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to log batch to MLflow: {e}")
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch evaluation complete: "
            f"{batch_result.successful_evaluations}/{batch_result.total_queries} successful, "
            f"avg_score={batch_result.average_overall_score:.3f}, "
            f"time={execution_time:.0f}ms"
        )
        
        return JSONResponse(
            content={
                "signal": ResponseSignal.BATCH_EVALUATION_SUCCESS.value,
                "batch_id": batch_result.batch_id,
                "total_queries": batch_result.total_queries,
                "successful_evaluations": batch_result.successful_evaluations,
                "failed_evaluations": batch_result.failed_evaluations,
                "average_overall_score": batch_result.average_overall_score,
                "aggregate_metrics": {
                    "faithfulness": batch_result.average_faithfulness,
                    "answer_relevancy": batch_result.average_answer_relevancy,
                    "context_precision": batch_result.average_context_precision
                },
                "reports": [report.dict() for report in batch_result.reports],
                "mlflow_experiment_url": mlflow_experiment_url,
                "execution_time_ms": execution_time
            }
        )
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "BATCH_EVALUATION_FAILED",
                "message": str(e)
            }
        )


# ==========================================
# INFO ENDPOINTS
# ==========================================
@evaluation_router.get("/metrics")
async def get_available_metrics(request: Request):
    """
    Get list of available evaluation metrics.
    
    Response:
        {
            "signal": "SUCCESS",
            "metrics": ["faithfulness", "answer_relevancy", "context_precision"],
            "descriptions": {...}
        }
    """
    evaluator = create_evaluation_service(request)
    
    metrics_info = {
        "faithfulness": {
            "name": "Faithfulness",
            "description": "Measures if the answer is grounded in retrieved contexts",
            "range": [0.0, 1.0],
            "requires_ground_truth": False
        },
        "answer_relevancy": {
            "name": "Answer Relevancy",
            "description": "Measures how relevant the answer is to the query",
            "range": [0.0, 1.0],
            "requires_ground_truth": False
        },
        "context_precision": {
            "name": "Context Precision",
            "description": "Measures quality of retrieved context ranking",
            "range": [0.0, 1.0],
            "requires_ground_truth": False
        }
    }
    
    return JSONResponse(
        content={
            "signal": "SUCCESS",
            "metrics": evaluator.get_available_metrics(),
            "descriptions": metrics_info
        }
    )