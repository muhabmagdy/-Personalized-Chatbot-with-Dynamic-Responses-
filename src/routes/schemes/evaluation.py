# project/src/routes/schemes/evaluation.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

# ==========================================
# EVALUATION REQUEST SCHEMAS
# ==========================================

class EvaluateRAGResponseRequest(BaseModel):
    """
    Request to evaluate a single RAG response.
    
    This is for evaluating already-generated answers (decoupled from generation).
    """
    query: str = Field(
        ...,
        min_length=1,
        description="Original user query"
    )
    
    answer: str = Field(
        ...,
        min_length=1,
        description="Generated answer to evaluate"
    )
    
    retrieved_contexts: List[str] = Field(
        ...,
        min_items=1,
        description="List of retrieved document texts used to generate the answer"
    )
    
    strategy_name: Optional[str] = Field(
        None,
        description="Name of RAG strategy used (for metadata)"
    )
    
    enabled_metrics: Optional[List[str]] = Field(
        None,
        description="Specific metrics to compute. If None, computes all enabled metrics."
    )
    
    log_to_mlflow: bool = Field(
        True,
        description="Whether to log results to MLflow"
    )
    
    mlflow_run_name: Optional[str] = Field(
        None,
        description="Custom MLflow run name. Auto-generated if not provided."
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are the main features of Python?",
                "answer": "Python is a high-level, interpreted programming language with dynamic typing and automatic memory management.",
                "retrieved_contexts": [
                    "Python is known for its simple syntax and readability.",
                    "Key features include dynamic typing, garbage collection, and extensive standard library."
                ],
                "strategy_name": "basic",
                "enabled_metrics": ["faithfulness", "answer_relevancy"],
                "log_to_mlflow": True,
                "mlflow_run_name": "python-features-eval"
            }
        }


class BatchEvaluationItem(BaseModel):
    """Single item in a batch evaluation request."""
    query: str
    answer: str
    retrieved_contexts: List[str]
    strategy_name: Optional[str] = None
    metadata: Optional[Dict] = None


class BatchEvaluateRequest(BaseModel):
    """
    Request to evaluate multiple RAG responses in batch.
    
    Useful for:
    - Testing different strategies on the same queries
    - Evaluating a test dataset
    - Comparing model versions
    """
    items: List[BatchEvaluationItem] = Field(
        ...,
        min_items=1,
        description="List of items to evaluate"
    )
    
    enabled_metrics: Optional[List[str]] = Field(
        None,
        description="Metrics to compute for all items"
    )
    
    log_to_mlflow: bool = Field(
        True,
        description="Whether to log batch results to MLflow"
    )
    
    mlflow_experiment_name: Optional[str] = Field(
        None,
        description="MLflow experiment name for this batch"
    )
    
    parallel: bool = Field(
        False,
        description="Enable parallel evaluation (experimental)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "query": "What is Python?",
                        "answer": "Python is a programming language...",
                        "retrieved_contexts": ["Python is..."],
                        "strategy_name": "basic"
                    },
                    {
                        "query": "How does Python handle memory?",
                        "answer": "Python uses automatic memory management...",
                        "retrieved_contexts": ["Python memory..."],
                        "strategy_name": "fusion"
                    }
                ],
                "enabled_metrics": ["faithfulness", "answer_relevancy"],
                "log_to_mlflow": True,
                "mlflow_experiment_name": "python-qa-comparison"
            }
        }


# ==========================================
# EVALUATION RESPONSE SCHEMAS
# ==========================================

class EvaluateRAGResponseResponse(BaseModel):
    """Response from single evaluation."""
    signal: str = "EVALUATION_SUCCESS"
    evaluation_report: Dict  # RAGASEvaluationReport as dict
    mlflow_run_id: Optional[str] = None
    mlflow_run_url: Optional[str] = None


class BatchEvaluateResponse(BaseModel):
    """Response from batch evaluation."""
    signal: str = "BATCH_EVALUATION_SUCCESS"
    batch_id: str
    total_queries: int
    successful_evaluations: int
    failed_evaluations: int
    average_overall_score: float
    aggregate_metrics: Dict[str, float]
    reports: List[Dict]  # List of RAGASEvaluationReport as dicts
    mlflow_experiment_url: Optional[str] = None


# ==========================================
# MLFLOW TRACKING SCHEMAS
# ==========================================

class LogMetricsRequest(BaseModel):
    """Request to manually log metrics to MLflow."""
    run_id: Optional[str] = Field(
        None,
        description="MLflow run ID. Creates new run if not provided."
    )
    
    experiment_name: Optional[str] = Field(
        None,
        description="Experiment name. Uses default if not provided."
    )
    
    metrics: Dict[str, float] = Field(
        ...,
        description="Metrics to log (name: value)"
    )
    
    params: Optional[Dict[str, str]] = Field(
        None,
        description="Parameters to log"
    )
    
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Tags to add"
    )
    
    run_name: Optional[str] = Field(
        None,
        description="Run name"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "custom-experiment",
                "metrics": {
                    "accuracy": 0.95,
                    "latency_ms": 234.5
                },
                "params": {
                    "model": "gemini-1.5-flash",
                    "temperature": "0.7"
                },
                "tags": {
                    "environment": "production",
                    "version": "v1.2.0"
                },
                "run_name": "production-run-001"
            }
        }


class GetExperimentInfoResponse(BaseModel):
    """Response with experiment information."""
    signal: str = "SUCCESS"
    experiment_id: str
    experiment_name: str
    artifact_location: str
    lifecycle_stage: str
    total_runs: int
    tracking_uri: str
    experiment_url: Optional[str] = None


class GetRunInfoResponse(BaseModel):
    """Response with run information."""
    signal: str = "SUCCESS"
    run_id: str
    run_name: str
    experiment_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]
    artifact_uri: str
    run_url: Optional[str] = None


# ==========================================
# ANSWER GENERATION SCHEMAS (Updated)
# ==========================================

class GenerateAnswerRequest(BaseModel):
    """
    Request for pure answer generation (no evaluation).
    
    Separated from evaluation for clean separation of concerns.
    """
    text: str = Field(
        ...,
        min_length=1,
        description="User's question"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of documents to retrieve"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Chat session ID. Auto-generated if not provided."
    )
    
    rag_type: str = Field(
        default="basic",
        description="RAG strategy: basic, fusion, rerank, sentence_window, auto_merging, web_search"
    )
    
    chat_history_limit: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum number of previous messages to include"
    )
    
    log_to_mlflow: bool = Field(
        False,
        description="Log generation metadata to MLflow"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "text": "What are the main features of Python?",
                "limit": 5,
                "session_id": "user-123-conv-1",
                "rag_type": "fusion",
                "chat_history_limit": 5,
                "log_to_mlflow": False
            }
        }


class GenerateAnswerResponse(BaseModel):
    """Response from answer generation."""
    signal: str = "ANSWER_GENERATED_SUCCESS"
    answer: str
    session_id: str
    rag_strategy: str
    rag_type: str
    retrieved_documents_count: int
    chat_history_length: int
    execution_time_ms: float
    mlflow_run_id: Optional[str] = None
    
    # For potential evaluation
    full_prompt: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None


# ==========================================
# COMBINED WORKFLOW SCHEMAS
# ==========================================

class GenerateAndEvaluateRequest(BaseModel):
    """
    Request for combined generation + evaluation workflow.
    
    This is a convenience endpoint that chains:
    1. Answer generation
    2. Automatic evaluation
    3. MLflow logging
    """
    # Generation params
    text: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    session_id: Optional[str] = None
    rag_type: str = Field(default="basic")
    chat_history_limit: int = Field(default=10, ge=0, le=50)
    
    # Evaluation params
    enabled_metrics: Optional[List[str]] = None
    
    # MLflow params
    log_to_mlflow: bool = Field(True)
    mlflow_run_name: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "What is Python?",
                "limit": 5,
                "rag_type": "basic",
                "enabled_metrics": ["faithfulness", "answer_relevancy"],
                "log_to_mlflow": True,
                "mlflow_run_name": "python-qa-test"
            }
        }


class GenerateAndEvaluateResponse(BaseModel):
    """Response from combined workflow."""
    signal: str = "GENERATE_AND_EVALUATE_SUCCESS"
    
    # Generation results
    answer: str
    session_id: str
    rag_strategy: str
    execution_time_ms: float
    
    # Evaluation results
    evaluation_report: Dict
    overall_score: float
    
    # MLflow tracking
    mlflow_run_id: Optional[str] = None
    mlflow_run_url: Optional[str] = None