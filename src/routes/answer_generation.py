# project/src/routes/answer_generation.py

from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.evaluation import (
    GenerateAnswerRequest,
    GenerateAnswerResponse,
    GenerateAndEvaluateRequest,
    GenerateAndEvaluateResponse
)
from models.ProjectModel import ProjectModel
from services.answer_generation.AnswerGenerationService import AnswerGenerationService
from services.evaluation.RAGASEvaluatorService import RAGASEvaluatorService
from services.tracking.MLflowTrackingService import MLflowTrackingService
from services.chat_memory.DatabaseChatMemory import DatabaseChatMemory
from models import ResponseSignal
import logging
import uuid
import time

logger = logging.getLogger('uvicorn.error')

answer_router = APIRouter(
    prefix="/api/v1/answer",
    tags=["api_v1", "answer_generation"],
)


def create_answer_service(request: Request) -> AnswerGenerationService:
    """Factory function to create answer generation service."""
    chat_memory = DatabaseChatMemory(db_client=request.app.db_client)
    
    return AnswerGenerationService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        chat_memory=chat_memory
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
# PURE ANSWER GENERATION (No Evaluation)
# ==========================================
@answer_router.post("/generate/{project_id}")
async def generate_answer(
    request: Request,
    project_id: int,
    gen_request: GenerateAnswerRequest
):
    """
    Generate answer using RAG without evaluation.
    
    This is the core answer generation endpoint. It:
    1. Retrieves relevant documents
    2. Generates answer using specified RAG strategy
    3. Manages chat history
    4. Optionally logs to MLflow
    
    Does NOT perform evaluation (use /evaluate or /generate-and-evaluate for that).
    
    Request Body:
        {
            "text": "What is Python?",
            "limit": 5,
            "session_id": "optional-session-id",
            "rag_type": "basic",
            "chat_history_limit": 5,
            "log_to_mlflow": false
        }
    
    Response:
        {
            "signal": "ANSWER_GENERATED_SUCCESS",
            "answer": "Python is...",
            "session_id": "abc-123",
            "rag_strategy": "Basic RAG",
            "rag_type": "basic",
            "retrieved_documents_count": 5,
            "chat_history_length": 3,
            "execution_time_ms": 1234.5,
            "mlflow_run_id": null,
            "full_prompt": "...",
            "retrieved_contexts": ["...", "..."]
        }
    """
    start_time = time.time()
    
    try:
        # Get project
        project_model = await ProjectModel.create_instance(
            db_client=request.app.db_client
        )
        project = await project_model.get_project_or_create_one(project_id=project_id)
        
        if not project:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value,
                    "message": f"Project {project_id} not found"
                }
            )
        
        # Generate session ID if not provided
        session_id = gen_request.session_id or str(uuid.uuid4())
        
        # Create answer generation service
        answer_service = create_answer_service(request)
        
        logger.info(
            f"Generating answer for project {project_id}, "
            f"query: '{gen_request.text[:50]}...', "
            f"strategy: {gen_request.rag_type}"
        )
        
        # Generate answer
        result = await answer_service.generate_answer(
            project=project,
            query=gen_request.text,
            session_id=session_id,
            limit=gen_request.limit,
            rag_type=gen_request.rag_type,
            chat_history_limit=gen_request.chat_history_limit
        )
        
        # Log to MLflow if requested
        mlflow_run_id = None
        
        if gen_request.log_to_mlflow:
            try:
                tracker = create_tracking_service(request)
                
                if tracker.is_tracking_enabled():
                    run_id = tracker.start_run(
                        run_name=f"generation_{session_id[:8]}",
                        tags={
                            "type": "generation",
                            "project_id": str(project_id),
                            "strategy": gen_request.rag_type
                        }
                    )
                    
                    tracker.log_metrics({
                        "execution_time_ms": result["execution_time_ms"],
                        "retrieved_docs": float(result["retrieved_documents_count"]),
                        "answer_length": float(len(result["answer"]))
                    })
                    
                    tracker.log_params({
                        "query": gen_request.text[:100],
                        "rag_type": gen_request.rag_type,
                        "limit": str(gen_request.limit)
                    })
                    
                    tracker.end_run()
                    mlflow_run_id = run_id
                    
                    logger.info(f"Logged generation to MLflow: {run_id}")
                    
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Answer generated successfully: "
            f"{len(result['answer'])} chars, "
            f"{result['retrieved_documents_count']} docs, "
            f"{total_time:.0f}ms"
        )
        
        return JSONResponse(
            content={
                "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
                "answer": result["answer"],
                "session_id": result["session_id"],
                "rag_strategy": result["strategy_name"],
                "rag_type": result["rag_type"],
                "retrieved_documents_count": result["retrieved_documents_count"],
                "chat_history_length": result["chat_history_length"],
                "execution_time_ms": total_time,
                "mlflow_run_id": mlflow_run_id,
                "full_prompt": result.get("full_prompt"),
                "retrieved_contexts": result.get("retrieved_contexts")
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "GENERATION_VALIDATION_ERROR",
                "message": str(e)
            }
        )
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": ResponseSignal.RAG_ANSWER_ERROR.value,
                "message": str(e)
            }
        )


# ==========================================
# COMBINED: GENERATE + EVALUATE
# ==========================================
@answer_router.post("/generate-and-evaluate/{project_id}")
async def generate_and_evaluate(
    request: Request,
    project_id: int,
    gen_eval_request: GenerateAndEvaluateRequest
):
    """
    Generate answer and automatically evaluate it.
    
    This is a convenience endpoint that combines:
    1. Answer generation
    2. Automatic RAGAs evaluation
    3. MLflow logging (optional)
    
    Useful for getting immediate feedback on answer quality.
    
    Request Body:
        {
            "text": "What is Python?",
            "limit": 5,
            "rag_type": "basic",
            "enabled_metrics": ["faithfulness", "answer_relevancy"],
            "log_to_mlflow": true,
            "mlflow_run_name": "python-qa-001"
        }
    
    Response:
        {
            "signal": "GENERATE_AND_EVALUATE_SUCCESS",
            "answer": "Python is...",
            "session_id": "abc-123",
            "rag_strategy": "Basic RAG",
            "execution_time_ms": 2345.6,
            "evaluation_report": {...},
            "overall_score": 0.85,
            "mlflow_run_id": "xyz789",
            "mlflow_run_url": "https://dagshub.com/..."
        }
    """
    start_time = time.time()
    
    try:
        # Get project
        project_model = await ProjectModel.create_instance(
            db_client=request.app.db_client
        )
        project = await project_model.get_project_or_create_one(project_id=project_id)
        
        if not project:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
            )
        
        # Generate session ID
        session_id = gen_eval_request.session_id or str(uuid.uuid4())
        
        # Step 1: Generate answer
        answer_service = create_answer_service(request)
        
        logger.info(
            f"Generating and evaluating answer for project {project_id}, "
            f"strategy: {gen_eval_request.rag_type}"
        )
        
        gen_result = await answer_service.generate_answer(
            project=project,
            query=gen_eval_request.text,
            session_id=session_id,
            limit=gen_eval_request.limit,
            rag_type=gen_eval_request.rag_type,
            chat_history_limit=gen_eval_request.chat_history_limit
        )
        
        # Step 2: Evaluate answer
        evaluator = create_evaluation_service(request)
        
        eval_report = await evaluator.evaluate_single(
            query=gen_eval_request.text,
            answer=gen_result["answer"],
            retrieved_contexts=gen_result["retrieved_contexts"],
            strategy_name=gen_result["strategy_name"],
            enabled_metrics=gen_eval_request.enabled_metrics
        )
        
        # Step 3: Log to MLflow if requested
        mlflow_run_id = None
        mlflow_run_url = None
        
        if gen_eval_request.log_to_mlflow:
            try:
                tracker = create_tracking_service(request)
                
                if tracker.is_tracking_enabled():
                    run_name = (
                        gen_eval_request.mlflow_run_name or
                        f"gen_eval_{session_id[:8]}"
                    )
                    
                    run_id = tracker.start_run(
                        run_name=run_name,
                        tags={
                            "type": "generation_and_evaluation",
                            "project_id": str(project_id),
                            "strategy": gen_eval_request.rag_type,
                            "evaluation_id": eval_report.evaluation_id
                        }
                    )
                    
                    # Log generation metrics
                    tracker.log_metrics({
                        "generation_time_ms": gen_result["execution_time_ms"],
                        "retrieved_docs": float(gen_result["retrieved_documents_count"]),
                        "answer_length": float(len(gen_result["answer"]))
                    })
                    
                    # Log evaluation metrics
                    tracker.log_metrics(eval_report.to_mlflow_metrics())
                    
                    # Log parameters
                    tracker.log_params({
                        "query": gen_eval_request.text[:100],
                        "rag_type": gen_eval_request.rag_type,
                        "limit": str(gen_eval_request.limit)
                    })
                    
                    # Log full report
                    tracker.log_evaluation_report(eval_report.dict())
                    
                    tracker.end_run()
                    
                    mlflow_run_id = run_id
                    mlflow_run_url = tracker.get_run_url(run_id)
                    
                    logger.info(f"Logged combined workflow to MLflow: {run_id}")
                    
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"Generation and evaluation complete: "
            f"score={eval_report.overall_score:.3f}, "
            f"time={total_time:.0f}ms"
        )
        
        return JSONResponse(
            content={
                "signal": "GENERATE_AND_EVALUATE_SUCCESS",
                "answer": gen_result["answer"],
                "session_id": session_id,
                "rag_strategy": gen_result["strategy_name"],
                "rag_type": gen_result["rag_type"],
                "execution_time_ms": total_time,
                "evaluation_report": eval_report.dict(),
                "overall_score": eval_report.overall_score,
                "mlflow_run_id": mlflow_run_id,
                "mlflow_run_url": mlflow_run_url
            }
        )
        
    except Exception as e:
        logger.error(f"Generation and evaluation failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "GENERATE_AND_EVALUATE_FAILED",
                "message": str(e)
            }
        )


# ==========================================
# CHAT SESSION MANAGEMENT
# ==========================================
@answer_router.delete("/session/{project_id}/{session_id}")
async def clear_chat_session(
    request: Request,
    project_id: int,
    session_id: str
):
    """Clear chat history for a specific session."""
    answer_service = create_answer_service(request)
    
    success = await answer_service.clear_chat_session(
        session_id=session_id,
        project_id=project_id
    )
    
    if success:
        return JSONResponse(
            content={
                "signal": "CHAT_SESSION_CLEARED",
                "session_id": session_id
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": "CHAT_SESSION_CLEAR_ERROR"}
        )


@answer_router.get("/sessions/{project_id}")
async def get_chat_sessions(
    request: Request,
    project_id: int,
    limit: int = 10
):
    """Get list of recent chat sessions for a project."""
    answer_service = create_answer_service(request)
    
    sessions = await answer_service.get_chat_sessions(
        project_id=project_id,
        limit=limit
    )
    
    return JSONResponse(
        content={
            "signal": "CHAT_SESSIONS_RETRIEVED",
            "project_id": project_id,
            "sessions": sessions,
            "count": len(sessions)
        }
    )