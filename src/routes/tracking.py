# project/src/routes/tracking.py

from fastapi import APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.evaluation import (
    LogMetricsRequest,
    GetExperimentInfoResponse,
    GetRunInfoResponse
)
from services.tracking.MLflowTrackingService import MLflowTrackingService
from models import ResponseSignal
import logging

logger = logging.getLogger('uvicorn.error')

tracking_router = APIRouter(
    prefix="/api/v1/tracking",
    tags=["api_v1", "tracking", "mlflow"],
)


def create_tracking_service(request: Request) -> MLflowTrackingService:
    """Factory function to create tracking service."""
    return MLflowTrackingService(settings=request.app.settings)


# ==========================================
# MLFLOW RUN MANAGEMENT
# ==========================================
@tracking_router.post("/log-metrics")
async def log_metrics_to_mlflow(
    request: Request,
    log_request: LogMetricsRequest
):
    """
    Manually log metrics, parameters, and tags to MLflow.
    
    Useful for custom tracking or integration with external systems.
    
    Request Body:
        {
            "run_id": "optional-existing-run-id",
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
            "run_name": "custom-run-001"
        }
    
    Response:
        {
            "signal": "METRICS_LOGGED_SUCCESS",
            "run_id": "abc123",
            "run_url": "https://dagshub.com/...",
            "metrics_count": 2,
            "params_count": 2,
            "tags_count": 2
        }
    """
    try:
        tracker = create_tracking_service(request)
        
        if not tracker.is_tracking_enabled():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "signal": "TRACKING_NOT_ENABLED",
                    "message": "MLflow tracking is not enabled or configured"
                }
            )
        
        # Determine run ID
        run_id = log_request.run_id
        
        if not run_id:
            # Start new run
            run_id = tracker.start_run(
                run_name=log_request.run_name,
                experiment_name=log_request.experiment_name,
                tags=log_request.tags
            )
            logger.info(f"Started new MLflow run: {run_id}")
        else:
            logger.info(f"Using existing MLflow run: {run_id}")
        
        # Log metrics
        if log_request.metrics:
            tracker.log_metrics(log_request.metrics, run_id=run_id)
            logger.debug(f"Logged {len(log_request.metrics)} metrics")
        
        # Log parameters
        if log_request.params:
            tracker.log_params(log_request.params, run_id=run_id)
            logger.debug(f"Logged {len(log_request.params)} parameters")
        
        # Log tags (if new run or additional tags)
        if log_request.tags and not log_request.run_id:
            tracker.log_tags(log_request.tags, run_id=run_id)
            logger.debug(f"Logged {len(log_request.tags)} tags")
        
        # End run if we created it
        if not log_request.run_id:
            tracker.end_run(run_id)
        
        # Get run URL
        run_url = tracker.get_run_url(run_id)
        
        return JSONResponse(
            content={
                "signal": ResponseSignal.METRICS_LOGGED_SUCCESS.value,
                "run_id": run_id,
                "run_url": run_url,
                "metrics_count": len(log_request.metrics) if log_request.metrics else 0,
                "params_count": len(log_request.params) if log_request.params else 0,
                "tags_count": len(log_request.tags) if log_request.tags else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "METRICS_LOGGING_FAILED",
                "message": str(e)
            }
        )


# ==========================================
# EXPERIMENT INFORMATION
# ==========================================
@tracking_router.get("/experiments/{experiment_name}")
async def get_experiment_info(
    request: Request,
    experiment_name: str
):
    """
    Get information about an MLflow experiment.
    
    Path Parameters:
        experiment_name: Name of the experiment
    
    Response:
        {
            "signal": "SUCCESS",
            "experiment_id": "1",
            "experiment_name": "rag-experiments",
            "artifact_location": "...",
            "lifecycle_stage": "active",
            "total_runs": 42,
            "tracking_uri": "https://dagshub.com/...",
            "experiment_url": "https://dagshub.com/..."
        }
    """
    try:
        tracker = create_tracking_service(request)
        
        if not tracker.is_tracking_enabled():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "signal": "TRACKING_NOT_ENABLED",
                    "message": "MLflow tracking is not enabled"
                }
            )
        
        # Get experiment info
        exp_info = tracker.get_experiment_info(experiment_name)
        
        if not exp_info:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": "EXPERIMENT_NOT_FOUND",
                    "message": f"Experiment '{experiment_name}' not found"
                }
            )
        
        # Get experiment URL
        exp_url = tracker.get_experiment_url(experiment_name)
        
        return JSONResponse(
            content={
                "signal": "SUCCESS",
                "experiment_id": exp_info["experiment_id"],
                "experiment_name": exp_info["experiment_name"],
                "artifact_location": exp_info["artifact_location"],
                "lifecycle_stage": exp_info["lifecycle_stage"],
                "total_runs": exp_info["total_runs"],
                "tracking_uri": request.app.settings.get_mlflow_tracking_uri(),
                "experiment_url": exp_url
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get experiment info: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "GET_EXPERIMENT_FAILED",
                "message": str(e)
            }
        )


@tracking_router.get("/experiments")
async def list_experiments(request: Request):
    """
    List all MLflow experiments.
    
    Response:
        {
            "signal": "SUCCESS",
            "experiments": [
                {
                    "name": "rag-experiments",
                    "id": "1",
                    "url": "https://dagshub.com/..."
                },
                ...
            ],
            "tracking_uri": "https://dagshub.com/..."
        }
    """
    try:
        tracker = create_tracking_service(request)
        
        if not tracker.is_tracking_enabled():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "signal": "TRACKING_NOT_ENABLED",
                    "message": "MLflow tracking is not enabled"
                }
            )
        
        # For now, return the default experiment
        # In a full implementation, you'd list all experiments
        default_exp = request.app.settings.MLFLOW_EXPERIMENT_NAME
        exp_info = tracker.get_experiment_info(default_exp)
        
        experiments = []
        if exp_info:
            experiments.append({
                "name": default_exp,
                "id": exp_info["experiment_id"],
                "url": tracker.get_experiment_url(default_exp)
            })
        
        return JSONResponse(
            content={
                "signal": "SUCCESS",
                "experiments": experiments,
                "tracking_uri": request.app.settings.get_mlflow_tracking_uri()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "LIST_EXPERIMENTS_FAILED",
                "message": str(e)
            }
        )


# ==========================================
# RUN INFORMATION
# ==========================================
@tracking_router.get("/runs/{run_id}")
async def get_run_info(
    request: Request,
    run_id: str
):
    """
    Get detailed information about an MLflow run.
    
    Path Parameters:
        run_id: MLflow run ID
    
    Response:
        {
            "signal": "SUCCESS",
            "run_id": "abc123",
            "run_name": "eval-001",
            "experiment_id": "1",
            "status": "FINISHED",
            "start_time": "2024-12-08T10:30:00Z",
            "end_time": "2024-12-08T10:31:00Z",
            "metrics": {...},
            "params": {...},
            "tags": {...},
            "artifact_uri": "...",
            "run_url": "https://dagshub.com/..."
        }
    """
    try:
        tracker = create_tracking_service(request)
        
        if not tracker.is_tracking_enabled():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "signal": "TRACKING_NOT_ENABLED",
                    "message": "MLflow tracking is not enabled"
                }
            )
        
        # Get run info
        run_info = tracker.get_run_info(run_id)
        
        if not run_info:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "signal": "RUN_NOT_FOUND",
                    "message": f"Run '{run_id}' not found"
                }
            )
        
        # Get run URL
        run_url = tracker.get_run_url(run_id)
        
        return JSONResponse(
            content={
                "signal": "SUCCESS",
                **run_info,
                "run_url": run_url
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get run info: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": "GET_RUN_FAILED",
                "message": str(e)
            }
        )


# ==========================================
# HEALTH CHECK
# ==========================================
@tracking_router.get("/health")
async def check_tracking_health(request: Request):
    """
    Check if MLflow tracking is properly configured and accessible.
    
    Response:
        {
            "signal": "SUCCESS",
            "tracking_enabled": true,
            "tracking_uri": "https://dagshub.com/...",
            "dagshub_configured": true,
            "default_experiment": "rag-experiments"
        }
    """
    tracker = create_tracking_service(request)
    settings = request.app.settings
    
    return JSONResponse(
        content={
            "signal": "SUCCESS",
            "tracking_enabled": tracker.is_tracking_enabled(),
            "tracking_uri": settings.get_mlflow_tracking_uri(),
            "dagshub_configured": settings.is_dagshub_configured(),
            "default_experiment": settings.MLFLOW_EXPERIMENT_NAME
        }
    )