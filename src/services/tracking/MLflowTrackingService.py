# project/src/services/tracking/MLflowTrackingService.py

import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import os

from services.interfaces.TrackingServiceInterface import TrackingServiceInterface
from helpers.config import Settings


class MLflowTrackingService(TrackingServiceInterface):
    """
    MLflow tracking service with DagsHub integration.
    
    Features:
    - Automatic DagsHub authentication
    - Experiment management
    - Metrics, parameters, and tags logging
    - Run URL generation
    
    SOLID Principles:
    - Single Responsibility: Only handles experiment tracking
    - Open/Closed: Implements interface, closed for modification
    - Dependency Inversion: Depends on Settings abstraction
    - Interface Segregation: Focused tracking interface
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize MLflow tracking service.
        
        Args:
            settings: Application settings with MLflow/DagsHub config
        """
        self.settings = settings
        self.logger = logging.getLogger("uvicorn")
        self._client: Optional[MlflowClient] = None
        self._current_run_id: Optional[str] = None
        
        # Initialize if tracking is enabled
        if self.settings.MLFLOW_ENABLE_TRACKING:
            self._initialize_tracking()
    
    def _initialize_tracking(self) -> None:
        """Initialize MLflow tracking with DagsHub authentication."""
        try:
            # Set tracking URI
            tracking_uri = self.settings.get_mlflow_tracking_uri()
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            
            # Authenticate with DagsHub if configured
            if self.settings.is_dagshub_configured():
                credentials = self.settings.get_dagshub_credentials()
                os.environ['MLFLOW_TRACKING_USERNAME'] = credentials['username']
                os.environ['MLFLOW_TRACKING_PASSWORD'] = credentials['token']
                self.logger.info(f"DagsHub authentication configured for: {credentials['username']}")
            
            # Initialize client
            self._client = MlflowClient()
            
            # Set or create default experiment
            experiment_name = self.settings.MLFLOW_EXPERIMENT_NAME
            experiment = self._client.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = self._client.create_experiment(experiment_name)
                self.logger.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
            else:
                mlflow.set_experiment(experiment_name)
                self.logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
            self.logger.info("MLflow tracking initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow tracking: {e}")
            self.logger.warning("MLflow tracking will be disabled")
    
    def is_tracking_enabled(self) -> bool:
        """Check if tracking is enabled and properly configured."""
        return (
            self.settings.MLFLOW_ENABLE_TRACKING and 
            self._client is not None
        )
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run (auto-generated if None)
            experiment_name: Experiment to log to (uses default if None)
            tags: Metadata tags to add
            
        Returns:
            Run ID
            
        Raises:
            RuntimeError: If tracking is not enabled
        """
        if not self.is_tracking_enabled():
            raise RuntimeError("MLflow tracking is not enabled or configured")
        
        try:
            # Set experiment if provided
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            # Start run
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self._current_run_id = run.info.run_id
            
            self.logger.info(f"Started MLflow run: {self._current_run_id} (name: {run_name})")
            
            return self._current_run_id
            
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def end_run(self, run_id: Optional[str] = None) -> None:
        """
        End the current or specified MLflow run.
        
        Args:
            run_id: Run to end (None for current run)
        """
        if not self.is_tracking_enabled():
            return
        
        try:
            if run_id and run_id != self._current_run_id:
                # End specific run (requires reactivating it first)
                with mlflow.start_run(run_id=run_id):
                    mlflow.end_run()
            else:
                # End current run
                mlflow.end_run()
                self._current_run_id = None
            
            self.logger.info(f"Ended MLflow run: {run_id or 'current'}")
            
        except Exception as e:
            self.logger.error(f"Failed to end MLflow run: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> None:
        """
        Log metrics to a run.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for tracking over time
            run_id: Run to log to (None for current)
        """
        if not self.is_tracking_enabled():
            self.logger.warning("MLflow tracking disabled, skipping metric logging")
            return
        
        try:
            if run_id and run_id != self._current_run_id:
                # Log to specific run
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics(metrics, step=step)
            else:
                # Log to current run
                mlflow.log_metrics(metrics, step=step)
            
            self.logger.debug(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def log_params(
        self,
        params: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> None:
        """
        Log parameters to a run.
        
        Args:
            params: Dictionary of parameter names to values
            run_id: Run to log to (None for current)
        """
        if not self.is_tracking_enabled():
            return
        
        try:
            # Convert all values to strings (MLflow requirement)
            str_params = {k: str(v) for k, v in params.items()}
            
            if run_id and run_id != self._current_run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_params(str_params)
            else:
                mlflow.log_params(str_params)
            
            self.logger.debug(f"Logged {len(params)} parameters to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters to MLflow: {e}")
    
    def log_tags(
        self,
        tags: Dict[str, str],
        run_id: Optional[str] = None
    ) -> None:
        """
        Log tags to a run.
        
        Args:
            tags: Dictionary of tag names to values
            run_id: Run to log to (None for current)
        """
        if not self.is_tracking_enabled():
            return
        
        try:
            if run_id and run_id != self._current_run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.set_tags(tags)
            else:
                mlflow.set_tags(tags)
            
            self.logger.debug(f"Logged {len(tags)} tags to MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to log tags to MLflow: {e}")
    
    def log_text(
        self,
        text: str,
        artifact_file: str,
        run_id: Optional[str] = None
    ) -> None:
        """
        Log text content as an artifact.
        
        Args:
            text: Text content to log
            artifact_file: Filename for the artifact
            run_id: Run to log to (None for current)
        """
        if not self.is_tracking_enabled():
            return
        
        try:
            if run_id and run_id != self._current_run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_text(text, artifact_file)
            else:
                mlflow.log_text(text, artifact_file)
            
            self.logger.debug(f"Logged text artifact: {artifact_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to log text artifact: {e}")
    
    def get_run_info(self, run_id: str) -> Dict:
        """
        Get detailed information about a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with run information
        """
        if not self.is_tracking_enabled():
            return {}
        
        try:
            run = self._client.get_run(run_id)
            
            return {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "artifact_uri": run.info.artifact_uri
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get run info: {e}")
            return {}
    
    def get_experiment_info(self, experiment_name: str) -> Dict:
        """
        Get information about an experiment.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Dictionary with experiment information
        """
        if not self.is_tracking_enabled():
            return {}
        
        try:
            experiment = self._client.get_experiment_by_name(experiment_name)
            
            if not experiment:
                return {}
            
            # Get run count
            runs = self._client.search_runs(
                experiment_ids=[experiment.experiment_id]
            )
            
            return {
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "total_runs": len(runs)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment info: {e}")
            return {}
    
    def get_run_url(self, run_id: str) -> Optional[str]:
        """
        Get DagsHub web URL for viewing a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            URL string or None if not available
        """
        if not self.settings.is_dagshub_configured():
            return None
        
        try:
            run = self._client.get_run(run_id)
            experiment_id = run.info.experiment_id
            
            return (
                f"https://dagshub.com/{self.settings.DAGSHUB_USERNAME}/"
                f"{self.settings.DAGSHUB_REPO_NAME}/experiments/"
                f"#/experiments/{experiment_id}/runs/{run_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate run URL: {e}")
            return None
    
    def get_experiment_url(self, experiment_name: str) -> Optional[str]:
        """
        Get DagsHub web URL for viewing an experiment.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            URL string or None if not available
        """
        if not self.settings.is_dagshub_configured():
            return None
        
        try:
            experiment = self._client.get_experiment_by_name(experiment_name)
            
            if not experiment:
                return None
            
            return (
                f"https://dagshub.com/{self.settings.DAGSHUB_USERNAME}/"
                f"{self.settings.DAGSHUB_REPO_NAME}/experiments/"
                f"#/experiments/{experiment.experiment_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate experiment URL: {e}")
            return None
    
    def log_evaluation_report(
        self,
        report_dict: Dict,
        run_id: Optional[str] = None
    ) -> None:
        """
        Log evaluation report to MLflow.
        
        This is a convenience method for logging RAGAs evaluation results.
        
        Args:
            report_dict: Evaluation report as dictionary
            run_id: Run to log to (None for current)
        """
        if not self.is_tracking_enabled():
            return
        
        if "timestamp" in report_dict and isinstance(report_dict["timestamp"], datetime):
            report_dict["timestamp"] = report_dict["timestamp"].isoformat()
        
        try:
            # Extract metrics
            metrics = {}
            if "overall_score" in report_dict:
                metrics["overall_score"] = report_dict["overall_score"]
            
            if "faithfulness" in report_dict and report_dict["faithfulness"]:
                metrics["faithfulness"] = report_dict["faithfulness"]["score"]
            
            if "answer_relevancy" in report_dict and report_dict["answer_relevancy"]:
                metrics["answer_relevancy"] = report_dict["answer_relevancy"]["score"]
            
            if "context_precision" in report_dict and report_dict["context_precision"]:
                metrics["context_precision"] = report_dict["context_precision"]["score"]
            
            if "total_execution_time_ms" in report_dict:
                metrics["execution_time_ms"] = report_dict["total_execution_time_ms"]
            
            # Log metrics
            self.log_metrics(metrics, run_id=run_id)
            
            # Log report as artifact
            import json
            report_json = json.dumps(report_dict, indent=2)
            self.log_text(
                report_json,
                "evaluation_report.json",
                run_id=run_id
            )
            
            self.logger.info(f"Logged evaluation report to MLflow (run: {run_id or 'current'})")
            
        except Exception as e:
            self.logger.error(f"Failed to log evaluation report: {e}")