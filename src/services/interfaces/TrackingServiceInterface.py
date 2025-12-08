# project/src/services/interfaces/TrackingServiceInterface.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime


class TrackingServiceInterface(ABC):
    """
    Abstract interface for experiment tracking services.
    
    Allows different tracking backends (MLflow, Weights & Biases, etc.)
    Follows Dependency Inversion Principle.
    """
    
    @abstractmethod
    def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new tracking run.
        
        Args:
            run_name: Name for this run
            experiment_name: Experiment to log to
            tags: Metadata tags
            
        Returns:
            Run ID
        """
        pass
    
    @abstractmethod
    def end_run(self, run_id: Optional[str] = None) -> None:
        """
        End the current or specified run.
        
        Args:
            run_id: Run to end (None for current)
        """
        pass
    
    @abstractmethod
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
            step: Optional step number
            run_id: Run to log to (None for current)
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_run_info(self, run_id: str) -> Dict:
        """
        Get information about a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with run information
        """
        pass
    
    @abstractmethod
    def get_experiment_info(self, experiment_name: str) -> Dict:
        """
        Get information about an experiment.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Dictionary with experiment information
        """
        pass
    
    @abstractmethod
    def get_run_url(self, run_id: str) -> Optional[str]:
        """
        Get web URL for viewing a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            URL string or None if not available
        """
        pass
    
    @abstractmethod
    def is_tracking_enabled(self) -> bool:
        """
        Check if tracking is enabled and configured.
        
        Returns:
            True if tracking is available
        """
        pass