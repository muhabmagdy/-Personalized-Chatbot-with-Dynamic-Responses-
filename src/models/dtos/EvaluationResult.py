from dataclasses import dataclass
from typing import Dict
from models.enums.EvaluationMethodEnum import EvaluationMethodEnum

@dataclass
class EvaluationResult:
    """
    Represents the result of a single evaluation metric.
    
    Attributes:
        metric_name: Name of the metric (e.g., "answer_relevance")
        score: Normalized score between 0.0 and 1.0
        method: Evaluation method used
        explanation: Human-readable explanation of the score
        metadata: Additional metric-specific data
    """
    metric_name: str
    score: float  # 0.0 to 1.0
    method: EvaluationMethodEnum
    explanation: str
    metadata: Dict[str, any]