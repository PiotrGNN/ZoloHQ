"""
ModelEvaluator - Model performance evaluation for trading AI
Supports accuracy, precision, recall, F1, custom metrics, logging, tracing, Sentry, and robust error handling.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import structlog
import sentry_sdk
from opentelemetry import trace
from typing import Any, Dict

logger = structlog.get_logger("ModelEvaluator")
tracer = trace.get_tracer("ai.models.ModelEvaluator")

class ModelEvaluator:
    """
    Model performance evaluation for trading AI.
    Supports accuracy, precision, recall, F1, custom metrics, logging, tracing, Sentry, and robust error handling.
    """
    def __init__(self) -> None:
        logger.info("ModelEvaluator_initialized")

    def evaluate(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Evaluate model predictions using standard metrics.
        Args:
            y_true (Any): True labels.
            y_pred (Any): Predicted labels.
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        with tracer.start_as_current_span("ModelEvaluator.evaluate"):
            try:
                result = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                    "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                    "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                }
                logger.info("evaluate_success", result=result)
                return result
            except Exception as e:
                logger.error("evaluate_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
