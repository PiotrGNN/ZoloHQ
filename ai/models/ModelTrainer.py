"""
ModelTrainer - Model training and lifecycle management for trading AI
Supports advanced training, saving, loading, evaluation, logging, tracing, and robust error handling.
"""

import joblib
import os
import structlog
import sentry_sdk
from opentelemetry import trace
from typing import Any, Optional

logger = structlog.get_logger("ModelTrainer")
tracer = trace.get_tracer("ai.models.ModelTrainer")

class ModelTrainer:
    """
    Model training and lifecycle management for trading AI.
    Supports advanced training, saving, loading, evaluation, logging, tracing, and robust error handling.
    """
    def __init__(self, model_dir: str = "ai_models") -> None:
        """
        Initialize ModelTrainer.
        Args:
            model_dir (str): Directory to save/load models.
        """
        self.model_dir: str = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("ModelTrainer_initialized", model_dir=model_dir)

    def train(self, model: Any, X: Any, y: Any) -> Any:
        """
        Train model on X, y.
        Args:
            model (Any): Model instance.
            X (Any): Training features.
            y (Any): Training labels.
        Returns:
            Any: Trained model.
        """
        with tracer.start_as_current_span("ModelTrainer.train"):
            try:
                model.fit(X, y)
                logger.info("model_trained")
                return model
            except Exception as e:
                logger.error("train_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def save(self, model: Any, name: str) -> str:
        """
        Save model to disk.
        Args:
            model (Any): Model instance.
            name (str): Model name.
        Returns:
            str: Path to saved model.
        """
        try:
            path = os.path.join(self.model_dir, f"{name}.pkl")
            joblib.dump(model, path)
            logger.info("model_saved", path=path)
            return path
        except Exception as e:
            logger.error("save_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            raise

    def load(self, name: str) -> Optional[Any]:
        """
        Load model from disk.
        Args:
            name (str): Model name.
        Returns:
            Optional[Any]: Loaded model or None.
        """
        try:
            path = os.path.join(self.model_dir, f"{name}.pkl")
            if os.path.exists(path):
                logger.info("model_loaded", name=name)
                return joblib.load(path)
            logger.warning("model_not_found", name=name)
            return None
        except Exception as e:
            logger.error("load_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return None

    def evaluate(self, model: Any, X: Any, y: Any) -> float:
        """
        Evaluate model performance.
        Args:
            model (Any): Model instance.
            X (Any): Features.
            y (Any): Labels.
        Returns:
            float: Evaluation score.
        """
        with tracer.start_as_current_span("ModelTrainer.evaluate"):
            try:
                score = model.score(X, y)
                logger.info("model_evaluated", score=score)
                return score
            except Exception as e:
                logger.error("evaluate_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                return 0.0
