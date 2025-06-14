"""
ModelManager - Model lifecycle management for trading AI
Handles registration, loading, saving, versioning, logging, tracing, and robust error handling.
"""

import os
import joblib
import structlog
import sentry_sdk
from opentelemetry import trace
from typing import Any, Dict, List, Optional

logger = structlog.get_logger("ModelManager")
tracer = trace.get_tracer("ai.models.ModelManager")

class ModelManager:
    """
    Model lifecycle management for trading AI.
    Handles registration, loading, saving, versioning, logging, tracing, and robust error handling.
    """
    def __init__(self, model_dir: str = "ai_models") -> None:
        self.model_dir: str = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.registry: Dict[str, Any] = {}
        logger.info("ModelManager_initialized", model_dir=model_dir)

    def register(self, name: str, model: Any) -> None:
        """
        Register a model and save it.
        Args:
            name (str): Model name.
            model (Any): Model instance.
        """
        with tracer.start_as_current_span("ModelManager.register"):
            try:
                self.registry[name] = model
                self.save(name, model)
                logger.info("model_registered", name=name)
            except Exception as e:
                logger.error("register_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def save(self, name: str, model: Any) -> str:
        """
        Save model to disk.
        Args:
            name (str): Model name.
            model (Any): Model instance.
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
        Load model from disk and register it.
        Args:
            name (str): Model name.
        Returns:
            Optional[Any]: Loaded model or None.
        """
        try:
            path = os.path.join(self.model_dir, f"{name}.pkl")
            if os.path.exists(path):
                model = joblib.load(path)
                self.registry[name] = model
                logger.info("model_loaded", name=name)
                return model
            logger.warning("model_not_found", name=name)
            return None
        except Exception as e:
            logger.error("load_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return None

    def list_models(self) -> List[str]:
        """
        List all model files in the model directory.
        Returns:
            List[str]: List of model file names.
        """
        try:
            models = [f for f in os.listdir(self.model_dir) if f.endswith(".pkl")]
            logger.info("models_listed", count=len(models))
            return models
        except Exception as e:
            logger.error("list_models_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return []
