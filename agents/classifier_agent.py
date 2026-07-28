"""
MindSense AI - PyTorch Classifier Agent
===========================================
Agent wrapping the DistilBERT + BiLSTM + Attention PyTorch model
with token-level LIME feature attribution.

Author: MindSense AI Team
"""

from typing import Any, Dict
from model.classifier.loader import classifier_loader
from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger(__name__)


class ClassifierAgent:
    """Agent running the Hybrid PyTorch Mental Health Model."""

    def __init__(self) -> None:
        logger.debug("ClassifierAgent (PyTorch Hybrid) initialized.")

    @timer
    def run(self, user_message: str) -> Dict[str, Any]:
        if not user_message or not user_message.strip():
            return self._default_result()

        try:
            result = classifier_loader.predict(user_message)
            result["agent"] = "ClassifierAgent"

            logger.info(
                f"[ClassifierAgent] label={result['label']} | "
                f"confidence={result['confidence']:.3f} | "
                f"architecture={result.get('model_architecture')}"
            )
            return result

        except Exception as e:
            logger.error(f"[ClassifierAgent] Error: {e}")
            return self._default_result(error=str(e))

    def _default_result(self, error: str = "") -> Dict[str, Any]:
        return {
            "label": "Normal",
            "confidence": 0.0,
            "all_scores": {"Normal": 1.0},
            "explainable_words": [],
            "model_architecture": "DistilBERT + BiLSTM + Attention",
            "agent": "ClassifierAgent",
            "error": error,
        }
