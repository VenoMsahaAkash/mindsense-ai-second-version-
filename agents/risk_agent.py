"""
MindSense AI - Risk Assessment Agent (Simplified)
===================================================
Evaluates emotional intensity / distress level without emergency crisis overrides.
Removed crisis popups/interrupts as requested.
"""

from typing import Any, Dict, List, Optional
from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger(__name__)


class RiskAgent:
    """Simplified distress level monitor (Low, Moderate, High)."""

    def __init__(self) -> None:
        logger.debug("RiskAgent initialized.")

    @timer
    def run(
        self,
        user_message: str,
        classification_label: str = "Normal",
        classification_confidence: float = 0.0,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        t_lower = user_message.lower()

        if any(w in t_lower for w in ["panic", "overwhelmed", "severe", "can't function"]):
            risk_level = "high"
            risk_score = 0.75
        elif any(w in t_lower for w in ["anxious", "depressed", "stressed", "worried"]):
            risk_level = "moderate"
            risk_score = 0.45
        else:
            risk_level = "low"
            risk_score = 0.15

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "is_crisis": False,  # Disabled crisis override
            "requires_crisis_protocol": False,
            "agent": "RiskAgent",
        }
