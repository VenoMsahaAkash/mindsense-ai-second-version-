"""
MindSense AI - Intent Agent
==============================
Detects the user's primary intent from their message.
Intent classification guides how the therapist agent structures its response.

Intents:
  - vent: User needs to be heard, not advised
  - seek_advice: User is asking for specific guidance
  - crisis: User is in acute distress or expressing self-harm ideation
  - info: User wants educational information about mental health
  - check_in: User is providing an update or checking in
  - general: Default when intent is unclear

Uses keyword + contextual heuristic matching (no external model required).

Usage::

    from agents.intent_agent import IntentAgent
    agent = IntentAgent()
    result = agent.run("I just need someone to talk to")
    # {"intent": "vent", "confidence": 0.85, ...}
"""

import re
from typing import Any, Dict, List, Tuple

from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent keyword patterns (ordered by priority — crisis first)
# ---------------------------------------------------------------------------
INTENT_PATTERNS: List[Tuple[str, List[str], float]] = [
    # (intent, keywords, base_confidence)
    (
        "crisis",
        [
            "kill myself", "want to die", "end my life", "suicide", "suicidal",
            "no reason to live", "better off dead", "hurt myself", "self-harm",
            "can't take it anymore", "give up on life", "overdose",
        ],
        0.95,
    ),
    (
        "seek_advice",
        [
            "what should i do", "how do i", "how can i", "any advice",
            "what can i do", "tips for", "help me with", "suggest",
            "recommend", "what works for", "strategies for", "techniques",
            "ways to cope", "how to deal", "how to handle",
        ],
        0.85,
    ),
    (
        "info",
        [
            "what is", "tell me about", "explain", "define", "meaning of",
            "what does", "how does", "is it normal", "is it common",
            "symptoms of", "signs of", "difference between", "learn about",
        ],
        0.80,
    ),
    (
        "vent",
        [
            "just want to talk", "need to vent", "nobody listens",
            "i feel", "i'm feeling", "i am feeling", "i've been feeling",
            "so tired of", "exhausted by", "can't stop thinking",
            "been struggling", "having a hard time", "going through",
            "not doing well", "feeling lost", "feeling alone",
        ],
        0.75,
    ),
    (
        "check_in",
        [
            "feeling better", "doing better", "update", "wanted to check in",
            "still struggling", "making progress", "had a good day",
            "had a bad day", "update you", "since last time",
        ],
        0.75,
    ),
]

# Question markers that suggest info-seeking or advice-seeking
QUESTION_PATTERNS = re.compile(
    r"\b(what|how|why|when|where|which|is it|can you|could you|do you)\b",
    re.IGNORECASE,
)


class IntentAgent:
    """
    Heuristic intent classifier for user messages.

    Detects whether the user wants to vent, seek advice, get information,
    check in, or is in crisis. Crisis intent always takes highest priority.
    """

    def __init__(self) -> None:
        logger.debug("IntentAgent initialized.")

    @timer
    def run(self, user_message: str, classification_label: str = "Normal") -> Dict[str, Any]:
        """
        Detect the primary intent of the user's message.

        Args:
            user_message: The user's current message text.
            classification_label: Pre-classified mental health label (from ClassifierAgent).
                                  Used to bias intent detection for high-risk labels.

        Returns:
            Dict containing:
              - ``intent`` (str): Detected intent string.
              - ``confidence`` (float): Confidence score 0–1.
              - ``secondary_intent`` (str|None): Second most likely intent.
              - ``agent`` (str): Agent identifier.

        Example::

            {"intent": "vent", "confidence": 0.82, "secondary_intent": "seek_advice"}
        """
        if not user_message or not user_message.strip():
            return self._default_result()

        msg_lower = user_message.lower().strip()
        intent_scores: Dict[str, float] = {
            "crisis": 0.0, "seek_advice": 0.0, "info": 0.0,
            "vent": 0.0, "check_in": 0.0, "general": 0.1,
        }

        # ── Keyword matching ───────────────────────────────────────────────
        for intent, keywords, base_conf in INTENT_PATTERNS:
            matched = sum(1 for kw in keywords if kw in msg_lower)
            if matched > 0:
                # Score scales with number of matched keywords
                score = min(base_conf, base_conf * (0.5 + matched * 0.25))
                intent_scores[intent] = max(intent_scores[intent], score)

        # ── Structural cues ────────────────────────────────────────────────
        # Question marks and question words → likely seeking advice or info
        if "?" in user_message:
            intent_scores["seek_advice"] = max(intent_scores["seek_advice"], 0.55)
            intent_scores["info"] = max(intent_scores["info"], 0.50)

        if QUESTION_PATTERNS.search(user_message):
            intent_scores["seek_advice"] = max(intent_scores["seek_advice"], 0.50)

        # ── Classification bias ────────────────────────────────────────────
        # Suicidal classification always bumps crisis intent
        if classification_label == "Suicidal":
            intent_scores["crisis"] = max(intent_scores["crisis"], 0.90)
        elif classification_label in ("Depression", "Bipolar"):
            intent_scores["vent"] = max(intent_scores["vent"], 0.60)
        elif classification_label == "Anxiety":
            intent_scores["seek_advice"] = max(intent_scores["seek_advice"], 0.55)

        # ── Pick top-2 intents ─────────────────────────────────────────────
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent, primary_conf = sorted_intents[0]
        secondary_intent = sorted_intents[1][0] if sorted_intents[1][1] > 0.3 else None

        logger.info(
            f"[IntentAgent] intent={primary_intent} | "
            f"confidence={primary_conf:.3f} | "
            f"secondary={secondary_intent}"
        )

        return {
            "intent": primary_intent,
            "confidence": round(primary_conf, 4),
            "secondary_intent": secondary_intent,
            "all_scores": {k: round(v, 4) for k, v in intent_scores.items()},
            "agent": "IntentAgent",
        }

    def _default_result(self) -> Dict[str, Any]:
        """Return a safe default intent result."""
        return {
            "intent": "general",
            "confidence": 0.5,
            "secondary_intent": None,
            "all_scores": {},
            "agent": "IntentAgent",
        }
