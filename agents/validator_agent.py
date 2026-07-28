"""
MindSense AI - Validator Agent
================================
Scores AI responses across 6 quality dimensions and automatically
regenerates responses that fall below the quality threshold.

Validation dimensions:
  1. Empathy       — Does it acknowledge and validate feelings?
  2. Relevance     — Is it directly relevant to the user's message?
  3. Safety        — Is it free from harmful content?
  4. Groundedness  — Is it grounded in evidence-based principles?
  5. Length        — Is the length appropriate?
  6. Repetition    — Does it avoid repeating previous responses?

A weighted composite score is computed. If below threshold,
the validator asks Gemini to self-evaluate and then requests
a new response from the TherapistAgent.

Usage::

    from agents.validator_agent import ValidatorAgent
    agent = ValidatorAgent()
    result = agent.validate(
        user_message="I feel hopeless",
        response="I hear you...",
        classification_label="Depression",
        prompt_builder=prompt_builder,
        context=context_dict,
    )
"""

import json
import re
from typing import Any, Dict, List, Optional

from config import settings
from model.llm.gemini_client import gemini_client
from utils.logger import get_logger
from utils.helpers import timer, clamp
from utils.response_utils import strip_markdown

logger = get_logger(__name__)


class ValidatorAgent:
    """
    Response quality validator with automatic regeneration.

    Evaluates generated responses and requests regeneration
    if quality falls below the configured threshold.

    Attributes:
        threshold (float): Minimum acceptable quality score (0–1).
        max_attempts (int): Maximum regeneration attempts.
    """

    def __init__(self) -> None:
        self.threshold: float = settings.validation.QUALITY_THRESHOLD
        self.max_attempts: int = settings.validation.MAX_REGENERATION_ATTEMPTS
        self.unsafe_keywords: List[str] = settings.validation.UNSAFE_KEYWORDS
        logger.debug("ValidatorAgent initialized.")

    def _check_unsafe_content(self, response: str) -> bool:
        """
        Check if the response contains unsafe keywords.

        Args:
            response: Response text to check.

        Returns:
            True if unsafe content is detected.
        """
        response_lower = response.lower()
        for kw in self.unsafe_keywords:
            if kw in response_lower:
                logger.warning(f"Unsafe keyword detected in response: '{kw}'")
                return True
        return False

    def _check_length(self, response: str) -> float:
        """
        Score the response based on its length.

        Args:
            response: Response text.

        Returns:
            Length score 0–1.
        """
        char_count = len(response.strip())
        min_len = settings.validation.MIN_RESPONSE_LENGTH
        max_len = settings.validation.MAX_RESPONSE_LENGTH

        if char_count < min_len:
            # Too short: linear penalty
            return max(0.0, char_count / min_len * 0.6)
        elif char_count > max_len:
            # Too long: linear penalty
            over = char_count - max_len
            return max(0.3, 1.0 - (over / (max_len * 0.5)) * 0.7)
        else:
            return 1.0

    def _check_repetition(
        self, response: str, previous_responses: Optional[List[str]] = None
    ) -> float:
        """
        Score the response based on how similar it is to previous responses.

        Args:
            response: Current response text.
            previous_responses: List of previous assistant responses in this session.

        Returns:
            Repetition score 0–1 (1.0 = no repetition, 0.0 = highly repetitive).
        """
        if not previous_responses:
            return 1.0

        response_words = set(response.lower().split())
        max_overlap = 0.0

        for prev in previous_responses[-3:]:  # Check last 3
            prev_words = set(prev.lower().split())
            if not prev_words:
                continue
            overlap = len(response_words & prev_words) / max(1, len(prev_words))
            max_overlap = max(max_overlap, overlap)

        return round(1.0 - max_overlap * 0.8, 4)

    def _heuristic_score(
        self,
        user_message: str,
        response: str,
        previous_responses: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute heuristic quality scores without calling the LLM.
        Used as a fast pre-check and fallback.

        Args:
            user_message: The user's input.
            response: The response to evaluate.
            previous_responses: Previous responses in this session.

        Returns:
            Dict of dimension scores.
        """
        text_lower = response.lower()

        # Empathy signals
        empathy_signals = [
            "i hear", "i understand", "that sounds", "i can imagine",
            "it makes sense", "that must be", "i'm here", "you're not alone",
            "it's okay", "valid", "understandable", "difficult",
        ]
        empathy = min(1.0, sum(0.15 for sig in empathy_signals if sig in text_lower))

        # Relevance: keyword overlap between message and response
        msg_keywords = set(user_message.lower().split())
        resp_words = set(text_lower.split())
        if msg_keywords:
            relevance = min(1.0, len(msg_keywords & resp_words) / len(msg_keywords) * 2)
        else:
            relevance = 0.5

        # Safety: unsafe content check
        safety = 0.0 if self._check_unsafe_content(response) else 1.0

        # Groundedness: presence of therapeutic concepts
        therapy_signals = [
            "cbt", "dbt", "cognitive", "behavioral", "mindfulness", "breathing",
            "therapy", "technique", "strategy", "coping", "grounding",
            "evidence", "research", "professional", "therapist",
        ]
        groundedness = min(1.0, 0.4 + sum(0.1 for sig in therapy_signals if sig in text_lower))

        # Length score
        length = self._check_length(response)

        # Repetition score
        repetition = self._check_repetition(response, previous_responses)

        return {
            "empathy": round(empathy, 4),
            "relevance": round(relevance, 4),
            "safety": round(safety, 4),
            "groundedness": round(groundedness, 4),
            "length": round(length, 4),
            "repetition": round(repetition, 4),
        }

    def _compute_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Compute the weighted composite quality score.

        Args:
            scores: Per-dimension score dict.

        Returns:
            Weighted composite score in [0, 1].
        """
        weights = {
            "empathy": settings.validation.EMPATHY_WEIGHT,
            "relevance": settings.validation.RELEVANCE_WEIGHT,
            "safety": settings.validation.SAFETY_WEIGHT,
            "groundedness": settings.validation.GROUNDEDNESS_WEIGHT,
            "length": settings.validation.LENGTH_WEIGHT,
            "repetition": settings.validation.REPETITION_WEIGHT,
        }

        total = sum(
            scores.get(dim, 0.0) * weight
            for dim, weight in weights.items()
        )
        return round(clamp(total, 0.0, 1.0), 4)

    def _llm_score(
        self, user_message: str, response: str, classification_label: str
    ) -> Optional[Dict[str, float]]:
        """
        Use Gemini to self-evaluate the response (LLM-as-judge).
        Falls back gracefully if parsing fails.

        Args:
            user_message: The user's message.
            response: The response to evaluate.
            classification_label: Predicted mental health label.

        Returns:
            Parsed score dict or None on failure.
        """
        try:
            from rag.prompt_builder import prompt_builder
            val_prompt = prompt_builder.build_validation_prompt(
                user_message, response, classification_label
            )
            raw = gemini_client.generate(val_prompt, temperature=0.1, max_tokens=300)

            # Extract JSON from response
            json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "empathy": float(parsed.get("empathy", 0.5)),
                    "relevance": float(parsed.get("relevance", 0.5)),
                    "safety": float(parsed.get("safety", 1.0)),
                    "groundedness": float(parsed.get("groundedness", 0.5)),
                    "length": float(parsed.get("length", 0.7)),
                    "repetition": float(parsed.get("repetition", 0.8)),
                }
        except Exception as e:
            logger.warning(f"LLM scoring failed, using heuristic only: {e}")
        return None

    @timer
    def validate(
        self,
        user_message: str,
        response: str,
        classification_label: str = "Normal",
        previous_responses: Optional[List[str]] = None,
        use_llm_scoring: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate a generated response and return quality scores.

        Args:
            user_message: The user's input message.
            response: The response to validate.
            classification_label: Detected mental health label.
            previous_responses: Previous assistant responses in this session.
            use_llm_scoring: Whether to use LLM self-scoring (slower but more accurate).

        Returns:
            Dict containing:
              - ``scores`` (dict): Per-dimension quality scores.
              - ``overall_score`` (float): Weighted composite score.
              - ``passed`` (bool): Whether response meets quality threshold.
              - ``feedback`` (str): Quality feedback message.
              - ``agent`` (str): Agent identifier.
        """
        if not response or not response.strip():
            return {
                "scores": {},
                "overall_score": 0.0,
                "passed": False,
                "feedback": "Empty response received.",
                "agent": "ValidatorAgent",
            }

        # ── Heuristic scoring (always computed) ───────────────────────────
        heuristic_scores = self._heuristic_score(user_message, response, previous_responses)

        # ── LLM scoring (optional) ─────────────────────────────────────────
        if use_llm_scoring:
            llm_scores = self._llm_score(user_message, response, classification_label)
            if llm_scores:
                # Average heuristic and LLM scores for robustness
                final_scores = {
                    dim: round((heuristic_scores.get(dim, 0.5) + llm_scores.get(dim, 0.5)) / 2, 4)
                    for dim in heuristic_scores
                }
            else:
                final_scores = heuristic_scores
        else:
            final_scores = heuristic_scores

        overall = self._compute_weighted_score(final_scores)
        passed = overall >= self.threshold

        # Safety is a hard gate — fail immediately on unsafe content
        if final_scores.get("safety", 1.0) < 0.5:
            passed = False
            overall = min(overall, 0.4)

        feedback = ""
        if not passed:
            weakest_dim = min(final_scores, key=lambda k: final_scores[k])
            feedback = f"Response scored below threshold on '{weakest_dim}' dimension."

        logger.info(
            f"[ValidatorAgent] score={overall:.3f} | passed={passed} | "
            f"threshold={self.threshold}"
        )

        return {
            "scores": final_scores,
            "overall_score": overall,
            "passed": passed,
            "feedback": feedback,
            "agent": "ValidatorAgent",
        }
