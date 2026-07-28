"""
MindSense AI - Analyzer Agent
================================
Performs deep NLP analysis of the user's message to extract:
  - Dominant emotion (from 8 base emotions)
  - Sentiment polarity (positive / neutral / negative)
  - Recurring themes and topics
  - Cognitive distortions detected

This analysis enriches the prompt context, allowing the therapist
agent to produce more personalized and targeted responses.

Uses heuristic analysis (no external model required) with rule-based
emotion detection from the NRC Emotion Lexicon principles.

Usage::

    from agents.analyzer_agent import AnalyzerAgent
    agent = AnalyzerAgent()
    result = agent.run("I feel so hopeless and trapped — nothing ever works for me")
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.helpers import timer
from utils.preprocessing import extract_keywords

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Emotion lexicon (NRC-inspired keyword sets)
# ---------------------------------------------------------------------------
EMOTION_LEXICON: Dict[str, List[str]] = {
    "sadness": [
        "sad", "depressed", "miserable", "unhappy", "heartbroken", "grief", "sorrow",
        "hopeless", "empty", "worthless", "alone", "lonely", "crying", "tears",
        "devastated", "crushed", "broken", "disappointed", "gloomy", "melancholy",
    ],
    "anxiety": [
        "anxious", "worried", "nervous", "scared", "fear", "panic", "dread",
        "apprehensive", "restless", "tense", "uneasy", "jittery", "overwhelmed",
        "catastrophe", "worst case", "what if", "can't breathe", "heart racing",
    ],
    "anger": [
        "angry", "furious", "rage", "mad", "frustrated", "irritated", "annoyed",
        "resentful", "bitter", "outraged", "hate", "fed up", "livid",
    ],
    "fear": [
        "terrified", "afraid", "frightened", "scared", "horror", "phobia",
        "threatening", "dangerous", "unsafe", "helpless",
    ],
    "disgust": [
        "disgusted", "revolted", "repulsed", "sick of", "hate", "loathe",
        "can't stand", "nauseated",
    ],
    "joy": [
        "happy", "joyful", "excited", "elated", "grateful", "thankful", "great",
        "wonderful", "amazing", "fantastic", "good", "better", "hopeful", "proud",
    ],
    "trust": [
        "trust", "safe", "secure", "supported", "understood", "heard", "cared for",
        "loved", "accepted", "belonging",
    ],
    "anticipation": [
        "hope", "looking forward", "excited about", "planning", "goal", "future",
        "expect", "await",
    ],
}

# Cognitive distortions keywords (Beck's cognitive model)
COGNITIVE_DISTORTIONS: Dict[str, List[str]] = {
    "catastrophizing": [
        "always", "never", "worst", "disaster", "catastrophe", "terrible",
        "horrible", "ruined", "destroyed",
    ],
    "black_and_white_thinking": [
        "all or nothing", "completely", "total failure", "perfect or", "either",
        "never do anything right", "always wrong",
    ],
    "mind_reading": [
        "they think", "everyone thinks", "they hate", "nobody likes",
        "they're judging", "people think",
    ],
    "fortune_telling": [
        "it will never", "nothing will change", "it's going to get worse",
        "i know it won't work", "this will fail",
    ],
    "personalization": [
        "it's my fault", "i caused", "i'm to blame", "because of me",
        "i ruined", "my fault",
    ],
    "emotional_reasoning": [
        "i feel stupid so i am", "i feel worthless", "i feel like a failure",
        "i feel like nobody", "because i feel",
    ],
    "overgeneralization": [
        "i always fail", "i never succeed", "this always happens to me",
        "nothing ever works", "things never get better",
    ],
    "mental_filter": [
        "only the bad", "even one bad thing", "can't see anything good",
        "all i see is", "nothing good",
    ],
}

# Theme vocabulary
THEME_PATTERNS: Dict[str, List[str]] = {
    "relationships": [
        "relationship", "partner", "boyfriend", "girlfriend", "spouse", "husband",
        "wife", "breakup", "divorce", "friendship", "family", "parents", "siblings",
    ],
    "work_school": [
        "work", "job", "boss", "coworker", "career", "school", "university",
        "exam", "grades", "study", "deadline", "fired", "quit",
    ],
    "self_worth": [
        "worthless", "useless", "failure", "not good enough", "stupid", "ugly",
        "nobody likes", "hate myself",
    ],
    "sleep": [
        "sleep", "insomnia", "can't sleep", "nightmares", "exhausted", "tired all the time",
        "fatigue", "no energy",
    ],
    "physical_health": [
        "pain", "sick", "illness", "hospital", "medication", "headache",
        "physical", "body", "appetite", "eating",
    ],
    "isolation": [
        "alone", "lonely", "no friends", "isolated", "nobody", "no one", "by myself",
        "nobody cares", "left out",
    ],
    "loss": [
        "loss", "death", "died", "grief", "mourning", "lost", "passed away",
        "missing", "gone",
    ],
    "trauma": [
        "trauma", "abuse", "assault", "ptsd", "flashback", "nightmare",
        "attacked", "violated",
    ],
}


class AnalyzerAgent:
    """
    Deep NLP analyzer for emotion, sentiment, distortion, and theme detection.

    All analysis is performed using rule-based heuristics — no
    external models are required. This ensures fast, reliable inference
    even in offline environments.
    """

    def __init__(self) -> None:
        logger.debug("AnalyzerAgent initialized.")

    def _detect_emotions(self, text_lower: str) -> Dict[str, float]:
        """Compute per-emotion scores from the lexicon."""
        scores: Dict[str, float] = {emotion: 0.0 for emotion in EMOTION_LEXICON}
        word_count = max(1, len(text_lower.split()))

        for emotion, keywords in EMOTION_LEXICON.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                scores[emotion] = min(1.0, matches / max(1, len(keywords) * 0.2))

        # Normalize to sum ≤ 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 4) for k, v in scores.items()}

        return scores

    def _detect_sentiment(self, emotion_scores: Dict[str, float]) -> str:
        """
        Infer overall sentiment from emotion distribution.

        Returns:
            "positive", "negative", or "neutral".
        """
        positive_emotions = {"joy", "trust", "anticipation"}
        negative_emotions = {"sadness", "anxiety", "anger", "fear", "disgust"}

        pos_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
        neg_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)

        if neg_score > 0.5:
            return "negative"
        elif pos_score > 0.5:
            return "positive"
        elif neg_score > pos_score:
            return "negative"
        elif pos_score > neg_score:
            return "positive"
        else:
            return "neutral"

    def _detect_distortions(self, text_lower: str) -> List[str]:
        """Detect cognitive distortion patterns in the text."""
        detected = []
        for distortion, phrases in COGNITIVE_DISTORTIONS.items():
            if any(phrase in text_lower for phrase in phrases):
                detected.append(distortion.replace("_", " "))
        return detected

    def _detect_themes(self, text_lower: str) -> List[str]:
        """Detect recurring life themes from the text."""
        detected = []
        for theme, keywords in THEME_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(theme.replace("_", " "))
        return detected

    @timer
    def run(self, user_message: str) -> Dict[str, Any]:
        """
        Perform comprehensive NLP analysis of the user's message.

        Args:
            user_message: The user's input text.

        Returns:
            Dict containing:
              - ``dominant_emotion`` (str): Primary detected emotion.
              - ``emotion_scores`` (dict): Per-emotion confidence scores.
              - ``sentiment`` (str): "positive" | "neutral" | "negative"
              - ``cognitive_distortions`` (list): Detected distortion patterns.
              - ``themes`` (list): Life themes present in message.
              - ``keywords`` (list): Top extracted keywords.
              - ``word_count`` (int): Message word count.
              - ``agent`` (str): Agent identifier.

        Example::

            {
                "dominant_emotion": "sadness",
                "sentiment": "negative",
                "cognitive_distortions": ["catastrophizing", "overgeneralization"],
                "themes": ["self worth", "isolation"],
                ...
            }
        """
        if not user_message or not user_message.strip():
            return self._default_result()

        text_lower = user_message.lower()
        words = text_lower.split()

        # ── Emotion detection ──────────────────────────────────────────────
        emotion_scores = self._detect_emotions(text_lower)
        dominant_emotion = max(emotion_scores, key=lambda k: emotion_scores[k])

        # If dominant score is very low, emotion is ambiguous
        if emotion_scores[dominant_emotion] < 0.05:
            dominant_emotion = "neutral"

        # ── Sentiment ─────────────────────────────────────────────────────
        sentiment = self._detect_sentiment(emotion_scores)

        # ── Cognitive distortions ──────────────────────────────────────────
        distortions = self._detect_distortions(text_lower)

        # ── Themes ────────────────────────────────────────────────────────
        themes = self._detect_themes(text_lower)

        # ── Keywords ──────────────────────────────────────────────────────
        keywords = extract_keywords(user_message, top_n=8)

        logger.info(
            f"[AnalyzerAgent] emotion={dominant_emotion} | sentiment={sentiment} | "
            f"distortions={distortions} | themes={themes}"
        )

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_scores,
            "sentiment": sentiment,
            "cognitive_distortions": distortions,
            "themes": themes,
            "keywords": keywords,
            "word_count": len(words),
            "agent": "AnalyzerAgent",
        }

    def _default_result(self) -> Dict[str, Any]:
        """Return a safe default analysis result."""
        return {
            "dominant_emotion": "neutral",
            "emotion_scores": {},
            "sentiment": "neutral",
            "cognitive_distortions": [],
            "themes": [],
            "keywords": [],
            "word_count": 0,
            "agent": "AnalyzerAgent",
        }
