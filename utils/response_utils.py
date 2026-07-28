"""
MindSense AI - Response Utilities
====================================
Handles response formatting, Markdown processing, and
structuring of agent outputs into a consistent API response format.

All functions are pure and side-effect free.

Usage::

    from utils.response_utils import build_api_response, format_response_markdown
"""

import re
import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from utils.helpers import get_utc_timestamp

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Standard API Response Builder
# ---------------------------------------------------------------------------

def build_api_response(
    message: str,
    session_id: str,
    classification: Optional[Dict[str, Any]] = None,
    risk_level: Optional[str] = None,
    intent: Optional[str] = None,
    validation_score: Optional[float] = None,
    sources: Optional[List[str]] = None,
    error: Optional[str] = None,
    status: str = "success",
) -> Dict[str, Any]:
    """
    Build a standardized JSON-serializable API response dictionary.

    Args:
        message: The assistant's text response.
        session_id: Current session identifier.
        classification: Mental health classification result dict.
        risk_level: Risk assessment level ("low", "moderate", "high", "critical").
        intent: Detected user intent string.
        validation_score: Quality score from the validator (0.0 – 1.0).
        sources: List of knowledge source citations used in the response.
        error: Error message if something went wrong.
        status: Response status ("success" or "error").

    Returns:
        Dictionary suitable for Flask ``jsonify()`` or JSON serialization.
    """
    response: Dict[str, Any] = {
        "status": status,
        "timestamp": get_utc_timestamp(),
        "session_id": session_id,
        "response": {
            "message": message,
            "markdown": format_response_markdown(message),
        },
    }

    if classification:
        response["classification"] = classification

    if risk_level:
        response["risk_level"] = risk_level

    if intent:
        response["intent"] = intent

    if validation_score is not None:
        response["quality_score"] = round(validation_score, 3)

    if sources:
        response["sources"] = deduplicate_sources(sources)

    if error:
        response["error"] = error

    return response


def build_error_response(
    error_message: str,
    session_id: str = "",
    status_code: int = 500,
) -> Dict[str, Any]:
    """
    Build a standardized error response.

    Args:
        error_message: Human-readable error description.
        session_id: Current session identifier.
        status_code: HTTP status code for logging purposes.

    Returns:
        Error response dictionary.
    """
    logger.error(f"API error response [{status_code}]: {error_message}")
    return {
        "status": "error",
        "timestamp": get_utc_timestamp(),
        "session_id": session_id,
        "response": {
            "message": "I'm having a bit of trouble right now. Could you try again in a moment?",
            "markdown": "I'm having a bit of trouble right now. Could you try again in a moment?",
        },
        "error": error_message,
    }


# ---------------------------------------------------------------------------
# Markdown Formatting
# ---------------------------------------------------------------------------

def format_response_markdown(text: str) -> str:
    """
    Ensure the response text is clean, well-formatted Markdown.
    Handles common LLM output artifacts such as excessive asterisks,
    inconsistent spacing, and double blank lines.

    Args:
        text: Raw LLM-generated text.

    Returns:
        Cleaned Markdown string suitable for frontend rendering.
    """
    if not text:
        return ""

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix common LLM artifact: bold with no content (**  **)
    text = re.sub(r"\*\*\s*\*\*", "", text)

    # Ensure lists have proper spacing after bullet
    text = re.sub(r"^(\s*[-*+])([^\s])", r"\1 \2", text, flags=re.MULTILINE)

    return text.strip()


def strip_markdown(text: str) -> str:
    """
    Remove Markdown formatting and return plain text.
    Useful for TTS, evaluation scoring, and logging.

    Args:
        text: Markdown-formatted string.

    Returns:
        Plain text string without Markdown syntax.
    """
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)

    # Remove headers
    text = re.sub(r"#{1,6}\s+", "", text)

    # Remove bold and italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

    # Remove links — keep display text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove images
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)

    # Remove horizontal rules
    text = re.sub(r"[-*_]{3,}", "", text)

    # Normalize whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Source Citation Utilities
# ---------------------------------------------------------------------------

def deduplicate_sources(sources: List[str]) -> List[str]:
    """
    Remove duplicate source citations while preserving order.

    Args:
        sources: List of source strings (file names, URLs, etc.).

    Returns:
        Deduplicated list of source strings.
    """
    seen = set()
    unique = []
    for src in sources:
        normalized = src.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def format_sources_for_prompt(sources: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved source chunks into a numbered list
    suitable for inclusion in an LLM prompt.

    Args:
        sources: List of dicts with keys: ``text``, ``source``, ``category``, ``score``.

    Returns:
        Formatted string block for prompt injection.
    """
    if not sources:
        return "No specific knowledge context retrieved."

    lines = ["[RETRIEVED KNOWLEDGE CONTEXT]"]
    for i, src in enumerate(sources, 1):
        category = src.get("category", "General")
        source_name = src.get("source", "Unknown")
        text = src.get("text", "").strip()
        lines.append(f"\n[Source {i}] [{category}] — {source_name}")
        lines.append(text[:400] + ("..." if len(text) > 400 else ""))

    return "\n".join(lines)


def format_conversation_for_prompt(
    history: List[Dict[str, str]], max_turns: int = 6
) -> str:
    """
    Format conversation history into a dialogue block for prompt injection.

    Args:
        history: List of ``{"role": "user"|"assistant", "content": "..."}`` dicts.
        max_turns: Maximum number of recent turns to include (default 6).

    Returns:
        Formatted conversation string.
    """
    if not history:
        return "No previous conversation."

    recent = history[-max_turns:]
    lines = ["[CONVERSATION HISTORY]"]
    for turn in recent:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stream Processing
# ---------------------------------------------------------------------------

def stream_text_generator(text: str, chunk_size: int = 10, delay: float = 0.03):
    """
    Generator that yields text in small chunks to simulate streaming output.
    Used for Server-Sent Events (SSE) when real streaming isn't available.

    Args:
        text: Full response text to stream.
        chunk_size: Number of characters per chunk.
        delay: Seconds to sleep between chunks (simulates latency).

    Yields:
        String chunks of the response.
    """
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        yield chunk
        if delay > 0:
            time.sleep(delay)
