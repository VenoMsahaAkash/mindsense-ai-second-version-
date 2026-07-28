"""
MindSense AI - General Helper Utilities
=========================================
Shared utility functions used across multiple modules.
Includes timing decorators, safe JSON operations, ID generation,
token counting, and miscellaneous helpers.

Usage::

    from utils.helpers import generate_session_id, safe_json_load, timer
"""

import uuid
import time
import json
import hashlib
import functools
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from utils.logger import get_logger

logger = get_logger(__name__)

# Generic TypeVar for decorator typing
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# ID & Hashing Utilities
# ---------------------------------------------------------------------------

def generate_session_id() -> str:
    """
    Generate a unique session identifier using UUID4.

    Returns:
        Hex string session ID (32 characters, no hyphens).

    Example::

        >>> sid = generate_session_id()
        >>> len(sid)
        32
    """
    return uuid.uuid4().hex


def generate_message_id() -> str:
    """
    Generate a unique message identifier with a timestamp prefix
    for natural chronological ordering.

    Returns:
        String in format: ``msg_<timestamp_ms>_<uuid4_hex[:8]>``
    """
    ts_ms = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return f"msg_{ts_ms}_{uid}"


def hash_text(text: str) -> str:
    """
    Compute a deterministic SHA-256 hash of a text string.
    Useful for caching and deduplication.

    Args:
        text: Input string to hash.

    Returns:
        64-character hex digest string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Timing & Performance
# ---------------------------------------------------------------------------

def timer(func: F) -> F:
    """
    Decorator that logs the execution time of the wrapped function.

    Args:
        func: Any callable to wrap.

    Returns:
        Wrapped callable with timing.

    Example::

        @timer
        def my_slow_function():
            time.sleep(2)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"[TIMER] {func.__qualname__} completed in {elapsed:.3f}s")
        return result

    return wrapper  # type: ignore


def get_utc_timestamp() -> str:
    """
    Get the current UTC timestamp as an ISO 8601 string.

    Returns:
        ISO 8601 UTC timestamp string, e.g., ``"2025-01-15T10:30:00Z"``
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Safe JSON Operations
# ---------------------------------------------------------------------------

def safe_json_load(path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load a JSON file, returning a default value on any error.

    Args:
        path: File path to the JSON file.
        default: Value to return if loading fails (default ``None``).

    Returns:
        Parsed JSON data or the default value.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug(f"JSON file not found: {path}, returning default.")
        return default
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {path}: {e}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error loading {path}: {e}")
        return default


def safe_json_save(data: Any, path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely serialize data to a JSON file.
    Creates parent directories if they don't exist.

    Args:
        data: Serializable Python object.
        path: Target file path.
        indent: JSON indentation level (default 2).

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Text & Token Utilities
# ---------------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text string using a simple
    word-split heuristic (approximately 1 token per 4 characters).
    This avoids requiring a full tokenizer for quick estimates.

    Args:
        text: Input text string.

    Returns:
        Estimated token count (integer).
    """
    return max(1, len(text) // 4)


def truncate_to_token_limit(text: str, max_tokens: int = 3000) -> str:
    """
    Truncate text to fit within an approximate token limit.

    Args:
        text: Input text to truncate.
        max_tokens: Maximum estimated token count.

    Returns:
        Truncated text string.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_sentence = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
    if last_sentence > 0:
        return truncated[: last_sentence + 1]
    return truncated


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a numeric value within [min_val, max_val].

    Args:
        value: The value to clamp.
        min_val: Lower bound (inclusive).
        max_val: Upper bound (inclusive).

    Returns:
        Clamped float value.
    """
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Data Structure Utilities
# ---------------------------------------------------------------------------

def flatten_list(nested: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested for item in sublist]


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of a given size.

    Args:
        lst: Input list.
        chunk_size: Maximum size of each chunk.

    Returns:
        List of list chunks.
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deduplicate_list(items: List[Any]) -> List[Any]:
    """
    Remove duplicates from a list while preserving insertion order.

    Args:
        items: Input list (elements must be hashable).

    Returns:
        Deduplicated list.
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries into one. Later dicts take precedence
    for overlapping keys (shallow merge).

    Args:
        *dicts: Variable number of dictionaries to merge.

    Returns:
        Merged dictionary.
    """
    result: Dict[str, Any] = {}
    for d in dicts:
        result.update(d)
    return result


# ---------------------------------------------------------------------------
# Validation Utilities
# ---------------------------------------------------------------------------

def is_empty_or_whitespace(text: Optional[str]) -> bool:
    """
    Check if a string is None, empty, or contains only whitespace.

    Args:
        text: String to check.

    Returns:
        True if the string is effectively empty.
    """
    return text is None or text.strip() == ""


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely traverse a nested dictionary using a sequence of keys.

    Args:
        d: The dictionary to traverse.
        *keys: Sequence of keys to traverse.
        default: Value to return if any key is missing.

    Returns:
        The nested value or default.

    Example::

        data = {"a": {"b": {"c": 42}}}
        safe_get(data, "a", "b", "c")  # → 42
        safe_get(data, "a", "x", "c", default=0)  # → 0
    """
    try:
        result = d
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        return default
