"""
MindSense AI - Text Preprocessing
===================================
Provides text cleaning, normalization, and tokenization utilities
used throughout the RAG pipeline and classifier.

Functions are pure (no side effects) and stateless for maximum
reusability and testability.

Usage::

    from utils.preprocessing import clean_text, normalize_whitespace
    cleaned = clean_text(raw_input)
"""

import re
import unicodedata
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Common English contractions to expand before processing
CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "won't": "will not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "that's": "that is", "there's": "there is",
    "what's": "what is", "who's": "who is", "let's": "let us",
}

# Regex patterns (compiled once for performance)
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
_RE_PHONE = re.compile(r"\b(\+?\d[\d\s\-().]{7,}\d)\b")
_RE_MULTIPLE_SPACES = re.compile(r"\s+")
_RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_RE_SPECIAL_CHARS = re.compile(r"[^\w\s.,!?;:'\"-]")
_RE_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to their closest ASCII equivalents.

    Args:
        text: Input string with possible Unicode characters.

    Returns:
        Unicode-normalized string using NFKD decomposition.
    """
    return unicodedata.normalize("NFKD", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters into a single space
    and strip leading/trailing whitespace.

    Args:
        text: Input string with irregular spacing.

    Returns:
        Whitespace-normalized string.
    """
    text = _RE_MULTIPLE_NEWLINES.sub("\n\n", text)
    text = _RE_MULTIPLE_SPACES.sub(" ", text)
    return text.strip()


def expand_contractions(text: str) -> str:
    """
    Expand English contractions (e.g., "I'm" → "I am").

    Args:
        text: Input text potentially containing contractions.

    Returns:
        Text with contractions expanded.
    """
    words = text.split()
    expanded = [CONTRACTIONS.get(word.lower(), word) for word in words]
    return " ".join(expanded)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return _RE_URL.sub("[URL]", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return _RE_EMAIL.sub("[EMAIL]", text)


def remove_phone_numbers(text: str) -> str:
    """Remove phone numbers from text."""
    return _RE_PHONE.sub("[PHONE]", text)


def clean_text(
    text: str,
    remove_url: bool = True,
    remove_email: bool = True,
    remove_phone: bool = True,
    expand_contractions_flag: bool = True,
    lowercase: bool = False,
    normalize_unicode_flag: bool = True,
) -> str:
    """
    Master text cleaning pipeline. Applies multiple cleaning steps
    in the correct sequence.

    Args:
        text: Raw input text to clean.
        remove_url: Whether to remove/mask URLs (default True).
        remove_email: Whether to remove/mask email addresses (default True).
        remove_phone: Whether to remove/mask phone numbers (default True).
        expand_contractions_flag: Whether to expand contractions (default True).
        lowercase: Whether to convert text to lowercase (default False).
        normalize_unicode_flag: Whether to normalize unicode characters (default True).

    Returns:
        Cleaned, normalized string ready for NLP processing.

    Example::

        >>> clean_text("I'm feeling really bad today... visit https://example.com")
        "I am feeling really bad today... visit [URL]"
    """
    if not text or not isinstance(text, str):
        logger.warning("clean_text received non-string or empty input")
        return ""

    if normalize_unicode_flag:
        text = normalize_unicode(text)

    if remove_url:
        text = remove_urls(text)

    if remove_email:
        text = remove_emails(text)

    if remove_phone:
        text = remove_phone_numbers(text)

    if expand_contractions_flag:
        text = expand_contractions(text)

    if lowercase:
        text = text.lower()

    text = normalize_whitespace(text)
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences using regex heuristics.

    Args:
        text: Input paragraph or document text.

    Returns:
        List of individual sentences (stripped, non-empty).
    """
    sentences = _RE_SENTENCE_END.split(text)
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_chars: int = 500, ellipsis: str = "...") -> str:
    """
    Truncate text to a maximum number of characters, preserving word boundaries.

    Args:
        text: Input text to truncate.
        max_chars: Maximum character count (default 500).
        ellipsis: String to append at truncation point.

    Returns:
        Truncated text ending at a word boundary.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + ellipsis


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract simple keyword tokens from text (stopword-filtered, lowercased).
    Used for BM25-style keyword scoring in the RAG retriever.

    Args:
        text: Input text for keyword extraction.
        top_n: Maximum number of keywords to return.

    Returns:
        List of keyword strings sorted by frequency (most frequent first).
    """
    # Minimal English stopwords (avoids NLTK dependency)
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
        "it", "its", "they", "them", "what", "which", "who", "this", "that",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "a", "an", "the", "and", "but", "or",
        "for", "nor", "on", "at", "to", "by", "up", "is", "in", "of",
        "so", "as", "if", "not", "no", "just", "very", "also", "more",
    }

    # Tokenize: split on non-alphanumeric characters
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    # Filter stopwords
    keywords = [t for t in tokens if t not in STOPWORDS]

    # Count frequencies
    freq: dict = {}
    for kw in keywords:
        freq[kw] = freq.get(kw, 0) + 1

    # Sort by frequency descending
    sorted_kw = sorted(freq, key=lambda k: freq[k], reverse=True)
    return sorted_kw[:top_n]


def chunk_text_by_sentences(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.
    Simpler alternative to LangChain's RecursiveCharacterTextSplitter for
    cases where langchain is unavailable.

    Args:
        text: Full document text to chunk.
        chunk_size: Target character length per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunk strings.
    """
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Carry overlap into next chunk
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:] + " " + sentence
            current_chunk = current_chunk.strip()

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
