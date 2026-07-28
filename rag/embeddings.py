"""
MindSense AI - RAG Embeddings Wrapper
=======================================
Thin convenience wrapper around EmbeddingModelLoader
that is imported by the RAG pipeline modules.

Provides encode_texts() and encode_query() helpers
to keep the RAG code clean and readable.

Usage::

    from rag.embeddings import encode_texts, encode_query
    vecs = encode_texts(["CBT helps with anxiety", "Mindfulness reduces stress"])
    q_vec = encode_query("How do I manage anxiety?")
"""

from typing import List
import numpy as np

from model.embedding_model.loader import embedding_loader
from utils.logger import get_logger

logger = get_logger(__name__)


def encode_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """
    Encode a list of text strings into embedding vectors.

    Args:
        texts: List of strings to encode.
        show_progress: Show tqdm progress bar for large batches.

    Returns:
        Float32 NumPy array of shape ``(n_texts, embedding_dim)``.
    """
    if not texts:
        logger.warning("encode_texts received empty list.")
        return np.array([])

    logger.debug(f"Encoding {len(texts)} texts for RAG indexing...")
    return embedding_loader.encode(texts, show_progress=show_progress)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query string into a 1D embedding vector.

    Args:
        query: Query text (user message or search string).

    Returns:
        1D Float32 NumPy array of shape ``(embedding_dim,)``.
    """
    if not query or not query.strip():
        logger.warning("encode_query received empty query string.")
        embedding_loader.load() if hasattr(embedding_loader, "load") else None
        import numpy as np
        from config import settings
        return np.zeros(settings.embedding.EMBEDDING_DIM, dtype=np.float32)

    logger.debug(f"Encoding query: '{query[:60]}...'")
    return embedding_loader.encode_single(query)


def warmup_embedding_model() -> None:
    """Warm up the embedding model at application startup."""
    embedding_loader.warmup()
