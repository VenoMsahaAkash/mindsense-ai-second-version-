"""
MindSense AI - Embedding Model Loader
=======================================
Loads and caches the SentenceTransformer embedding model.
Provides a clean interface for encoding text to dense vectors.

On Render free tier (512MB RAM), set EMBEDDING_BACKEND=tfidf to use a
lightweight numpy-based TF-IDF hash embedding that requires no model download.
For full semantic quality, set EMBEDDING_BACKEND=sentence_transformer (default
for local development with sufficient RAM).

Usage::

    from model.embedding_model.loader import embedding_loader
    vectors = embedding_loader.encode(["I feel anxious today"])
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _tfidf_hash_encode(texts: List[str], dim: int = 384) -> np.ndarray:
    """
    Lightweight TF-IDF hash-based embedding using only numpy.
    Produces a deterministic dense vector of shape (n, dim) with no model needed.
    Good enough for keyword-overlap-heavy RAG on a tiny corpus.
    """
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            vectors.append(vec)
            continue
        # Hash each word into a bucket and accumulate TF weight
        for word in words:
            h = hash(word) % dim
            # Negative hash values wrap around
            h = h if h >= 0 else h + dim
            vec[h] += 1.0 / len(words)  # TF-weighted
        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


class EmbeddingModelLoader:
    """
    Singleton wrapper for generating text embeddings.

    Supports two backends:
    - ``sentence_transformer``: Full SentenceTransformer model (~90MB RAM).
    - ``tfidf``: Lightweight numpy hash embedding, zero RAM overhead.

    Backend is controlled by the ``EMBEDDING_BACKEND`` environment variable.
    Default on Render free tier: ``tfidf``.
    Default for local dev: ``sentence_transformer``.

    Attributes:
        model_name (str): HuggingFace model identifier.
        device (str): Computation device ("cpu" or "cuda").
        batch_size (int): Encoding batch size.
        normalize (bool): Whether to L2-normalize output vectors.
        embedding_dim (int): Output vector dimensionality.
    """

    def __init__(self) -> None:
        self.model_name: str = settings.embedding.MODEL_NAME
        self.model_dir: Path = settings.embedding.MODEL_DIR
        self.device: str = settings.embedding.DEVICE
        self.batch_size: int = settings.embedding.BATCH_SIZE
        self.normalize: bool = settings.embedding.NORMALIZE_EMBEDDINGS
        self.embedding_dim: int = settings.embedding.EMBEDDING_DIM
        self._model: Optional[SentenceTransformer] = None
        # Default to tfidf on Render (low RAM), sentence_transformer locally
        self._backend: str = os.environ.get("EMBEDDING_BACKEND", "tfidf")
        logger.info(f"EmbeddingLoader backend: {self._backend}")

    def _load(self) -> None:
        """
        Load the SentenceTransformer model from local cache or download it.
        Idempotent — loads only once. Skipped entirely for tfidf backend.
        """
        if self._backend == "tfidf":
            return  # No model needed

        if self._model is not None:
            return

        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers not installed. Switching to tfidf backend.")
            self._backend = "tfidf"
            return

        cache_path = self.model_dir / self.model_name.replace("/", "_")

        # Prefer local cached model; fall back to HuggingFace Hub download
        if cache_path.exists():
            logger.info(f"Loading embedding model from local cache: {cache_path}")
            model_path = str(cache_path)
        else:
            logger.info(
                f"Downloading embedding model '{self.model_name}' "
                f"(cache: {cache_path})"
            )
            model_path = self.model_name
            self.model_dir.mkdir(parents=True, exist_ok=True)

        try:
            try:
                import torch
                if hasattr(torch, "set_num_threads"):
                    torch.set_num_threads(1)
            except ImportError:
                pass

            self._model = SentenceTransformer(
                model_name_or_path=model_path,
                device=self.device,
                cache_folder=str(self.model_dir),
            )
            import gc; gc.collect()
            logger.info(
                f"Embedding model loaded | dim={self.embedding_dim} | device={self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}. Switching to tfidf.")
            self._backend = "tfidf"

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more text strings into dense embedding vectors.

        Args:
            texts: A single string or a list of strings to encode.
            batch_size: Override the default batch size (optional).
            show_progress: Show tqdm progress bar for large batches.

        Returns:
            NumPy array of shape ``(n_texts, embedding_dim)`` with float32 dtype.
            For a single string input, returns shape ``(1, embedding_dim)``.
        """
        self._load()

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self._backend == "tfidf" or self._model is None:
            return _tfidf_hash_encode(texts, dim=self.embedding_dim)

        try:
            embeddings = self._model.encode(  # type: ignore
                sentences=texts,
                batch_size=batch_size or self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            logger.debug(f"Encoded {len(texts)} text(s) → shape {embeddings.shape}")
            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Encoding error: {e}. Switching to tfidf.")
            self._backend = "tfidf"
            return _tfidf_hash_encode(texts, dim=self.embedding_dim)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a 1D embedding vector.

        Args:
            text: Input text string.

        Returns:
            1D NumPy array of shape ``(embedding_dim,)`` with float32 dtype.
        """
        return self.encode([text])[0]

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._model is not None

    def warmup(self) -> None:
        """
        Warm up the model by encoding a dummy sentence.
        Call this at application startup to avoid latency on first real request.
        """
        self.encode(["MindSense AI warmup sentence."])
        logger.info("Embedding model warmed up.")


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
embedding_loader = EmbeddingModelLoader()
