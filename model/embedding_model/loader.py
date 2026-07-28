"""
MindSense AI - Embedding Model Loader
=======================================
Loads and caches the SentenceTransformer embedding model.
Provides a clean interface for encoding text to dense vectors.

The model is downloaded automatically on first use and cached
in model/embedding_model/ for offline use.

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


class EmbeddingModelLoader:
    """
    Singleton wrapper around SentenceTransformer for generating text embeddings.

    The model is loaded lazily on first call to ``encode()`` and reused
    for all subsequent requests. The model is cached to disk in
    ``model/embedding_model/`` to avoid repeated downloads.

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

    def _load(self) -> None:
        """
        Load the SentenceTransformer model from local cache or download it.
        Idempotent — loads only once.
        """
        if self._model is not None:
            return

        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers package not installed yet. Embeddings disabled.")
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
            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(1)

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
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model load failed: {e}") from e

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

        Example::

            vectors = loader.encode(["I feel anxious", "CBT helps with anxiety"])
            # vectors.shape == (2, 384)
        """
        self._load()

        if isinstance(texts, str):
            texts = [texts]

        if not texts or self._model is None:
            return np.zeros((len(texts) if texts else 0, self.embedding_dim), dtype=np.float32)

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
            logger.error(f"Encoding error: {e}")
            raise RuntimeError(f"Failed to encode texts: {e}") from e

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
