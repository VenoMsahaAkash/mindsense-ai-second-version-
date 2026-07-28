"""
MindSense AI - FAISS Index Manager
=====================================
Manages the FAISS vector index: loading, saving, searching,
and incremental updates.

Supports both flat (exact) and IVF (approximate) FAISS indices.
Metadata is stored in a companion JSON file for chunk-to-source mapping.

Usage::

    from model.faiss.index_manager import index_manager
    results = index_manager.search(query_vector, top_k=10)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from config import settings
from utils.logger import get_logger
from utils.helpers import safe_json_load, safe_json_save

logger = get_logger(__name__)


class FAISSIndexManager:
    """
    Manages a FAISS index for dense vector similarity search.

    The index stores dense embeddings of knowledge base chunks.
    A companion metadata JSON file maps each vector index position
    to its source document, category, and text content.

    Attributes:
        index_dir (Path): Directory containing the index and metadata files.
        index_file (str): Name of the FAISS index file.
        metadata_file (str): Name of the metadata JSON file.
        embedding_dim (int): Dimensionality of stored vectors.
    """

    def __init__(self) -> None:
        self.index_dir: Path = settings.faiss.INDEX_DIR
        self.index_file: str = settings.faiss.INDEX_FILE
        self.metadata_file: str = settings.faiss.METADATA_FILE
        self.embedding_dim: int = settings.embedding.EMBEDDING_DIM
        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []
        self._loaded: bool = False

    @property
    def index_path(self) -> Path:
        """Full path to the FAISS index file."""
        return self.index_dir / self.index_file

    @property
    def metadata_path(self) -> Path:
        """Full path to the metadata JSON file."""
        return self.index_dir / self.metadata_file

    def _create_index(self) -> faiss.Index:
        """
        Create a new empty FAISS index.

        Returns:
            Flat L2 index (exact search) suitable for corpora up to ~100k vectors.
        """
        index_type = settings.faiss.INDEX_TYPE

        if index_type == "IVF":
            # Approximate search — faster but requires training
            nlist = settings.faiss.NLIST
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            logger.info(f"Created IVF FAISS index | dim={self.embedding_dim} | nlist={nlist}")
        else:
            # Exact search — always correct, no training required
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for normalized vectors
            logger.info(f"Created Flat (exact) FAISS index | dim={self.embedding_dim}")

        return index

    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if no index exists.
        """
        if self._loaded:
            return True

        if not self.index_path.exists():
            logger.warning(f"FAISS index not found at {self.index_path}")
            logger.warning("Run 'python rag/build_index.py' to build the knowledge index.")
            return False

        try:
            self._index = faiss.read_index(str(self.index_path))
            self._metadata = safe_json_load(self.metadata_path, default=[])
            self._loaded = True
            n_vectors = self._index.ntotal
            logger.info(
                f"FAISS index loaded | vectors={n_vectors} | "
                f"metadata_entries={len(self._metadata)}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def save(self) -> bool:
        """
        Persist the FAISS index and metadata to disk.

        Returns:
            True on success, False on failure.
        """
        if self._index is None:
            logger.error("Cannot save — no index in memory.")
            return False

        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            safe_json_save(self._metadata, self.metadata_path)
            logger.info(
                f"FAISS index saved | path={self.index_path} | "
                f"vectors={self._index.ntotal}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_entries: List[Dict[str, Any]],
    ) -> bool:
        """
        Add embedding vectors and their metadata to the index.

        Args:
            vectors: Float32 NumPy array of shape ``(n, embedding_dim)``.
            metadata_entries: List of dicts describing each vector.
                              Must have same length as ``vectors``.

        Returns:
            True on success, False on failure.

        Raises:
            ValueError: If vectors and metadata_entries have mismatched lengths.
        """
        if len(vectors) != len(metadata_entries):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata ({len(metadata_entries)}) "
                f"must have the same length."
            )

        if self._index is None:
            self._index = self._create_index()

            # Train IVF index if required
            if isinstance(self._index, faiss.IndexIVFFlat) and not self._index.is_trained:
                if len(vectors) < settings.faiss.NLIST:
                    logger.warning(
                        f"Not enough vectors to train IVF index "
                        f"(need {settings.faiss.NLIST}, got {len(vectors)}). "
                        f"Using Flat index instead."
                    )
                    self._index = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    logger.info(f"Training IVF index on {len(vectors)} vectors...")
                    self._index.train(vectors)

        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)  # Normalize for cosine similarity via IP

        self._index.add(vectors)
        self._metadata.extend(metadata_entries)

        logger.info(
            f"Added {len(vectors)} vectors to index | "
            f"total={self._index.ntotal}"
        )
        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search against the FAISS index.

        Args:
            query_vector: Query embedding of shape ``(embedding_dim,)`` or ``(1, embedding_dim)``.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of result dicts, each containing:
              - ``text`` (str): Source chunk text.
              - ``source`` (str): Source document filename.
              - ``category`` (str): Knowledge category (CBT, DBT, etc.).
              - ``score`` (float): Cosine similarity score (0.0 – 1.0).
              - ``chunk_id`` (int): Position in the metadata list.

        Returns empty list if index is not loaded.
        """
        if not self._loaded:
            loaded = self.load()
            if not loaded:
                logger.warning("Returning empty results — FAISS index not available.")
                return []

        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty.")
            return []

        # Reshape and normalize query vector
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Limit top_k to index size
        actual_k = min(top_k, self._index.ntotal)

        # Set nprobe for IVF indices
        if isinstance(self._index, faiss.IndexIVFFlat):
            self._index.nprobe = settings.faiss.NPROBE

        distances, indices = self._index.search(query_vector, actual_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]
            result = {
                "text": meta.get("text", ""),
                "source": meta.get("source", "Unknown"),
                "category": meta.get("category", "General"),
                "score": float(dist),  # Inner product = cosine similarity for normalized vectors
                "chunk_id": int(idx),
            }
            results.append(result)

        logger.debug(f"FAISS search returned {len(results)} results for top_k={top_k}")
        return results

    def reset(self) -> None:
        """Clear the index and metadata. Useful for rebuilding."""
        self._index = None
        self._metadata = []
        self._loaded = False
        logger.info("FAISS index reset.")

    @property
    def size(self) -> int:
        """Number of vectors currently stored in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def is_ready(self) -> bool:
        """Whether the index is loaded and ready for search."""
        return self._loaded and self._index is not None and self._index.ntotal > 0


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
index_manager = FAISSIndexManager()
