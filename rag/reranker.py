"""
MindSense AI - Cross-Encoder Reranker
=======================================
Reranks FAISS retrieval results using a cross-encoder model
(ms-marco-MiniLM-L-6-v2) for higher-precision relevance scoring.

The cross-encoder jointly encodes query + passage pairs, providing
much richer relevance signals than bi-encoder cosine similarity alone.

Falls back gracefully to the original fused score order if the
cross-encoder model cannot be loaded.

Usage::

    from rag.reranker import reranker
    reranked = reranker.rerank(query="I feel anxious", results=raw_results, top_k=5)
"""

from typing import Any, Dict, List, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    The cross-encoder model jointly scores query-passage pairs,
    providing higher-quality relevance estimates than the bi-encoder
    used for initial retrieval.

    Attributes:
        model_name (str): HuggingFace cross-encoder model identifier.
        device (str): Computation device ("cpu" or "cuda").
        top_k (int): Number of final results to return after reranking.
    """

    def __init__(self) -> None:
        self.model_name: str = settings.rag.RERANKER_MODEL
        self.device: str = settings.rag.RERANKER_DEVICE
        self.top_k: int = settings.rag.TOP_K_RERANK
        self._model: Optional[Any] = None
        self._available: bool = False

    def _load(self) -> None:
        """
        Load the cross-encoder model lazily.
        Sets ``_available = False`` if loading fails (graceful degradation).
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
            self._available = True
            logger.info(f"Cross-encoder reranker loaded: {self.model_name}")
        except Exception as e:
            logger.warning(
                f"Cross-encoder reranker unavailable ({e}). "
                f"Using fallback score ordering."
            )
            self._available = False

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results using the cross-encoder model.

        Args:
            query: Original user query string.
            results: Initial retrieval results from the FAISS retriever.
                     Each dict must have a ``"text"`` key.
            top_k: Number of results to return after reranking.
                   Defaults to settings.rag.TOP_K_RERANK.

        Returns:
            List of result dicts sorted by cross-encoder relevance score,
            with an added ``"rerank_score"`` field.
            Falls back to original ordering if reranker is unavailable.
        """
        self._load()

        k = top_k or self.top_k

        if not results:
            return []

        if not self._available or self._model is None:
            logger.debug("Using fallback ordering (no cross-encoder).")
            return self._fallback_rerank(results, k)

        try:
            # Build query-passage pairs for cross-encoder scoring
            pairs = [(query, result.get("text", "")[:512]) for result in results]

            # Score all pairs
            scores = self._model.predict(pairs)

            # Attach rerank scores
            for result, score in zip(results, scores):
                result["rerank_score"] = round(float(score), 4)

            # Sort by rerank score (descending)
            reranked = sorted(results, key=lambda r: r.get("rerank_score", 0.0), reverse=True)

            final = reranked[:k]
            logger.debug(
                f"Reranked {len(results)} → {len(final)} results | "
                f"top_score={final[0]['rerank_score'] if final else 'N/A'}"
            )
            return final

        except Exception as e:
            logger.error(f"Reranker error: {e}. Using fallback ordering.")
            return self._fallback_rerank(results, k)

    def _fallback_rerank(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback: sort by fused_score from the retriever and return top_k.

        Args:
            results: List of retrieval results.
            top_k: Number to return.

        Returns:
            Top-k results sorted by ``fused_score``.
        """
        sorted_results = sorted(
            results,
            key=lambda r: r.get("fused_score", r.get("dense_score", 0.0)),
            reverse=True,
        )
        for r in sorted_results:
            r["rerank_score"] = r.get("fused_score", r.get("dense_score", 0.0))
        return sorted_results[:top_k]

    @property
    def is_available(self) -> bool:
        """Whether the cross-encoder model is loaded and ready."""
        return self._available


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
reranker = CrossEncoderReranker()
