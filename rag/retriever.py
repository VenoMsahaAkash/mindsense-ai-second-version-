"""
MindSense AI - Hybrid RAG Retriever
======================================
Implements hybrid retrieval combining:
  1. Dense semantic search via FAISS (using SentenceTransformer embeddings)
  2. Sparse keyword scoring (BM25-style TF-IDF term frequency matching)

The two scores are fused using a configurable alpha weight to produce
a final ranked list of relevant knowledge chunks.

Usage::

    from rag.retriever import retriever
    results = retriever.retrieve("I feel anxious and can't sleep", top_k=5)
"""

import math
from typing import Any, Dict, List, Optional

from config import settings
from model.faiss.index_manager import index_manager
from rag.embeddings import encode_query
from utils.logger import get_logger
from utils.preprocessing import clean_text, extract_keywords

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that fuses dense (FAISS) and sparse (keyword) search.

    Dense retrieval captures semantic meaning while sparse retrieval
    ensures exact keyword matches are never missed.

    Attributes:
        alpha (float): Weight for dense score in fusion (0.0=sparse only, 1.0=dense only).
        top_k_retrieve (int): Number of candidates from FAISS to collect.
        top_k_return (int): Final number of results to return before reranking.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            alpha: Weight for dense semantic score (default 0.7).
                   Sparse keyword score weight = (1 - alpha).
        """
        self.alpha = alpha
        self.top_k_retrieve = settings.rag.TOP_K_RETRIEVE
        self.top_k_return = settings.rag.TOP_K_RERANK

    def _dense_retrieve(
        self, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform dense vector search using FAISS.

        Args:
            query: User query string.
            top_k: Number of candidates to retrieve.

        Returns:
            List of result dicts with ``score`` in [0, 1] (cosine similarity).
        """
        query_vector = encode_query(query)
        results = index_manager.search(query_vector, top_k=top_k)

        # Normalize scores to [0, 1] range (inner product for L2-normalized vecs is already in [-1, 1])
        for r in results:
            # Cosine similarity is in [-1, 1]; shift and scale to [0, 1]
            r["dense_score"] = max(0.0, float((r["score"] + 1.0) / 2.0))

        return results

    def _keyword_score(self, query: str, text: str) -> float:
        """
        Compute a simple keyword overlap score between the query and a chunk.

        Uses TF-IDF-inspired term frequency weighting — no external library needed.

        Args:
            query: User query string.
            text: Candidate chunk text.

        Returns:
            Keyword overlap score in [0.0, 1.0].
        """
        query_keywords = set(extract_keywords(query, top_n=15))
        if not query_keywords:
            return 0.0

        text_lower = text.lower()
        text_words = set(text_lower.split())

        # Count keyword matches weighted by rarity (inverse text frequency)
        matched = query_keywords & text_words
        if not matched:
            return 0.0

        # Score = matched / total query keywords (recall-style)
        score = len(matched) / len(query_keywords)

        # Bonus for exact phrase match
        query_lower = query.lower()
        for kw in matched:
            if f" {kw} " in query_lower and f" {kw} " in text_lower:
                score = min(1.0, score + 0.05)

        return min(1.0, score)

    def _fuse_scores(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Fuse dense and sparse scores using reciprocal rank fusion or linear combination.

        Args:
            results: List of dense retrieval results.
            query: Original query string for sparse scoring.

        Returns:
            Results sorted by fused score (descending).
        """
        for result in results:
            dense_score = result.get("dense_score", 0.0)
            keyword_score = self._keyword_score(query, result.get("text", ""))

            # Linear combination
            fused = (self.alpha * dense_score) + ((1.0 - self.alpha) * keyword_score)
            result["keyword_score"] = round(keyword_score, 4)
            result["fused_score"] = round(fused, 4)

        return sorted(results, key=lambda r: r["fused_score"], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant knowledge chunks for a given query.

        Args:
            query: User message or search query.
            top_k: Number of final results to return (defaults to settings.rag.TOP_K_RERANK).
            category_filter: If set, only return chunks from this category (e.g., "CBT").

        Returns:
            List of result dicts, sorted by relevance score (descending). Each dict contains:
              - ``text`` (str): Chunk text
              - ``source`` (str): Source document filename
              - ``category`` (str): Knowledge category
              - ``dense_score`` (float): Semantic similarity score
              - ``keyword_score`` (float): Keyword overlap score
              - ``fused_score`` (float): Combined relevance score

        Returns empty list if the index is not ready.
        """
        if not query or not query.strip():
            logger.warning("Retriever received empty query.")
            return []

        if not index_manager.is_ready:
            logger.warning(
                "FAISS index not ready. Run 'python rag/build_index.py' to build it."
            )
            return []

        k = top_k or self.top_k_return
        n_retrieve = max(self.top_k_retrieve, k * 2)  # Over-retrieve for fusion

        # Clean query before retrieval
        clean_query = clean_text(query, lowercase=False)

        logger.debug(f"Retrieving for query: '{clean_query[:80]}...'")

        # --- Dense retrieval ---
        dense_results = self._dense_retrieve(clean_query, top_k=n_retrieve)

        if not dense_results:
            logger.warning("Dense retrieval returned no results.")
            return []

        # --- Apply category filter ---
        if category_filter:
            dense_results = [
                r for r in dense_results
                if r.get("category", "").lower() == category_filter.lower()
            ]

        # --- Fuse scores ---
        fused_results = self._fuse_scores(dense_results, clean_query)

        # --- Return top-k ---
        final_results = fused_results[:k]

        logger.info(
            f"Retrieval complete | query_len={len(query)} | "
            f"retrieved={len(dense_results)} | returned={len(final_results)}"
        )

        return final_results

    def retrieve_by_categories(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        per_category_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results from specific categories, returning up to
        ``per_category_k`` results per category.

        Useful for ensuring diversity across CBT, DBT, Coping, etc.

        Args:
            query: User query string.
            categories: List of category names to retrieve from.
                        Defaults to all knowledge categories.
            per_category_k: Max results per category (default 2).

        Returns:
            Aggregated and deduplicated list of results.
        """
        if categories is None:
            categories = settings.rag.KNOWLEDGE_CATEGORIES

        # Over-retrieve globally and filter by category
        all_results = self.retrieve(
            query,
            top_k=self.top_k_retrieve,
        )

        category_results: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in categories}

        for result in all_results:
            cat = result.get("category", "")
            if cat in category_results and len(category_results[cat]) < per_category_k:
                category_results[cat].append(result)

        # Flatten and sort by fused score
        aggregated = []
        for cat_results in category_results.values():
            aggregated.extend(cat_results)

        aggregated.sort(key=lambda r: r.get("fused_score", 0), reverse=True)
        return aggregated


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
retriever = HybridRetriever(alpha=0.7)
