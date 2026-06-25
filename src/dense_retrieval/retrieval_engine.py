"""
Retrieval Engine - Unified dense retrieval with caching and configuration.
"""
import logging
from typing import Optional

from src.dense_retrieval.dense_retriever import DenseRetriever
from src.vectorstore.vector_store import SearchResult

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    High-level retrieval engine wrapping dense retrieval.
    Adds query validation, result post-processing, and statistics.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        min_score_threshold: float = 0.0,
        use_mmr: bool = False,
    ):
        self.dense_retriever = dense_retriever
        self.min_score_threshold = min_score_threshold
        self.use_mmr = use_mmr
        self._query_log: list[dict] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Retrieve with score thresholding and logging."""
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        results = self.dense_retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Apply minimum score threshold
        if self.min_score_threshold > 0:
            results = [r for r in results if r.score >= self.min_score_threshold]

        # Log query
        self._query_log.append({
            "query": query,
            "results_count": len(results),
            "top_score": results[0].score if results else 0.0,
        })

        return results

    def get_query_history(self) -> list[dict]:
        return self._query_log[-100:]  # Last 100 queries
