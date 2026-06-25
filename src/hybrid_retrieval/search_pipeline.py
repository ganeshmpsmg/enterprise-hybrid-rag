"""
Search Pipeline - End-to-end search from query to ranked results.
Orchestrates query expansion, hybrid retrieval, and result formatting.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.hybrid_retrieval.hybrid_retriever import HybridRetriever, HybridResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    """Final search response returned to the API."""

    query: str
    results: list[HybridResult]
    total_results: int
    latency_ms: float
    retrieval_metadata: dict = field(default_factory=dict)


class SearchPipeline:
    """
    End-to-end search pipeline:
    1. Query preprocessing
    2. Hybrid retrieval (dense + sparse + RRF)
    3. Result deduplication
    4. Response formatting
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        top_k: int = 10,
        deduplicate: bool = True,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.top_k = top_k
        self.deduplicate = deduplicate

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> SearchResponse:
        """Execute search and return SearchResponse."""
        t0 = time.perf_counter()
        k = top_k or self.top_k

        results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=k,
            filter_metadata=filter_metadata,
        )

        if self.deduplicate:
            results = self._deduplicate(results)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            latency_ms=round(elapsed_ms, 2),
            retrieval_metadata={"top_k": k, "filter": filter_metadata},
        )

    def _deduplicate(self, results: list[HybridResult]) -> list[HybridResult]:
        """Remove near-duplicate chunks (same content)."""
        seen_content = set()
        unique = []
        for r in results:
            # Use first 100 chars as dedup key
            key = r.content[:100].strip().lower()
            if key not in seen_content:
                seen_content.add(key)
                unique.append(r)
        return unique
