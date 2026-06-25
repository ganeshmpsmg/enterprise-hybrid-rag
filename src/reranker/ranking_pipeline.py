"""
Ranking Pipeline - Combines hybrid retrieval with cross-encoder reranking.
Full retrieval -> reranking pipeline with configurable stages.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.hybrid_retrieval.hybrid_retriever import HybridRetriever
from src.reranker.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """Final ranked result after all retrieval and reranking stages."""

    chunk_id: str
    doc_id: str
    content: str
    final_score: float
    rank: int
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "final_score": self.final_score,
            "rank": self.rank,
            "hybrid_score": self.hybrid_score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
        }


class RankingPipeline:
    """
    Full ranking pipeline: Hybrid Retrieval -> Cross-Encoder Reranking.

    Stage 1: Hybrid retrieval (dense + sparse + RRF) -> top-N candidates
    Stage 2: Cross-encoder reranking -> top-K final results

    This two-stage approach balances efficiency and accuracy:
    - Stage 1 (fast): reduces corpus to manageable candidate set
    - Stage 2 (accurate): precise relevance scoring on small candidate set
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        hybrid_top_k: int = 20,  # Candidates for reranker
        rerank_top_k: int = 5,  # Final results after reranking
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[RankedResult]:
        """
        Full retrieve + rerank pipeline.

        Args:
            query: User query
            top_k: Final number of results (overrides rerank_top_k)
            filter_metadata: Metadata filter

        Returns:
            Reranked list of RankedResult objects
        """
        final_k = top_k or self.rerank_top_k
        t0 = time.perf_counter()

        # Stage 1: Hybrid retrieval
        hybrid_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=self.hybrid_top_k,
            filter_metadata=filter_metadata,
        )

        if not hybrid_results:
            return []

        # Convert to dict format for reranker
        candidates = [r.to_dict() for r in hybrid_results]
        hybrid_score_map = {r["chunk_id"]: r["score"] for r in candidates}

        # Stage 2: Cross-encoder reranking
        reranked = self.reranker.rerank(
            query=query, candidates=candidates, top_k=final_k
        )

        # Build final results
        results = []
        for rank, item in enumerate(reranked):
            cid = item.get("chunk_id", "")
            results.append(
                RankedResult(
                    chunk_id=cid,
                    doc_id=item.get("doc_id", ""),
                    content=item.get("content", ""),
                    final_score=item.get("rerank_score", 0.0),
                    rank=rank,
                    hybrid_score=hybrid_score_map.get(cid, 0.0),
                    rerank_score=item.get("rerank_score", 0.0),
                    metadata=item.get("metadata", {}),
                )
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Ranking pipeline: {len(hybrid_results)} -> {len(results)} results "
            f"in {elapsed*1000:.1f}ms"
        )
        return results
