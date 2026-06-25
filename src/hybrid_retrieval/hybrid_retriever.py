"""
Hybrid Retriever - Combines dense (semantic) and sparse (BM25) retrieval with RRF.
This is the core retrieval component of the Enterprise RAG system.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.dense_retrieval.dense_retriever import DenseRetriever
from src.hybrid_retrieval.rrf import reciprocal_rank_fusion, weighted_score_fusion
from src.sparse_retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Unified result from hybrid retrieval."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    rank: int
    metadata: dict = field(default_factory=dict)
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    source_ranks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "retrieval_type": "hybrid",
            "metadata": self.metadata,
        }


class HybridRetriever:
    """
    Hybrid Retriever: Dense + Sparse + RRF fusion.

    Architecture:
        Query
          |-- Dense Retriever (semantic) --> top-N dense results
          |-- Sparse Retriever (BM25)    --> top-N sparse results
          |
          └-> RRF Fusion --> top-K fused results

    Why hybrid?
    - Dense: handles semantic similarity, handles vocabulary mismatch
    - Sparse: handles exact keyword matching, rare terms, acronyms
    - Together: covers both semantic and lexical relevance

    BEIR benchmark shows hybrid retrieval consistently outperforms
    either dense-only or sparse-only by 3-8% nDCG@10.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
        fusion_method: str = "rrf",  # "rrf" | "weighted"
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._stats = {"queries": 0, "total_ms": 0.0}

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
        return_source_results: bool = False,
    ) -> list[HybridResult]:
        """
        Execute hybrid retrieval for a query.

        Args:
            query: Natural language query
            top_k: Final number of results to return
            filter_metadata: Optional metadata filter applied to both retrievers
            return_source_results: If True, include per-ranker results in metadata

        Returns:
            List of HybridResult sorted by fused score
        """
        t0 = time.perf_counter()

        # ── 1. Dense retrieval ─────────────────────────
        dense_results_raw = self.dense_retriever.retrieve(
            query=query,
            top_k=self.dense_top_k,
            filter_metadata=filter_metadata,
        )
        dense_results = [r.to_dict() for r in dense_results_raw]

        # ── 2. Sparse (BM25) retrieval ─────────────────
        sparse_results = self.sparse_retriever.search(
            query=query,
            top_k=self.sparse_top_k,
            filter_metadata=filter_metadata,
        )

        # ── 3. Fuse results ────────────────────────────
        if not dense_results and not sparse_results:
            return []

        ranked_lists = []
        weights = []

        if dense_results:
            ranked_lists.append(dense_results)
            weights.append(self.dense_weight)
        if sparse_results:
            ranked_lists.append(sparse_results)
            weights.append(self.sparse_weight)

        if self.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=weights)
        else:
            fused = weighted_score_fusion(ranked_lists, weights=weights)

        # ── 4. Build HybridResult objects ──────────────
        # Build lookup maps for per-source ranks
        dense_rank_map = {r.get("chunk_id"): r.get("rank") for r in dense_results}
        sparse_rank_map = {r.get("chunk_id"): r.get("rank") for r in sparse_results}

        results = []
        for rank, item in enumerate(fused[:top_k]):
            cid = item.get("chunk_id", "")
            result = HybridResult(
                chunk_id=cid,
                doc_id=item.get("doc_id", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                rank=rank,
                metadata=item.get("metadata", {}),
                dense_rank=dense_rank_map.get(cid),
                sparse_rank=sparse_rank_map.get(cid),
                source_ranks=item.get("source_ranks", {}),
            )
            results.append(result)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._stats["queries"] += 1
        self._stats["total_ms"] += elapsed_ms

        logger.info(
            f"Hybrid retrieval: {len(results)} results | "
            f"dense={len(dense_results)}, sparse={len(sparse_results)} | "
            f"{elapsed_ms:.1f}ms"
        )

        return results

    def get_stats(self) -> dict:
        return {
            "queries": self._stats["queries"],
            "avg_latency_ms": round(
                self._stats["total_ms"] / max(self._stats["queries"], 1), 2
            ),
            "dense_top_k": self.dense_top_k,
            "sparse_top_k": self.sparse_top_k,
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
        }
