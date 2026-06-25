"""
Semantic Search - High-level semantic search with query preprocessing and MMR.
Extends dense retrieval with Maximal Marginal Relevance for diversity.
"""

import logging
from typing import Optional
import numpy as np

from src.dense_retrieval.dense_retriever import DenseRetriever
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.vector_store import SearchResult

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic search with MMR (Maximal Marginal Relevance) for diverse results.

    MMR balances relevance and diversity:
        MMR = argmax[ lambda * sim(q, di) - (1-lambda) * max(sim(dj, di)) ]

    Lambda controls the relevance-diversity tradeoff:
    - lambda=1.0: Pure relevance (same as standard dense retrieval)
    - lambda=0.5: Balanced (default)
    - lambda=0.0: Maximum diversity
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        embedding_pipeline: EmbeddingPipeline,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
    ):
        self.dense_retriever = dense_retriever
        self.embedding_pipeline = embedding_pipeline
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
        fetch_k: int = 30,  # Fetch more candidates for MMR
    ) -> list[SearchResult]:
        """
        Semantic search with optional MMR diversity.

        Args:
            query: Search query
            top_k: Final number of results
            filter_metadata: Metadata filter
            fetch_k: Candidates to fetch before MMR (must be >= top_k)

        Returns:
            Diverse, relevant search results
        """
        if self.use_mmr:
            # Fetch more candidates for MMR reranking
            candidates = self.dense_retriever.retrieve(
                query=query,
                top_k=max(fetch_k, top_k * 3),
                filter_metadata=filter_metadata,
            )
            return self._apply_mmr(query, candidates, top_k)
        else:
            return self.dense_retriever.retrieve(query, top_k, filter_metadata)

    def _apply_mmr(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Apply MMR to select diverse and relevant results."""
        if len(candidates) <= top_k:
            return candidates

        # Embed query and candidate texts
        query_emb = self.embedding_pipeline.embed_query(query)
        candidate_texts = [c.content for c in candidates]
        candidate_embs = self.embedding_pipeline.embed_documents(candidate_texts)

        # Normalize
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        cand_norms = candidate_embs / (
            np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
        )

        # Query-candidate similarities
        query_sims = cand_norms @ query_norm

        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not selected_indices:
                # First: pick highest query similarity
                best = int(np.argmax([query_sims[i] for i in remaining_indices]))
                best_idx = remaining_indices[best]
            else:
                # MMR: balance query similarity and redundancy
                mmr_scores = []
                selected_embs = cand_norms[selected_indices]
                for i in remaining_indices:
                    relevance = self.mmr_lambda * query_sims[i]
                    redundancy = (1 - self.mmr_lambda) * np.max(
                        cand_norms[i] @ selected_embs.T
                    )
                    mmr_scores.append(relevance - redundancy)
                best = int(np.argmax(mmr_scores))
                best_idx = remaining_indices[best]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        results = []
        for rank, idx in enumerate(selected_indices):
            result = candidates[idx]
            result.rank = rank
            result.metadata["mmr_applied"] = True
            results.append(result)

        return results
