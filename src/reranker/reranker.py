"""
Reranker - High-level reranking interface with score normalization.
"""
import logging
from typing import Optional
from src.reranker.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)


class Reranker:
    """
    Production reranker with configurable backend.
    Currently supports: CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        normalize_scores: bool = True,
    ):
        self.top_k = top_k
        self.normalize_scores = normalize_scores
        self._cross_encoder = CrossEncoderReranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Rerank candidates and optionally normalize scores to [0,1]."""
        k = top_k or self.top_k
        results = self._cross_encoder.rerank(query, candidates, top_k=k)

        if self.normalize_scores and results:
            scores = [r.get("rerank_score", 0.0) for r in results]
            min_s, max_s = min(scores), max(scores)
            rng = max_s - min_s
            for r in results:
                raw = r.get("rerank_score", 0.0)
                r["rerank_score_normalized"] = round(
                    (raw - min_s) / rng if rng > 0 else 1.0, 4
                )

        return results

    def get_stats(self) -> dict:
        return self._cross_encoder.get_stats()
