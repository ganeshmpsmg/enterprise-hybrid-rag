"""
Cross-Encoder Reranker - ms-marco-MiniLM-L-6-v2 for precise relevance scoring.

Architecture difference:
  Bi-encoder (embedding): query and doc encoded SEPARATELY -> cosine similarity
  Cross-encoder: query and doc encoded TOGETHER -> relevance score

Cross-encoder is slower but far more accurate because it uses full attention
between query and document tokens (vs independent encoding in bi-encoders).
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Cross-encoder based reranker using sentence-transformers.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking dataset
    - Input: [CLS] query [SEP] passage [SEP]
    - Output: relevance score (logit)
    - 6-layer MiniLM: fast inference, strong performance

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device or self._auto_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._score_count = 0
        self._total_time = 0.0

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        score_field: str = "rerank_score",
    ) -> list[dict]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: The original query string
            candidates: List of result dicts, each must have 'content' key
            top_k: Number of top results to return after reranking
            score_field: Key name for the reranking score in output

        Returns:
            Top-k candidates sorted by cross-encoder score (highest first)
        """
        if not candidates:
            return []

        t0 = time.perf_counter()

        # Build (query, passage) pairs
        pairs = [(query, c.get("content", "")) for c in candidates]

        # Score all pairs
        scores = self._score_pairs(pairs)

        # Sort by reranker score
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        elapsed = time.perf_counter() - t0
        self._score_count += len(candidates)
        self._total_time += elapsed

        # Build result list
        results = []
        for new_rank, (score, candidate) in enumerate(ranked[:top_k]):
            result = dict(candidate)
            result[score_field] = round(float(score), 4)
            result["original_rank"] = candidate.get("rank", -1)
            result["rank"] = new_rank
            result["retrieval_type"] = (
                f"{candidate.get('retrieval_type', 'unknown')}_reranked"
            )
            results.append(result)

        logger.info(
            f"Reranked {len(candidates)} -> {len(results)} results in {elapsed*1000:.1f}ms | "
            f"top_score={results[0][score_field] if results else 0:.4f}"
        )
        return results

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, passage) pairs directly."""
        return self._score_pairs(pairs)

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "total_scored": self._score_count,
            "avg_ms_per_pair": round(
                (self._total_time / max(self._score_count, 1)) * 1000, 2
            ),
        }

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run cross-encoder inference in batches."""
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            try:
                scores = self.model.predict(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_scores.extend(scores.tolist())
            except Exception as e:
                logger.error(f"Cross-encoder scoring failed for batch {i}: {e}")
                all_scores.extend([0.0] * len(batch))
        return all_scores

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            logger.info(f"Cross-encoder loaded: {self.model_name}")
        except ImportError:
            raise ImportError("sentence-transformers not installed.")

    def _auto_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
