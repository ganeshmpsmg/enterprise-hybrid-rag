"""
Reciprocal Rank Fusion (RRF) - Combines multiple ranked lists into one.
RRF is the standard method for merging dense + sparse retrieval results.
"""

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    weights: Optional[list[float]] = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion algorithm.

    Formula:
        RRF_score(d) = sum_over_rankers[ weight_r / (k + rank_r(d)) ]

    Where:
        k = 60 (default, prevents very high scores for top-ranked docs)
        rank = 1-indexed position in each ranked list
        weight = optional per-ranker weight (default=1.0 for all)

    Args:
        ranked_lists: List of result lists, each sorted by score (best first).
                      Each result must have 'chunk_id' and optionally 'doc_id', 'content', 'metadata'.
        k: RRF constant (typically 60)
        weights: Optional weight per ranker. Default: equal weights.

    Returns:
        Fused and re-ranked list of results with rrf_score field.

    Example:
        dense_results = [{"chunk_id": "A", "score": 0.9}, {"chunk_id": "B", "score": 0.7}]
        sparse_results = [{"chunk_id": "B", "score": 25.0}, {"chunk_id": "C", "score": 20.0}]
        fused = rrf([dense_results, sparse_results], k=60)
        # Result: B gets score from rank 2 in dense + rank 1 in sparse
        #         A gets score from rank 1 in dense only
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists length ({len(ranked_lists)})"
        )

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Accumulate RRF scores
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_info: dict[str, dict] = {}  # chunk_id -> best result metadata

    for ranker_idx, (result_list, weight) in enumerate(zip(ranked_lists, weights)):
        for rank_0, result in enumerate(result_list):
            rank_1 = rank_0 + 1  # 1-indexed rank
            chunk_id = result.get("chunk_id", "")
            if not chunk_id:
                continue

            rrf_score = weight / (k + rank_1)
            rrf_scores[chunk_id] += rrf_score

            # Keep the result metadata (use first occurrence, or best score)
            if chunk_id not in doc_info:
                doc_info[chunk_id] = dict(result)
                doc_info[chunk_id]["source_ranks"] = {}
            doc_info[chunk_id]["source_ranks"][f"ranker_{ranker_idx}"] = rank_1

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for rank, chunk_id in enumerate(sorted_ids):
        result = dict(doc_info[chunk_id])
        result["score"] = round(rrf_scores[chunk_id], 6)
        result["rrf_score"] = round(rrf_scores[chunk_id], 6)
        result["rank"] = rank
        result["retrieval_type"] = "hybrid_rrf"
        fused_results.append(result)

    logger.debug(
        f"RRF fusion: {sum(len(r) for r in ranked_lists)} total results "
        f"-> {len(fused_results)} unique results"
    )
    return fused_results


def weighted_score_fusion(
    ranked_lists: list[list[dict]],
    weights: Optional[list[float]] = None,
    normalize_scores: bool = True,
) -> list[dict]:
    """
    Alternative: direct score fusion (normalize and weight scores).

    Less robust than RRF because:
    - Different rankers use different score scales
    - Requires careful normalization

    Use RRF unless you have a strong reason to use this.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    combined_scores: dict[str, float] = defaultdict(float)
    doc_info: dict[str, dict] = {}

    for result_list, weight in zip(ranked_lists, weights):
        if not result_list:
            continue

        # Normalize scores to [0, 1]
        if normalize_scores:
            max_score = max(r.get("score", 0) for r in result_list) or 1.0
            min_score = min(r.get("score", 0) for r in result_list)
            score_range = max_score - min_score or 1.0
        else:
            max_score, min_score, score_range = 1.0, 0.0, 1.0

        for result in result_list:
            chunk_id = result.get("chunk_id", "")
            if not chunk_id:
                continue
            raw_score = result.get("score", 0)
            norm_score = (
                (raw_score - min_score) / score_range if normalize_scores else raw_score
            )
            combined_scores[chunk_id] += weight * norm_score

            if chunk_id not in doc_info:
                doc_info[chunk_id] = dict(result)

    sorted_ids = sorted(
        combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True
    )
    results = []
    for rank, chunk_id in enumerate(sorted_ids):
        result = dict(doc_info[chunk_id])
        result["score"] = round(combined_scores[chunk_id], 6)
        result["rank"] = rank
        result["retrieval_type"] = "hybrid_weighted"
        results.append(result)

    return results
