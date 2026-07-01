"""
Score Fusion Utilities - Additional fusion strategies and score normalization.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def softmax_normalize(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Apply softmax normalization to scores."""
    if not scores:
        return []
    arr = np.array(scores) / temperature
    arr = arr - arr.max()  # Numerical stability
    exp_arr = np.exp(arr)
    return (exp_arr / exp_arr.sum()).tolist()


def convex_combination_fusion(
    result_lists: list[list[dict]],
    weights: list[float],
) -> list[dict]:
    """
    Linear convex combination of normalized scores.
    Weights must sum to 1.0.
    """
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    assert len(result_lists) == len(weights)

    combined: dict[str, float] = defaultdict(float)
    info: dict[str, dict] = {}

    for result_list, w in zip(result_lists, weights):
        if not result_list:
            continue
        scores = [r.get("score", 0.0) for r in result_list]
        norm_scores = min_max_normalize(scores)
        for r, ns in zip(result_list, norm_scores):
            cid = r.get("chunk_id", "")
            combined[cid] += w * ns
            if cid not in info:
                info[cid] = dict(r)

    sorted_ids = sorted(combined, key=lambda x: combined[x], reverse=True)
    results = []
    for rank, cid in enumerate(sorted_ids):
        item = dict(info[cid])
        item["score"] = round(combined[cid], 6)
        item["rank"] = rank
        item["retrieval_type"] = "hybrid_convex"
        results.append(item)
    return results
