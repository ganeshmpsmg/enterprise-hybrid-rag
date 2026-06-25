"""
Retrieval Metrics - Precision@K, Recall@K, MRR, nDCG for RAG evaluation.
"""
import logging
import math
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@K: fraction of top-K retrieved that are relevant.
    P@K = |relevant ∩ retrieved[:K]| / K
    """
    if k == 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@K: fraction of relevant docs found in top-K.
    R@K = |relevant ∩ retrieved[:K]| / |relevant|
    """
    if not relevant or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """
    Average Precision: area under precision-recall curve.
    AP = (1/R) * sum_k[ P@k * rel(k) ]
    """
    if not relevant or not retrieved:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant)


def mean_average_precision(
    all_retrieved: list[list[str]], all_relevant: list[set[str]]
) -> float:
    """MAP: mean of AP across all queries."""
    if not all_retrieved:
        return 0.0
    aps = [average_precision(r, rel) for r, rel in zip(all_retrieved, all_relevant)]
    return float(np.mean(aps))


def mean_reciprocal_rank(
    all_retrieved: list[list[str]], all_relevant: list[set[str]]
) -> float:
    """
    MRR: mean of reciprocal rank of first relevant result.
    MRR = (1/|Q|) * sum_q[ 1/rank_q ]
    """
    if not all_retrieved:
        return 0.0
    rrs = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs))


def dcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
    use_graded: bool = False,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    DCG@K: Discounted Cumulative Gain.
    DCG@K = sum_i[ rel_i / log2(i+1) ]  for i in 1..K
    """
    top_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, 1):
        if use_graded and relevance_scores:
            rel = relevance_scores.get(doc_id, 0.0)
        else:
            rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    nDCG@K: Normalized DCG. Divides DCG by ideal DCG.
    nDCG@K = DCG@K / IDCG@K ∈ [0, 1]
    """
    dcg = dcg_at_k(retrieved, relevant, k)
    # Ideal DCG: all relevant docs retrieved first
    ideal_retrieved = list(relevant)[:k]
    idcg = dcg_at_k(ideal_retrieved, relevant, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    queries: list[str],
    retrieved_lists: list[list[str]],
    relevant_sets: list[set[str]],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Evaluate retrieval system across multiple metrics and K values.

    Args:
        queries: Query strings
        retrieved_lists: Retrieved doc IDs per query
        relevant_sets: Relevant doc IDs per query
        k_values: K values to evaluate at

    Returns:
        Dict of metric_name -> score
    """
    results = {}
    n = len(queries)

    for k in k_values:
        prec = np.mean([precision_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)])
        rec = np.mean([recall_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)])
        ndcg = np.mean([ndcg_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)])
        results[f"precision@{k}"] = round(float(prec), 4)
        results[f"recall@{k}"] = round(float(rec), 4)
        results[f"ndcg@{k}"] = round(float(ndcg), 4)

    results["mrr"] = round(mean_reciprocal_rank(retrieved_lists, relevant_sets), 4)
    results["map"] = round(mean_average_precision(retrieved_lists, relevant_sets), 4)
    results["num_queries"] = n

    logger.info(f"Retrieval evaluation: MRR={results['mrr']}, MAP={results['map']}, "
                f"nDCG@10={results.get('ndcg@10', 'N/A')}")
    return results
