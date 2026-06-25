"""
Embedding Utilities - Helper functions for embedding operations.
Cosine similarity, normalization, dimensionality reduction, etc.
"""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(
    query_embs: np.ndarray, doc_embs: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query embeddings and document embeddings.

    Args:
        query_embs: Shape (n_queries, dim)
        doc_embs:   Shape (n_docs, dim)

    Returns:
        Similarity matrix of shape (n_queries, n_docs)
    """
    # Normalize
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    return q_norm @ d_norm.T


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize an array of embedding vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k scores (highest first)."""
    k = min(k, len(scores))
    return np.argsort(scores)[::-1][:k]


def batch_cosine_similarity(
    query_emb: np.ndarray, corpus_embs: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    """
    Compute cosine similarity for large corpora using batching to avoid OOM.
    """
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    similarities = np.zeros(len(corpus_embs), dtype=np.float32)
    for i in range(0, len(corpus_embs), batch_size):
        batch = corpus_embs[i:i + batch_size]
        batch_norm = batch / (np.linalg.norm(batch, axis=1, keepdims=True) + 1e-10)
        similarities[i:i + len(batch)] = batch_norm @ query_norm
    return similarities


def deduplicate_embeddings(
    embeddings: np.ndarray,
    threshold: float = 0.98,
) -> tuple[np.ndarray, list[int]]:
    """
    Remove near-duplicate embeddings based on cosine similarity threshold.

    Returns:
        (unique_embeddings, kept_indices)
    """
    kept_indices = [0]
    for i in range(1, len(embeddings)):
        sims = cosine_similarity_matrix(
            embeddings[i:i+1], embeddings[np.array(kept_indices)]
        )[0]
        if np.max(sims) < threshold:
            kept_indices.append(i)
    return embeddings[np.array(kept_indices)], kept_indices


def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean pooling over token embeddings with attention mask.
    Used when working with raw HuggingFace transformer outputs.
    """
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def embedding_stats(embeddings: np.ndarray) -> dict:
    """Compute descriptive statistics for a set of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "count": len(embeddings),
        "dimension": embeddings.shape[1] if embeddings.ndim == 2 else 0,
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "dtype": str(embeddings.dtype),
    }
