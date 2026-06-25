"""
Dense Retriever - Semantic similarity search using embedding vectors.
Finds documents semantically similar to the query even without exact term matches.
"""
import logging
import time
from typing import Optional
import numpy as np

from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense vector-based retrieval (semantic search).

    Unlike BM25/TF-IDF which match on keywords, dense retrieval:
    - Embeds both query and documents into a shared vector space
    - Finds documents closest to query in that space (cosine/dot similarity)
    - Handles synonyms, paraphrases, and semantic variations naturally

    Example:
        Query: "How does attention mechanism work in transformers?"
        Retrieves: passages about "self-attention", "multi-head attention", etc.
        even if those exact words aren't in the query.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
    ):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        self._query_count = 0
        self._total_query_time = 0.0

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Retrieve documents semantically similar to the query.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"file_type": "pdf"})

        Returns:
            List of SearchResult objects sorted by similarity score
        """
        t0 = time.perf_counter()

        # Embed query
        query_embedding = self.embedding_pipeline.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Add retrieval type to each result
        for rank, result in enumerate(results):
            result.rank = rank
            result.metadata["retrieval_type"] = "dense"

        elapsed = time.perf_counter() - t0
        self._query_count += 1
        self._total_query_time += elapsed

        logger.debug(
            f"Dense retrieval: {len(results)} results for "
            f"'{query[:50]}...' in {elapsed*1000:.1f}ms"
        )
        return results

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[list[SearchResult]]:
        """Retrieve results for multiple queries efficiently."""
        all_results = []
        # Embed all queries at once for efficiency
        query_embeddings = self.embedding_pipeline.embed_queries(queries)
        for query, qemb in zip(queries, query_embeddings):
            results = self.vector_store.search(
                query_embedding=qemb,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
            for rank, r in enumerate(results):
                r.rank = rank
                r.metadata["retrieval_type"] = "dense"
            all_results.append(results)
        return all_results

    def get_stats(self) -> dict:
        return {
            "query_count": self._query_count,
            "avg_query_time_ms": round(
                (self._total_query_time / max(self._query_count, 1)) * 1000, 2
            ),
            "vector_store_stats": self.vector_store.get_stats(),
        }
