"""
Index Builder - Orchestrates the full indexing pipeline from chunks to vector store.
Coordinates embedding generation and vector store insertion.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.chunking.chunker import Chunk
from src.embeddings.embedding_pipeline import EmbeddingPipeline
from src.vectorstore.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of an indexing operation."""

    doc_id: str
    chunks_indexed: int
    embeddings_generated: int
    time_sec: float
    success: bool
    error: Optional[str] = None


class IndexBuilder:
    """
    Builds vector store indexes from chunks.

    Pipeline:
    chunks -> embed -> add to vector store

    Features:
    - Handles large document batches efficiently
    - Reports indexing progress and statistics
    - Supports re-indexing (upsert semantics)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
    ):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

    def index_chunks(self, chunks: list[Chunk]) -> IndexingResult:
        """
        Index a list of chunks into the vector store.

        Args:
            chunks: Chunks to index (must all have same doc_id for single document)

        Returns:
            IndexingResult with statistics
        """
        if not chunks:
            return IndexingResult("", 0, 0, 0.0, True)

        doc_id = chunks[0].doc_id
        t0 = time.perf_counter()

        try:
            # 1. Generate embeddings
            embed_result = self.embedding_pipeline.embed_chunks(chunks)
            if not embed_result.embedded_chunks:
                return IndexingResult(doc_id, 0, 0, 0.0, False, "Embedding failed")

            # 2. Prepare data for vector store
            chunk_ids = [ec.chunk_id for ec in embed_result.embedded_chunks]
            contents = [ec.content for ec in embed_result.embedded_chunks]
            metadatas = [
                {**ec.metadata, "doc_id": ec.doc_id}
                for ec in embed_result.embedded_chunks
            ]
            import numpy as np

            embeddings = np.array([ec.embedding for ec in embed_result.embedded_chunks])

            # 3. Add to vector store
            added = self.vector_store.add_embeddings(
                chunk_ids, embeddings, contents, metadatas
            )

            elapsed = time.perf_counter() - t0
            logger.info(f"Indexed doc {doc_id[:8]}: {added} chunks in {elapsed:.2f}s")
            return IndexingResult(
                doc_id=doc_id,
                chunks_indexed=added,
                embeddings_generated=embed_result.successful,
                time_sec=round(elapsed, 3),
                success=True,
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(f"Indexing failed for doc {doc_id[:8]}: {e}")
            return IndexingResult(doc_id, 0, 0, elapsed, False, str(e))

    def index_multiple_documents(
        self, doc_chunks: dict[str, list[Chunk]]
    ) -> list[IndexingResult]:
        """Index chunks from multiple documents."""
        results = []
        for doc_id, chunks in doc_chunks.items():
            result = self.index_chunks(chunks)
            results.append(result)
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Indexed {success_count}/{len(results)} documents successfully")
        return results
