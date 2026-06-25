"""
Embedding Pipeline - Orchestrates the full embedding workflow for the RAG system.
Processes chunks, generates embeddings, handles errors, and tracks progress.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.chunking.chunker import Chunk
from src.embeddings.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector."""

    chunk: Chunk
    embedding: np.ndarray
    embed_time_ms: float = 0.0

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def metadata(self) -> dict:
        return self.chunk.metadata


@dataclass
class EmbeddingResult:
    """Result of embedding a batch of chunks."""

    embedded_chunks: list[EmbeddedChunk]
    total_chunks: int
    successful: int
    failed: int
    total_time_sec: float
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return round(self.successful / max(self.total_chunks, 1), 3)


class EmbeddingPipeline:
    """
    End-to-end embedding pipeline for the RAG system.

    Responsibilities:
    1. Take chunks from the chunking stage
    2. Prepare text for embedding (prefix, truncation)
    3. Run batch embedding with progress tracking
    4. Handle failures gracefully
    5. Return EmbeddedChunk objects ready for vector store

    Supports two embedding modes:
    - 'passage': for document chunks (indexed)
    - 'query': for search queries (retrieved)
    Some models (e.g., E5) use different prefixes for each.
    """

    # Prefix templates for asymmetric embedding models
    PASSAGE_PREFIX = "passage: "  # For E5 models
    QUERY_PREFIX = "query: "  # For E5 models

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        use_passage_prefix: bool = False,  # Set True for E5/BGE models
        max_text_length: int = 2000,
    ):
        self.embedding_model = model or EmbeddingModel(
            model_name=model_name, batch_size=batch_size
        )
        self.batch_size = batch_size
        self.use_passage_prefix = use_passage_prefix
        self.max_text_length = max_text_length

    def embed_chunks(self, chunks: list[Chunk]) -> EmbeddingResult:
        """
        Embed a list of chunks.

        Args:
            chunks: Chunks from the chunking pipeline

        Returns:
            EmbeddingResult with embedded chunks and statistics
        """
        if not chunks:
            return EmbeddingResult([], 0, 0, 0, 0.0)

        logger.info(f"Embedding {len(chunks)} chunks...")
        t0 = time.perf_counter()

        embedded = []
        errors = []

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
                texts = [self._prepare_text(c.content, mode="passage") for c in batch]
                t_batch = time.perf_counter()
                embeddings = self.embedding_model.embed_batch(texts)
                batch_ms = (time.perf_counter() - t_batch) * 1000

                for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                    embedded.append(
                        EmbeddedChunk(
                            chunk=chunk,
                            embedding=emb,
                            embed_time_ms=batch_ms / len(batch),
                        )
                    )

                logger.debug(
                    f"Batch {i//self.batch_size + 1}: "
                    f"{len(batch)} chunks in {batch_ms:.1f}ms"
                )
            except Exception as e:
                error_msg = f"Batch {i}-{i+len(batch)} failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        total_time = time.perf_counter() - t0
        result = EmbeddingResult(
            embedded_chunks=embedded,
            total_chunks=len(chunks),
            successful=len(embedded),
            failed=len(chunks) - len(embedded),
            total_time_sec=round(total_time, 3),
            errors=errors,
        )

        logger.info(
            f"Embedding complete: {result.successful}/{result.total_chunks} chunks "
            f"in {total_time:.2f}s ({result.success_rate*100:.1f}% success rate)"
        )
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query.

        Args:
            query: Search query string

        Returns:
            Query embedding vector
        """
        text = self._prepare_text(query, mode="query")
        return self.embedding_model.embed(text)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed multiple queries."""
        texts = [self._prepare_text(q, mode="query") for q in queries]
        return self.embedding_model.embed_batch(texts)

    def _prepare_text(self, text: str, mode: str = "passage") -> str:
        """
        Prepare text for embedding.

        - Optionally adds passage/query prefix for asymmetric models
        - Truncates very long texts to avoid token overflow
        """
        # Truncate
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]

        # Add prefix for asymmetric models (E5, BGE)
        if self.use_passage_prefix:
            if mode == "passage":
                text = self.PASSAGE_PREFIX + text
            elif mode == "query":
                text = self.QUERY_PREFIX + text

        return text.strip()
