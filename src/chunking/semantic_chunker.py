"""
Semantic Chunker - Groups text by semantic similarity using embeddings.
Creates semantically coherent chunks by detecting topic shifts.
"""

import logging
from typing import Optional
import numpy as np
from src.chunking.chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Splits text at semantic boundaries detected by embedding similarity drops.

    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Compute cosine similarity between consecutive sentences
    4. Detect "breakpoints" where similarity drops below threshold
    5. Merge sentences between breakpoints into chunks

    Advantage: chunks are semantically coherent (same topic).
    Cost: requires embedding model inference.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        breakpoint_percentile: float = 85.0,
        embedding_model: Optional[object] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.breakpoint_percentile = breakpoint_percentile
        self._embedding_model = embedding_model  # Injected dependency

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []

        meta = metadata or {}
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            # Can't do semantic splitting with one sentence
            return self._fallback_chunk(text, doc_id, meta)

        # Embed sentences
        embeddings = self._embed_sentences(sentences)

        if embeddings is None:
            logger.warning("Embedding failed, falling back to recursive chunking")
            return self._fallback_chunk(text, doc_id, meta)

        # Find breakpoints
        breakpoints = self._find_breakpoints(embeddings)

        # Build chunks from breakpoints
        sentence_groups = self._group_by_breakpoints(sentences, breakpoints)

        chunks = []
        char_offset = 0
        for i, group in enumerate(sentence_groups):
            content = " ".join(group).strip()
            if not content:
                continue
            # If group is still too large, split it further
            if len(content) > self.chunk_size * 2:
                sub_chunks = self._split_large_group(
                    content, doc_id, i, char_offset, meta
                )
                chunks.extend(sub_chunks)
                char_offset += len(content)
                continue

            start = text.find(content[:50], char_offset)
            if start == -1:
                start = char_offset
            end = start + len(content)

            chunk = self._build_chunk(
                content=content,
                doc_id=doc_id,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata={
                    **meta,
                    "chunk_strategy": "semantic",
                    "sentence_count": len(group),
                    "chunk_index": i,
                },
            )
            chunks.append(chunk)
            char_offset = max(0, end - self.chunk_overlap)

        filtered = self._filter_chunks(chunks)
        logger.debug(f"Semantic chunker: {len(filtered)} chunks from doc {doc_id[:8]}")
        return filtered

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        import re

        # Split on sentence-ending punctuation followed by whitespace and capital
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _embed_sentences(self, sentences: list[str]) -> Optional[np.ndarray]:
        """Generate embeddings for sentences."""
        if self._embedding_model is None:
            # Lazy load a lightweight model
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                return None
        try:
            embeddings = self._embedding_model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def _find_breakpoints(self, embeddings: np.ndarray) -> list[int]:
        """Find indices where topic shifts occur (low similarity)."""
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)

        if not similarities:
            return []

        threshold = np.percentile(similarities, 100 - self.breakpoint_percentile)
        breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        return breakpoints

    def _group_by_breakpoints(
        self, sentences: list[str], breakpoints: list[int]
    ) -> list[list[str]]:
        """Group sentences by breakpoints."""
        groups = []
        start = 0
        for bp in breakpoints:
            groups.append(sentences[start:bp])
            start = bp
        groups.append(sentences[start:])
        return [g for g in groups if g]

    def _fallback_chunk(self, text: str, doc_id: str, meta: dict) -> list[Chunk]:
        """Fall back to recursive chunking."""
        from src.chunking.recursive_chunker import RecursiveChunker

        rc = RecursiveChunker(self.chunk_size, self.chunk_overlap, self.min_chunk_size)
        return rc.chunk(text, doc_id, meta)

    def _split_large_group(
        self, text: str, doc_id: str, base_index: int, char_offset: int, meta: dict
    ) -> list[Chunk]:
        """Split oversized semantic groups using recursive chunker."""
        from src.chunking.recursive_chunker import RecursiveChunker

        rc = RecursiveChunker(self.chunk_size, self.chunk_overlap, self.min_chunk_size)
        sub_chunks = rc.chunk(
            text, doc_id, {**meta, "chunk_strategy": "semantic_recursive"}
        )
        # Adjust chunk indices
        for j, c in enumerate(sub_chunks):
            c.chunk_index = base_index * 100 + j
        return sub_chunks
