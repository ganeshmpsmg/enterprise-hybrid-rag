"""
Base Chunker - Abstract base class defining the chunking interface.
All chunking strategies implement this interface for pipeline consistency.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    A single text chunk with full provenance metadata.
    This is the atomic unit passed through the retrieval pipeline.
    """

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int  # Position in document
    start_char: int  # Character offset start in original doc
    end_char: int  # Character offset end in original doc
    metadata: dict = field(default_factory=dict)  # Inherited + chunk-specific metadata

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def is_empty(self) -> bool:
        return len(self.content.strip()) == 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "word_count": self.word_count,
            "char_count": self.char_count,
            **self.metadata,
        }


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate deterministic chunk ID from doc_id and position."""
    return f"{doc_id[:8]}_{chunk_index:04d}"


class BaseChunker(ABC):
    """
    Abstract base chunker. All chunking strategies extend this class.

    Subclasses implement:
    - chunk(text, doc_id, metadata) -> list[Chunk]

    Common features:
    - Chunk size validation
    - Empty chunk filtering
    - Metadata propagation
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """Split text into chunks."""
        ...

    def _filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Remove empty and undersized chunks."""
        return [c for c in chunks if c.char_count >= self.min_chunk_size]

    def _build_chunk(
        self,
        content: str,
        doc_id: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: dict,
    ) -> Chunk:
        """Factory method for creating chunks with consistent IDs."""
        return Chunk(
            chunk_id=make_chunk_id(doc_id, chunk_index),
            doc_id=doc_id,
            content=content.strip(),
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=dict(metadata),
        )

    def get_stats(self, chunks: list[Chunk]) -> dict:
        """Compute statistics for a list of chunks."""
        if not chunks:
            return {"total": 0}
        lengths = [c.char_count for c in chunks]
        return {
            "total": len(chunks),
            "avg_chars": round(sum(lengths) / len(lengths), 1),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "total_chars": sum(lengths),
        }
