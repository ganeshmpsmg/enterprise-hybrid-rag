"""
Vector Store Interface - Abstract base for all vector database backends.
Provides a unified API regardless of backend (FAISS, ChromaDB, Qdrant).
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


class VectorStore(ABC):
    """
    Abstract interface for vector database backends.

    All backends must implement:
    - add_embeddings: Index embeddings with metadata
    - search: Find nearest neighbors by vector similarity
    - delete: Remove documents by ID
    - get_stats: Return index statistics
    """

    @abstractmethod
    def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        metadatas: list[dict],
    ) -> int:
        """Add embeddings to the index. Returns number of items added."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbors."""
        ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns deleted count."""
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Return index statistics."""
        ...

    @abstractmethod
    def save(self, path: str):
        """Persist index to disk."""
        ...

    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        ...
