"""
ChromaDB Manager - Persistent vector store with metadata filtering.
Best for: persistent storage, metadata filtering, easy setup.
"""

import logging
from typing import Optional
import numpy as np

from src.vectorstore.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class ChromaManager(VectorStore):
    """
    ChromaDB vector store backend.

    Advantages over FAISS:
    - Native metadata filtering (filter by doc_id, file_type, topics, etc.)
    - Persistent storage out-of-the-box
    - Supports deletion by document ID
    - HTTP client/server mode for distributed setups

    Usage:
        manager = ChromaManager(collection_name="ml_docs", persist_dir="./chroma_db")
        manager.add_embeddings(chunk_ids, embeddings, contents, metadatas)
        results = manager.search(query_emb, top_k=10, filter_metadata={"file_type": "pdf"})
    """

    def __init__(
        self,
        collection_name: str = "ml_documents",
        persist_dir: Optional[str] = "./chroma_db",
        host: Optional[str] = None,
        port: int = 8001,
        distance_function: str = "cosine",
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.host = host
        self.port = port
        self.distance_function = distance_function
        self._client = None
        self._collection = None

    @property
    def client(self):
        """Lazy initialize ChromaDB client."""
        if self._client is None:
            self._init_client()
        return self._client

    @property
    def collection(self):
        """Lazy get/create collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_function},
            )
        return self._collection

    def _init_client(self):
        """Initialize ChromaDB client (local or HTTP)."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        if self.host:
            # HTTP client mode (for Docker/K8s deployments)
            logger.info(f"Connecting to ChromaDB server at {self.host}:{self.port}")
            self._client = chromadb.HttpClient(host=self.host, port=self.port)
        else:
            # Local persistent client
            logger.info(f"Creating local ChromaDB at {self.persist_dir}")
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        logger.info(f"ChromaDB client initialized. Collection: {self.collection_name}")

    def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        metadatas: list[dict],
    ) -> int:
        """Add embeddings to ChromaDB collection."""
        if not chunk_ids:
            return 0

        # ChromaDB requires serializable metadata values (no lists)
        sanitized_metas = [self._sanitize_metadata(meta) for meta in metadatas]
        embeddings_list = embeddings.tolist()
        # Batch add to avoid memory issues
        batch_size = 100
        added = 0
        for i in range(0, len(chunk_ids), batch_size):
            end = i + batch_size
            try:
                self.collection.upsert(
                    ids=chunk_ids[i:end],
                    embeddings=embeddings_list[i:end],
                    documents=contents[i:end],
                    metadatas=sanitized_metas[i:end],
                )
                added += len(chunk_ids[i:end])
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"ChromaDB upsert error (batch {i}-{end}): {repr(e)}")

        logger.info(
            f"ChromaDB: upserted {added} vectors. Total: {self.collection.count()}"
        )
        return added

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search ChromaDB with optional metadata filtering."""
        if self.collection.count() == 0:
            return []

        # Build ChromaDB where clause from filter_metadata
        where = None
        if filter_metadata:
            where = self._build_where_clause(filter_metadata)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return []

        search_results = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for rank, (chunk_id, distance, doc, meta) in enumerate(
            zip(ids, distances, documents, metadatas)
        ):
            # Convert distance to similarity score
            # ChromaDB cosine distance is (1 - cosine_similarity)
            score = 1.0 - distance if self.distance_function == "cosine" else -distance

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", ""),
                    content=doc,
                    score=round(score, 4),
                    metadata=meta,
                    rank=rank,
                )
            )

        return search_results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a document by doc_id metadata field."""
        try:
            # Get all chunk IDs with this doc_id
            results = self.collection.get(where={"doc_id": doc_id})
            ids_to_delete = results["ids"]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(
                    f"ChromaDB: deleted {len(ids_to_delete)} chunks for doc {doc_id[:8]}"
                )
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"ChromaDB delete error: {e}")
            return 0

    def get_stats(self) -> dict:
        return {
            "backend": "chromadb",
            "collection": self.collection_name,
            "total_vectors": self.collection.count(),
            "distance_function": self.distance_function,
            "persist_dir": self.persist_dir,
        }

    def save(self, path: str):
        """ChromaDB persists automatically when using PersistentClient."""
        logger.info("ChromaDB: automatic persistence (no manual save needed)")

    def load(self, path: str):
        """ChromaDB loads automatically from persist_dir."""
        logger.info(f"ChromaDB: loading from {path}")
        self.persist_dir = path
        self._client = None  # Force re-init with new path
        self._collection = None

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Convert non-serializable metadata values for ChromaDB."""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                sanitized[k] = ", ".join(str(x) for x in v)
            elif isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif v is None:
                sanitized[k] = ""
            else:
                sanitized[k] = str(v)
        return sanitized

    def _build_where_clause(self, filters: dict) -> dict:
        """Build ChromaDB where clause from simple filter dict."""
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: {"$eq": str(value) if not isinstance(value, bool) else value}}
        # Multiple conditions: use $and
        conditions = []
        for key, value in filters.items():
            conditions.append(
                {key: {"$eq": str(value) if not isinstance(value, bool) else value}}
            )
        return {"$and": conditions}
