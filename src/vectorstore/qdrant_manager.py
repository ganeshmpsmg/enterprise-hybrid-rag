"""
Qdrant Manager - Production-grade vector database with advanced filtering.
Best for: production deployments, complex filters, scalability, deletions.
"""
import logging
import uuid
from typing import Optional
import numpy as np

from src.vectorstore.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class QdrantManager(VectorStore):
    """
    Qdrant vector store backend.

    Advantages:
    - Rich payload filtering (nested conditions, range queries, geo)
    - Efficient on-disk indexing (handles billions of vectors)
    - Native support for multiple vectors per point
    - Atomic upserts and deletes
    - REST + gRPC APIs
    - Kubernetes-native deployment

    Architecture:
    - Points: {id, vector, payload}
    - Collections: index of points with shared vector config
    - Payload: arbitrary JSON metadata (like ChromaDB but more powerful)
    """

    def __init__(
        self,
        collection_name: str = "ml_documents",
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        distance: str = "Cosine",  # Cosine | Euclid | Dot
        dimension: int = 384,
        on_disk: bool = False,     # Store vectors on disk (for large collections)
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self.distance = distance
        self.dimension = dimension
        self.on_disk = on_disk
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._init_client()
        return self._client

    def _init_client(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

        logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
        self._client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            timeout=30,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
        dist_map = {"Cosine": Distance.COSINE, "Euclid": Distance.EUCLID, "Dot": Distance.DOT}
        dist = dist_map.get(self.distance, Distance.COSINE)

        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=dist, on_disk=self.on_disk),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name} (dim={self.dimension})")
        else:
            logger.info(f"Using existing Qdrant collection: {self.collection_name}")

    def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        metadatas: list[dict],
    ) -> int:
        """Upsert embeddings into Qdrant."""
        from qdrant_client.models import PointStruct

        if not chunk_ids:
            return 0

        points = []
        for chunk_id, emb, content, meta in zip(chunk_ids, embeddings, contents, metadatas):
            # Qdrant requires UUID or integer IDs; use hash of chunk_id
            point_id = self._to_qdrant_id(chunk_id)
            payload = {**meta, "content": content, "chunk_id": chunk_id}
            points.append(PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload=payload,
            ))

        # Batch upsert
        batch_size = 100
        added = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                added += len(batch)
            except Exception as e:
                logger.error(f"Qdrant upsert error: {e}")

        logger.info(f"Qdrant: upserted {added} points")
        return added

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search Qdrant with optional payload filtering."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            query_filter = Filter(must=conditions)

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

        search_results = []
        for rank, hit in enumerate(results):
            payload = hit.payload or {}
            search_results.append(SearchResult(
                chunk_id=payload.get("chunk_id", str(hit.id)),
                doc_id=payload.get("doc_id", ""),
                content=payload.get("content", ""),
                score=round(hit.score, 4),
                metadata={k: v for k, v in payload.items() if k != "content"},
                rank=rank,
            ))
        return search_results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all points with matching doc_id payload field."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            count = result.result.get("deleted", 0) if hasattr(result, "result") else 0
            logger.info(f"Qdrant: deleted chunks for doc {doc_id[:8]}")
            return count
        except Exception as e:
            logger.error(f"Qdrant delete error: {e}")
            return 0

    def get_stats(self) -> dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "backend": "qdrant",
                "collection": self.collection_name,
                "total_vectors": info.points_count,
                "dimension": self.dimension,
                "distance": self.distance,
                "status": str(info.status),
            }
        except Exception as e:
            return {"backend": "qdrant", "error": str(e)}

    def save(self, path: str):
        """Qdrant persists automatically. Snapshot for backup."""
        logger.info("Qdrant: creating snapshot for backup")
        try:
            self.client.create_snapshot(collection_name=self.collection_name)
        except Exception as e:
            logger.warning(f"Qdrant snapshot failed: {e}")

    def load(self, path: str):
        """Qdrant loads from server. Use snapshot restore for backup recovery."""
        logger.info("Qdrant: connecting to server (data already persisted)")

    def _to_qdrant_id(self, chunk_id: str) -> str:
        """Convert chunk_id string to valid Qdrant UUID."""
        # Qdrant accepts UUID strings
        try:
            return str(uuid.UUID(chunk_id))
        except ValueError:
            # Hash the chunk_id into a UUID namespace
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
