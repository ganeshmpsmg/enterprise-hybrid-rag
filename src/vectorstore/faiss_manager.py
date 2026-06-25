"""
FAISS Manager - Facebook AI Similarity Search vector store backend.
Best for: high-throughput, in-memory similarity search on CPU/GPU.
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np

from src.vectorstore.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class FAISSManager(VectorStore):
    """
    FAISS-based vector store for fast approximate nearest neighbor search.

    Index types:
    - IndexFlatL2:  Exact search, best accuracy, slower for large datasets
    - IndexIVFFlat: Inverted file index, fast approximate search, needs training
    - IndexHNSWFlat: Graph-based, very fast, high memory

    For RAG (up to ~1M chunks): IndexFlatIP (inner product = cosine with normalized vectors)
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "Flat",  # Flat | IVFFlat | HNSW
        metric: str = "cosine",
        nlist: int = 100,          # For IVFFlat: number of clusters
        nprobe: int = 10,          # For IVFFlat: clusters to search
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self._index = None
        # Metadata storage (FAISS stores only vectors, not metadata)
        self._chunk_ids: list[str] = []
        self._contents: list[str] = []
        self._metadatas: list[dict] = []
        self._doc_id_to_indices: dict[str, list[int]] = {}

    def _get_or_create_index(self):
        """Lazy create FAISS index."""
        if self._index is not None:
            return self._index
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        if self.index_type == "Flat":
            if self.metric == "cosine":
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine when normalized)
            else:
                self._index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self._index.nprobe = self.nprobe
        elif self.index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 = M parameter
        else:
            self._index = faiss.IndexFlatIP(self.dimension)

        logger.info(f"Created FAISS index: type={self.index_type}, dim={self.dimension}")
        return self._index

    def add_embeddings(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        contents: list[str],
        metadatas: list[dict],
    ) -> int:
        """Add embeddings to FAISS index."""
        if len(embeddings) == 0:
            return 0

        index = self._get_or_create_index()
        embeddings_f32 = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            import faiss
            faiss.normalize_L2(embeddings_f32)

        # Train IVFFlat if needed
        if self.index_type == "IVFFlat" and not index.is_trained:
            if len(embeddings_f32) >= self.nlist:
                logger.info(f"Training IVFFlat index with {len(embeddings_f32)} vectors")
                index.train(embeddings_f32)
            else:
                logger.warning(
                    f"Not enough vectors to train IVFFlat ({len(embeddings_f32)} < {self.nlist}). "
                    "Switching to Flat index."
                )
                import faiss as faiss_lib
                self._index = faiss_lib.IndexFlatIP(self.dimension)
                index = self._index

        start_idx = len(self._chunk_ids)
        index.add(embeddings_f32)

        # Store metadata
        for i, (cid, content, meta) in enumerate(zip(chunk_ids, contents, metadatas)):
            doc_id = meta.get("doc_id", cid)
            global_idx = start_idx + i
            self._chunk_ids.append(cid)
            self._contents.append(content)
            self._metadatas.append(meta)
            if doc_id not in self._doc_id_to_indices:
                self._doc_id_to_indices[doc_id] = []
            self._doc_id_to_indices[doc_id].append(global_idx)

        logger.info(f"FAISS: added {len(chunk_ids)} vectors. Total: {index.ntotal}")
        return len(chunk_ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search FAISS index for nearest neighbors."""
        index = self._get_or_create_index()
        if index.ntotal == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        if self.metric == "cosine":
            import faiss
            faiss.normalize_L2(query)

        # Search with extra candidates if we need to filter
        search_k = min(top_k * 3 if filter_metadata else top_k, index.ntotal)
        scores, indices = index.search(query, search_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            if idx >= len(self._chunk_ids):
                continue

            meta = self._metadatas[idx]

            # Apply metadata filters
            if filter_metadata:
                if not self._matches_filter(meta, filter_metadata):
                    continue

            results.append(SearchResult(
                chunk_id=self._chunk_ids[idx],
                doc_id=meta.get("doc_id", ""),
                content=self._contents[idx],
                score=float(score),
                metadata=meta,
                rank=len(results),
            ))

            if len(results) >= top_k:
                break

        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        FAISS doesn't support deletion directly.
        We mark deleted items and rebuild the index.
        """
        indices_to_delete = self._doc_id_to_indices.get(doc_id, [])
        if not indices_to_delete:
            return 0
        # Mark for deletion (rebuild required for FAISS)
        logger.warning(
            f"FAISS deletion of {len(indices_to_delete)} vectors for doc {doc_id}. "
            "Note: FAISS requires full index rebuild for deletion."
        )
        # Remove from metadata
        keep_indices = [
            i for i in range(len(self._chunk_ids))
            if i not in set(indices_to_delete)
        ]
        self._rebuild_from_indices(keep_indices)
        del self._doc_id_to_indices[doc_id]
        return len(indices_to_delete)

    def get_stats(self) -> dict:
        index = self._get_or_create_index()
        return {
            "backend": "faiss",
            "index_type": self.index_type,
            "total_vectors": index.ntotal,
            "dimension": self.dimension,
            "metric": self.metric,
            "unique_documents": len(self._doc_id_to_indices),
        }

    def save(self, path: str):
        """Save FAISS index and metadata to disk."""
        import faiss
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        metadata = {
            "chunk_ids": self._chunk_ids,
            "contents": self._contents,
            "metadatas": self._metadatas,
            "doc_id_to_indices": self._doc_id_to_indices,
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str):
        """Load FAISS index and metadata from disk."""
        import faiss
        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self._index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._chunk_ids = meta["chunk_ids"]
        self._contents = meta["contents"]
        self._metadatas = meta["metadatas"]
        self._doc_id_to_indices = meta["doc_id_to_indices"]
        logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")

    def _matches_filter(self, metadata: dict, filters: dict) -> bool:
        """Check if metadata matches all filter conditions."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True

    def _rebuild_from_indices(self, keep_indices: list[int]):
        """Rebuild index keeping only specified indices."""
        import faiss
        if not keep_indices:
            self._index = None
            self._chunk_ids = []
            self._contents = []
            self._metadatas = []
            return
        # This requires re-embedding which we can't do here
        # In production, use a different approach or Qdrant which supports deletion
        logger.warning("FAISS rebuild after deletion is not fully implemented. Use Qdrant for deletion-heavy workloads.")
