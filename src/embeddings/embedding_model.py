"""
Embedding Model - Wraps sentence-transformers for production use.
Handles model loading, batching, caching, and device management.
"""
import hashlib
import logging
import time
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Production wrapper around SentenceTransformer embedding models.

    Features:
    - Lazy model loading (loaded on first use)
    - Automatic device selection (CUDA > MPS > CPU)
    - Batch processing with configurable batch size
    - Embedding normalization (for cosine similarity)
    - Simple in-memory LRU cache for repeated queries
    - Dimension validation

    Model: sentence-transformers/all-MiniLM-L6-v2
    - Embedding dim: 384
    - Max sequence length: 256 tokens
    - Speed: ~14,000 sentences/second on GPU
    - Size: 22MB
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
        cache_size: int = 1000,
        max_seq_length: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device or self._auto_select_device()
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_size = cache_size
        self.max_seq_length = max_seq_length
        self._model = None   # Lazy load
        self._cache: dict[str, np.ndarray] = {}  # Simple LRU cache
        self._cache_order: list[str] = []
        self._embed_count = 0
        self._total_time = 0.0

    @property
    def model(self):
        """Lazy load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text to embed

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts with batching and caching.

        Args:
            texts: List of input texts

        Returns:
            2D numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        # Separate cached and uncached texts
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts in batches
        if uncached_texts:
            t0 = time.perf_counter()
            embeddings = self._embed_with_model(uncached_texts)
            elapsed = time.perf_counter() - t0
            self._embed_count += len(uncached_texts)
            self._total_time += elapsed

            for idx, (orig_i, emb) in enumerate(zip(uncached_indices, embeddings)):
                results[orig_i] = emb
                self._store_cache(self._cache_key(uncached_texts[idx]), emb)

        return np.array(results)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.
        Some models use different prompts for queries vs. passages.
        """
        return self.embed(query)

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Embed a list of document passages."""
        return self.embed_batch(documents)

    def get_stats(self) -> dict:
        """Return embedding performance statistics."""
        return {
            "model": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "total_embedded": self._embed_count,
            "total_time_sec": round(self._total_time, 2),
            "avg_ms_per_text": round(
                (self._total_time / max(self._embed_count, 1)) * 1000, 2
            ),
            "cache_size": len(self._cache),
        }

    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            t0 = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
            elapsed = time.perf_counter() - t0
            dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded: {self.model_name} | dim={dim} | "
                f"device={self.device} | load_time={elapsed:.2f}s"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def _embed_with_model(self, texts: list[str]) -> np.ndarray:
        """Run model inference in batches."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=len(batch),
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def _cache_key(self, text: str) -> str:
        """Generate cache key from text hash."""
        return hashlib.md5(text.encode()).hexdigest()

    def _store_cache(self, key: str, embedding: np.ndarray):
        """Store embedding in LRU cache."""
        if len(self._cache) >= self.cache_size:
            # Evict oldest entry
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = embedding
        self._cache_order.append(key)

    def _auto_select_device(self) -> str:
        """Auto-select best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
