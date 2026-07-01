"""
TF-IDF Retriever - Scikit-learn based TF-IDF sparse retrieval.
Complements BM25; faster to update incrementally.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """
    TF-IDF based sparse retrieval using scikit-learn.

    TF-IDF (Term Frequency - Inverse Document Frequency):
    - TF: How often a term appears in a document
    - IDF: Log of (total docs / docs containing term) -- penalizes common words
    - Score: TF * IDF

    Advantages vs BM25:
    - Supports incremental updates via partial_fit
    - Can use n-grams for phrase matching
    - Scikit-learn ecosystem integration

    Disadvantages vs BM25:
    - No document length normalization
    - BM25 generally outperforms on IR benchmarks
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        sublinear_tf: bool = True,  # Apply log normalization to TF
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]\w+\b",
        )
        self._doc_matrix = None
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict] = []

    def fit(
        self,
        corpus: list[str],
        chunk_ids: list[str],
        metadatas: Optional[list[dict]] = None,
    ):
        """Build TF-IDF index from corpus."""
        logger.info(f"Building TF-IDF index from {len(corpus)} documents...")
        self._corpus = corpus
        self._chunk_ids = chunk_ids
        self._metadatas = metadatas or [{} for _ in corpus]
        self._doc_matrix = self._vectorizer.fit_transform(corpus)
        vocab_size = len(self._vectorizer.vocabulary_)
        logger.info(f"TF-IDF index built: {len(corpus)} docs, vocab_size={vocab_size}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Search TF-IDF index."""
        if self._doc_matrix is None:
            raise RuntimeError("TF-IDF not fitted. Call fit() first.")

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._doc_matrix).flatten()

        if filter_metadata:
            for i, meta in enumerate(self._metadatas):
                if not self._matches_filter(meta, filter_metadata):
                    scores[i] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            results.append(
                {
                    "chunk_id": self._chunk_ids[idx],
                    "doc_id": self._metadatas[idx].get("doc_id", ""),
                    "content": self._corpus[idx],
                    "score": float(scores[idx]),
                    "metadata": self._metadatas[idx],
                    "rank": rank,
                    "retrieval_type": "sparse_tfidf",
                }
            )
        return results

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vectorizer": self._vectorizer,
            "doc_matrix": self._doc_matrix,
            "corpus": self._corpus,
            "chunk_ids": self._chunk_ids,
            "metadatas": self._metadatas,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"TF-IDF index saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._vectorizer = data["vectorizer"]
        self._doc_matrix = data["doc_matrix"]
        self._corpus = data["corpus"]
        self._chunk_ids = data["chunk_ids"]
        self._metadatas = data["metadatas"]
        logger.info(f"TF-IDF index loaded: {len(self._corpus)} documents")

    def get_stats(self) -> dict:
        return {
            "indexed": self._doc_matrix is not None,
            "total_documents": len(self._corpus),
            "vocabulary_size": (
                len(self._vectorizer.vocabulary_) if self._doc_matrix is not None else 0
            ),
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
        }

    def _matches_filter(self, metadata: dict, filters: dict) -> bool:
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True
