"""
BM25 Retriever - Best Match 25 sparse retrieval algorithm.
BM25 is the gold standard for keyword-based document retrieval.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 (Best Match 25) retrieval implementation.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._bm25 = None
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    def fit(
        self,
        corpus: list[str],
        chunk_ids: list[str],
        metadatas: Optional[list[dict]] = None,
    ):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")

        logger.info(f"Building BM25 index from {len(corpus)} chunks...")
        self._corpus = corpus
        self._chunk_ids = chunk_ids
        self._metadatas = metadatas or [{} for _ in corpus]

        self._tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
        )
        logger.info(f"BM25 index built: {len(corpus)} documents.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Searches the BM25 index for the top_k most relevant chunks.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built; returning empty sparse results.")
            return []

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
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
                    "retrieval_type": "sparse_bm25",
                }
            )
        return results

    def add_documents(
        self,
        new_texts: list[str],
        new_ids: list[str],
        new_metadatas: Optional[list[dict]] = None,
    ):
        logger.info(f"Adding {len(new_texts)} documents to index...")
        self._corpus.extend(new_texts)
        self._chunk_ids.extend(new_ids)
        self._metadatas.extend(new_metadatas or [{} for _ in new_texts])
        self.fit(self._corpus, self._chunk_ids, self._metadatas)

    def get_stats(self) -> dict:
        if self._bm25 is None:
            return {"indexed": False, "total_documents": 0}
        return {
            "indexed": True,
            "total_documents": len(self._corpus),
            "vocabulary_size": len(self._bm25.idf),
        }

    def _tokenize(self, text: str) -> list[str]:
        """Improved tokenizer using regex to remove punctuation."""
        import re

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()
