"""
Keyword Search - Exact and fuzzy keyword matching for precise retrieval.
Complements BM25/TF-IDF for exact term requirements.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class KeywordSearcher:
    """
    Exact keyword search using regex and string matching.
    Useful for precise term requirements (e.g., exact model names, paper titles).
    """

    def __init__(self):
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict] = []

    def index(self, corpus: list[str], chunk_ids: list[str], metadatas: Optional[list[dict]] = None):
        self._corpus = corpus
        self._chunk_ids = chunk_ids
        self._metadatas = metadatas or [{} for _ in corpus]

    def search(
        self,
        query: str,
        top_k: int = 10,
        exact_match: bool = False,
        case_sensitive: bool = False,
    ) -> list[dict]:
        """
        Search corpus using keyword matching.

        Args:
            query: Query string (can contain multiple terms)
            top_k: Max results
            exact_match: If True, require all query terms; if False, any term matches
            case_sensitive: Case sensitivity

        Returns:
            Results sorted by match count
        """
        keywords = query.split() if not exact_match else [query]
        flags = 0 if case_sensitive else re.IGNORECASE

        results = []
        for i, doc in enumerate(self._corpus):
            match_count = 0
            matched_terms = []
            for kw in keywords:
                pattern = re.escape(kw)
                matches = re.findall(pattern, doc, flags)
                if matches:
                    match_count += len(matches)
                    matched_terms.append(kw)

            if exact_match and len(matched_terms) < len(keywords):
                continue
            if match_count == 0:
                continue

            score = match_count / max(len(doc.split()), 1)  # Normalize by doc length
            results.append({
                "chunk_id": self._chunk_ids[i],
                "doc_id": self._metadatas[i].get("doc_id", ""),
                "content": self._corpus[i],
                "score": round(score, 4),
                "metadata": self._metadatas[i],
                "matched_terms": matched_terms,
                "match_count": match_count,
                "retrieval_type": "keyword",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        for rank, r in enumerate(results[:top_k]):
            r["rank"] = rank
        return results[:top_k]
