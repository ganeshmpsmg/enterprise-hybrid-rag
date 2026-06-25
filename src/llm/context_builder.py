"""
Context Builder - Assembles and ranks context for LLM answer generation.
Deduplicates, trims, and orders retrieved chunks optimally.
"""

import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds the final context window for LLM generation from ranked retrieval results.

    Tasks:
    1. Deduplicate overlapping chunks
    2. Order chunks for optimal LLM context (best at start and end - "lost in middle" problem)
    3. Trim to fit within token budget
    4. Extract source citations
    """

    def __init__(
        self,
        max_context_chars: int = 4000,
        max_chunks: int = 5,
        dedup_threshold: float = 0.85,
        position_strategy: str = "best_first",  # best_first | chronological | lost_in_middle
    ):
        self.max_context_chars = max_context_chars
        self.max_chunks = max_chunks
        self.dedup_threshold = dedup_threshold
        self.position_strategy = position_strategy

    def build(self, ranked_results: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Build context from ranked results.

        Args:
            ranked_results: List of retrieval results sorted by relevance

        Returns:
            Tuple of (context_chunks, source_citations)
        """
        if not ranked_results:
            return [], []

        # 1. Deduplicate
        deduplicated = self._deduplicate(ranked_results)

        # 2. Select top chunks within budget
        selected = self._select_within_budget(deduplicated)

        # 3. Reorder for LLM
        ordered = self._reorder_chunks(selected)

        # 4. Build source citations
        citations = self._build_citations(ordered)

        logger.debug(
            f"Context built: {len(ranked_results)} -> {len(selected)} chunks "
            f"({sum(len(c.get('content','')) for c in selected)} chars)"
        )
        return ordered, citations

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """Remove near-duplicate chunks using simple overlap detection."""
        unique = []
        seen_content_starts = set()
        for result in results:
            content = result.get("content", "")
            # Use first 80 chars as fingerprint
            fingerprint = content[:80].strip().lower()
            if fingerprint not in seen_content_starts:
                seen_content_starts.add(fingerprint)
                unique.append(result)
        return unique

    def _select_within_budget(self, results: list[dict]) -> list[dict]:
        """Select chunks within character budget and max_chunks limit."""
        selected = []
        total_chars = 0
        for result in results[: self.max_chunks]:
            content = result.get("content", "")
            if total_chars + len(content) > self.max_context_chars:
                # Truncate last chunk if needed
                remaining = self.max_context_chars - total_chars
                if remaining > 200:  # Only add if meaningful size
                    truncated = dict(result)
                    truncated["content"] = content[:remaining] + "..."
                    selected.append(truncated)
                break
            selected.append(result)
            total_chars += len(content)
        return selected

    def _reorder_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Reorder chunks to mitigate "Lost in the Middle" problem.
        Research shows LLMs best utilize content at the beginning and end of context.

        Strategies:
        - best_first: Most relevant first (default)
        - lost_in_middle: Most relevant at start + end, less relevant in middle
        - chronological: Order by page number / document position
        """
        if self.position_strategy == "lost_in_middle" and len(chunks) > 2:
            # Put best at start, second best at end, rest in middle
            sorted_chunks = sorted(
                chunks,
                key=lambda x: x.get("score", x.get("final_score", 0)),
                reverse=True,
            )
            if len(sorted_chunks) <= 2:
                return sorted_chunks
            best = sorted_chunks[0]
            second_best = sorted_chunks[1]
            middle = sorted_chunks[2:]
            return [best] + middle + [second_best]
        elif self.position_strategy == "chronological":
            return sorted(
                chunks,
                key=lambda x: (
                    x.get("metadata", {}).get("file_name", ""),
                    x.get("metadata", {}).get("page_number", 0),
                    x.get("metadata", {}).get("chunk_index", 0),
                ),
            )
        else:
            return chunks  # best_first: already sorted by retrieval

    def _build_citations(self, chunks: list[dict]) -> list[dict]:
        """Extract source citations from context chunks."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            citations.append(
                {
                    "source_number": i,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "file_name": meta.get("file_name", ""),
                    "page_number": meta.get("page_number"),
                    "title": meta.get("title", ""),
                    "score": chunk.get("final_score", chunk.get("score", 0)),
                }
            )
        return citations
