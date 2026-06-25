"""
Recursive Chunker - LangChain-style recursive character text splitter.
Best general-purpose chunker: splits on paragraphs, then sentences, then words.
"""
import logging
from typing import Optional
from src.chunking.chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a hierarchy of separators.

    Split hierarchy (tries each in order until chunks are small enough):
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentence-ending punctuation
    4. Spaces (word boundaries)
    5. Characters (last resort)

    This produces more semantically coherent chunks than fixed-size splitting
    because it tries to preserve paragraph and sentence boundaries.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        separators: Optional[list[str]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Recursively split text into chunks respecting semantic boundaries.
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        raw_splits = self._split_text(text, self.separators)

        # Merge small splits into chunks of target size with overlap
        merged = self._merge_splits(raw_splits)

        chunks = []
        char_offset = 0
        for i, content in enumerate(merged):
            start = text.find(content, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(content)

            chunk = self._build_chunk(
                content=content,
                doc_id=doc_id,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata={
                    **meta,
                    "chunk_strategy": "recursive",
                    "chunk_size_target": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunk_index": i,
                    "total_chunks": len(merged),
                },
            )
            chunks.append(chunk)
            char_offset = max(0, end - self.chunk_overlap)

        filtered = self._filter_chunks(chunks)
        logger.debug(
            f"Recursive chunker: {len(filtered)} chunks from doc {doc_id[:8]} "
            f"| avg_chars={self.get_stats(filtered).get('avg_chars', 0)}"
        )
        return filtered

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if not separators:
            # Base case: character-level split
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        good_splits = []
        current_batch = []

        for split in splits:
            if len(split) <= self.chunk_size:
                current_batch.append(split)
            else:
                # This split is too big; recurse
                if current_batch:
                    good_splits.extend(current_batch)
                    current_batch = []
                sub_splits = self._split_text(split, remaining_separators)
                good_splits.extend(sub_splits)

        if current_batch:
            good_splits.extend(current_batch)

        return [s for s in good_splits if s.strip()]

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """
        Merge small splits into target-size chunks with overlap.
        Uses a sliding window approach.
        """
        if not splits:
            return []

        merged_chunks = []
        current_splits = []
        current_len = 0

        for split in splits:
            split_len = len(split)

            if current_len + split_len > self.chunk_size and current_splits:
                # Emit current chunk
                chunk_text = " ".join(current_splits).strip()
                if chunk_text:
                    merged_chunks.append(chunk_text)
                # Handle overlap: keep last N chars worth of splits
                overlap_splits = []
                overlap_len = 0
                for s in reversed(current_splits):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_splits.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_splits = overlap_splits
                current_len = overlap_len

            current_splits.append(split)
            current_len += split_len

        # Emit last chunk
        if current_splits:
            chunk_text = " ".join(current_splits).strip()
            if chunk_text:
                merged_chunks.append(chunk_text)

        return merged_chunks
