"""
Metadata Chunker - Attaches rich metadata to chunks from any chunking strategy.
Acts as a wrapper/decorator over other chunkers to enrich chunk metadata.
"""
import logging
from typing import Optional
from src.chunking.chunker import BaseChunker, Chunk
from src.chunking.recursive_chunker import RecursiveChunker

logger = logging.getLogger(__name__)


class MetadataChunker:
    """
    Wraps any chunker and enriches chunks with document-level metadata.

    This ensures every chunk carries:
    - Document source information (file_name, file_type, doc_id)
    - Position information (chunk_index, total_chunks, is_first, is_last)
    - Content statistics (word_count, char_count)
    - Page number (if available from page_contents)
    - Custom metadata from the document

    In RAG systems, metadata on chunks enables:
    - Filtered retrieval ("only from paper X")
    - Provenance tracking ("answer came from page 5 of paper Y")
    - Quality-weighted scoring
    """

    def __init__(
        self,
        base_chunker: Optional[BaseChunker] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.base_chunker = base_chunker or RecursiveChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        doc_metadata: dict,
        page_contents: Optional[list[dict]] = None,
    ) -> list[Chunk]:
        """
        Chunk a document and enrich each chunk with metadata.

        Args:
            text: Full document text
            doc_id: Document identifier
            doc_metadata: Document-level metadata dict
            page_contents: Optional per-page content for page number mapping

        Returns:
            List of chunks with full metadata
        """
        # Get base chunks
        chunks = self.base_chunker.chunk(text, doc_id, doc_metadata)
        if not chunks:
            return []

        total = len(chunks)
        # Build page map if page_contents provided
        page_map = self._build_page_map(page_contents) if page_contents else {}

        for i, chunk in enumerate(chunks):
            # Add position metadata
            chunk.metadata.update({
                "is_first_chunk": i == 0,
                "is_last_chunk": i == total - 1,
                "total_chunks": total,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
            })

            # Add document metadata
            for key in ["file_name", "file_type", "title", "authors", "topics",
                        "publication_year", "source_path", "content_quality"]:
                if key in doc_metadata:
                    chunk.metadata[key] = doc_metadata[key]

            # Map chunk to page number
            if page_map:
                page_num = self._find_page_number(chunk, page_map)
                if page_num:
                    chunk.metadata["page_number"] = page_num

        logger.info(
            f"MetadataChunker: {total} chunks for doc {doc_id[:8]} "
            f"({doc_metadata.get('file_name', 'unknown')})"
        )
        return chunks

    def _build_page_map(self, page_contents: list[dict]) -> dict[int, tuple[int, int]]:
        """
        Build a map of page_number -> (start_char, end_char) in the full text.
        """
        page_map = {}
        offset = 0
        for page in page_contents:
            content = page.get("content", "")
            page_map[page["page_number"]] = (offset, offset + len(content))
            offset += len(content) + 2  # +2 for \n\n separator
        return page_map

    def _find_page_number(self, chunk: Chunk, page_map: dict) -> Optional[int]:
        """Find which page a chunk falls in based on character offset."""
        chunk_mid = (chunk.start_char + chunk.end_char) // 2
        for page_num, (start, end) in page_map.items():
            if start <= chunk_mid < end:
                return page_num
        return None
