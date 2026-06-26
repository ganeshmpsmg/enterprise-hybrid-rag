"""
Ingestion Service - Orchestrates full document ingestion pipeline.
From raw bytes to indexed chunks.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from src.chunking.metadata_chunker import MetadataChunker
from src.chunking.recursive_chunker import RecursiveChunker
from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.metadata_extractor import MetadataExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.vectorstore.index_builder import IndexBuilder
from src.sparse_retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class IngestionService:
    """
    End-to-end document ingestion service.

    Pipeline:
    raw bytes -> load -> clean -> extract metadata -> chunk -> embed -> index
    """

    def __init__(
        self,
        index_builder: IndexBuilder,
        sparse_retriever: BM25Retriever,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.index_builder = index_builder
        self.sparse_retriever = sparse_retriever
        self.doc_loader = DocumentLoader()
        self.text_cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()
        self.chunker = MetadataChunker(
            base_chunker=RecursiveChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._indexed_docs: list[str] = []
        self._all_chunks: list = []

    content = await file.read()
    
    background_tasks.add_task(
        
        _ingestion_service.ingest_bytes,
        
        content,
        
        file.filename,
        file.content_type,   # optional but recommended
        
    )
        """
        Ingest document from raw bytes.
        """

        # Save to temp file for loaders that need file path
        ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return self.ingest_file(tmp_path, original_filename=filename)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def ingest_file(
        self, file_path: str, original_filename: Optional[str] = None
    ) -> dict:
        """Ingest a document from a file path."""
        path = Path(file_path)

        # 1. Load document
        doc = self.doc_loader.load(path)
        if hasattr(doc, "is_empty") and doc.is_empty:
            raise ValueError(f"Document is empty: {path.name}")
        elif not doc.content.strip():
            raise ValueError(f"Document has no extractable text: {path.name}")

        # 2. Clean text
        cleaned_text, _ = self.text_cleaner.clean(doc.content)

        # 3. Extract metadata
        metadata = self.metadata_extractor.extract(
            text=cleaned_text,
            file_name=original_filename or doc.file_name,
            file_type=doc.file_type,
            file_size_bytes=doc.metadata.get("file_size_bytes", 0),
            doc_id=doc.doc_id,
            existing_metadata=doc.metadata,
            total_pages=doc.total_pages,
        )

        # 4. Chunk document
        chunks = self.chunker.chunk_document(
            text=cleaned_text,
            doc_id=doc.doc_id,
            doc_metadata=metadata.to_dict(),
            page_contents=doc.page_contents,
        )

        if not chunks:
            raise ValueError(f"No chunks generated for {path.name}")

        # 5. Index in vector store
        index_result = self.index_builder.index_chunks(chunks)
        if not index_result.success:
            raise RuntimeError(f"Indexing failed: {index_result.error}")

        # 6. Update sparse retriever (BM25)
        existing_corpus = self.sparse_retriever._corpus
        new_texts = [c.content for c in chunks]
        new_ids = [c.chunk_id for c in chunks]
        new_metas = [{"doc_id": c.doc_id, **c.metadata} for c in chunks]

        self.sparse_retriever.fit(
            corpus=existing_corpus + new_texts,
            chunk_ids=(self.sparse_retriever._chunk_ids + new_ids),
            metadatas=(self.sparse_retriever._metadatas + new_metas),
        )
        print("=" * 50)
        print("BM25 FIT COMPLETE")
        print("Corpus size:", len(self.sparse_retriever._corpus))
        print("Chunk IDs:", len(self.sparse_retriever._chunk_ids))
        print("BM25 object:", self.sparse_retriever._bm25 is not None)
        print("=" * 50)

        self._indexed_docs.append(doc.doc_id)
        self._all_chunks.extend(chunks)

        logger.info(
            f"Ingested: {path.name} | doc_id={doc.doc_id[:8]} | "
            f"chunks={index_result.chunks_indexed}"
        )

        return {
            "file_id": doc.doc_id,
            "doc_id": doc.doc_id,
            "file_name": original_filename or doc.file_name,
            "chunks_indexed": index_result.chunks_indexed,
            "total_pages": doc.total_pages,
            "word_count": doc.word_count,
        }

    def get_stats(self) -> dict:
        return {
            "indexed_documents": len(self._indexed_docs),
            "total_chunks": len(self._all_chunks),
        }
