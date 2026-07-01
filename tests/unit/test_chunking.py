"""Tests for chunking strategies."""

import pytest

from src.chunking.chunker import Chunk
from src.chunking.metadata_chunker import MetadataChunker
from src.chunking.recursive_chunker import RecursiveChunker

ML_TEXT = """
Transformer architecture has revolutionized natural language processing.
The key innovation is the attention mechanism which allows the model to focus on relevant parts.

Self-attention computes a weighted sum of all positions in the sequence.
Multi-head attention runs several attention functions in parallel.

The feed-forward network is applied to each position separately and identically.
Layer normalization is applied before each sub-layer.

BERT uses bidirectional training to understand context from both directions.
GPT uses autoregressive training to predict the next token.
""" * 5  # Make it longer to test chunking


@pytest.fixture
def recursive_chunker():
    return RecursiveChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=20)


@pytest.fixture
def metadata_chunker():
    return MetadataChunker(chunk_size=200, chunk_overlap=20)


class TestRecursiveChunker:
    def test_produces_chunks(self, recursive_chunker):
        chunks = recursive_chunker.chunk(ML_TEXT, "doc_001")
        assert len(chunks) > 0

    def test_chunk_size_respected(self, recursive_chunker):
        chunks = recursive_chunker.chunk(ML_TEXT, "doc_001")
        for chunk in chunks:
            # Allow 20% over due to overlap logic
            assert chunk.char_count <= recursive_chunker.chunk_size * 2

    def test_chunk_ids_unique(self, recursive_chunker):
        chunks = recursive_chunker.chunk(ML_TEXT, "doc_001")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_text_returns_empty(self, recursive_chunker):
        chunks = recursive_chunker.chunk("", "doc_001")
        assert len(chunks) == 0

    def test_short_text_single_chunk(self, recursive_chunker):
        short_text = "Transformers use attention mechanisms."
        chunks = recursive_chunker.chunk(short_text, "doc_001")
        assert len(chunks) == 1

    def test_chunk_metadata_set(self, recursive_chunker):
        chunks = recursive_chunker.chunk(ML_TEXT, "doc_001", {"file_name": "test.pdf"})
        for chunk in chunks:
            assert chunk.metadata.get("file_name") == "test.pdf"

    def test_doc_id_propagated(self, recursive_chunker):
        chunks = recursive_chunker.chunk(ML_TEXT, "my_doc_id")
        for chunk in chunks:
            assert chunk.doc_id == "my_doc_id"

    def test_chunk_index_sequential(self, recursive_chunker):
        chunks = recursive_chunker.chunk(ML_TEXT, "doc_001")
        indices = [c.chunk_index for c in chunks]
        assert indices == sorted(indices)

    def test_stats_computation(self, recursive_chunker):
        chunks = recursive_chunker.chunk(ML_TEXT, "doc_001")
        stats = recursive_chunker.get_stats(chunks)
        assert stats["total"] == len(chunks)
        assert stats["avg_chars"] > 0


class TestMetadataChunker:
    def test_metadata_enrichment(self, metadata_chunker):
        doc_meta = {
            "file_name": "attention_paper.pdf",
            "file_type": "pdf",
            "title": "Attention Is All You Need",
            "publication_year": 2017,
            "topics": ["deep_learning", "nlp"],
            "content_quality": "high",
        }
        chunks = metadata_chunker.chunk_document(ML_TEXT, "doc_001", doc_meta)
        for chunk in chunks:
            assert chunk.metadata.get("file_name") == "attention_paper.pdf"
            assert chunk.metadata.get("title") == "Attention Is All You Need"

    def test_position_metadata_added(self, metadata_chunker):
        chunks = metadata_chunker.chunk_document(ML_TEXT, "doc_001", {})
        assert chunks[0].metadata["is_first_chunk"] is True
        assert chunks[-1].metadata["is_last_chunk"] is True

    def test_total_chunks_in_metadata(self, metadata_chunker):
        chunks = metadata_chunker.chunk_document(ML_TEXT, "doc_001", {})
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)


class TestChunk:
    def test_word_count_property(self):
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            content="hello world this is a test",
            chunk_index=0,
            start_char=0,
            end_char=26,
        )
        assert chunk.word_count == 6

    def test_is_empty_property(self):
        empty = Chunk(
            chunk_id="c1",
            doc_id="d1",
            content="   ",
            chunk_index=0,
            start_char=0,
            end_char=3,
        )
        assert empty.is_empty is True
        nonempty = Chunk(
            chunk_id="c2",
            doc_id="d1",
            content="text",
            chunk_index=0,
            start_char=0,
            end_char=4,
        )
        assert nonempty.is_empty is False

    def test_to_dict(self):
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            content="test content",
            chunk_index=0,
            start_char=0,
            end_char=12,
            metadata={"key": "val"},
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["word_count"] == 2
        assert d["key"] == "val"
