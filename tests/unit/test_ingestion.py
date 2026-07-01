"""Tests for document ingestion pipeline."""

import pytest

from src.ingestion.data_validator import DocumentValidator
from src.ingestion.file_manager import FileManager
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.text_cleaner import TextCleaner


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal fake PDF-like file for testing."""
    p = tmp_path / "test.pdf"
    p.write_bytes(b"%PDF-1.4 fake content for testing")
    return p


@pytest.fixture
def sample_txt(tmp_path):
    """Create a sample text file."""
    p = tmp_path / "test.txt"
    p.write_text(
        "This is a sample machine learning document about transformers and attention mechanisms."
    )
    return p


class TestDocumentValidator:
    def test_validates_existing_txt_file(self, sample_txt):
        validator = DocumentValidator()
        result = validator.validate(sample_txt)
        assert result.is_valid is True
        assert result.file_size_mb > 0

    def test_rejects_missing_file(self, tmp_path):
        validator = DocumentValidator()
        result = validator.validate(tmp_path / "nonexistent.pdf")
        assert result.is_valid is False
        assert any("not found" in e for e in result.errors)

    def test_rejects_oversized_file(self, tmp_path):
        validator = DocumentValidator(max_file_size_mb=0.0001)
        p = tmp_path / "big.txt"
        p.write_text("x" * 1000)
        result = validator.validate(p)
        assert result.is_valid is False

    def test_rejects_unsupported_extension(self, tmp_path):
        validator = DocumentValidator()
        p = tmp_path / "test.xyz"
        p.write_text("content")
        result = validator.validate(p)
        assert result.is_valid is False
        assert any("Unsupported" in e for e in result.errors)

    def test_detects_duplicate(self, sample_txt):
        validator = DocumentValidator()
        r1 = validator.validate(sample_txt)
        r2 = validator.validate(sample_txt)
        assert r1.is_valid is True
        assert r2.has_warnings  # Second time is duplicate

    def test_batch_validation(self, sample_txt, tmp_path):
        bad_file = tmp_path / "bad.xyz"
        bad_file.write_text("x")
        validator = DocumentValidator()
        results = validator.validate_batch([sample_txt, bad_file])
        assert results[str(sample_txt)].is_valid is True
        assert results[str(bad_file)].is_valid is False


class TestFileManager:
    def test_save_and_retrieve_file(self, tmp_dir):
        fm = FileManager(
            upload_dir=str(tmp_dir / "raw"),
            processed_dir=str(tmp_dir / "processed"),
            embeddings_dir=str(tmp_dir / "emb"),
        )
        content = b"Hello RAG world"
        info = fm.save_upload(content, "test.txt")
        assert info["file_id"]
        assert info["original_filename"] == "test.txt"
        assert info["file_size_bytes"] == len(content)

        retrieved_info = fm.get_file_info(info["file_id"])
        assert retrieved_info is not None

    def test_duplicate_detection(self, tmp_dir):
        fm = FileManager(
            upload_dir=str(tmp_dir / "raw"),
            processed_dir=str(tmp_dir / "processed"),
            embeddings_dir=str(tmp_dir / "emb"),
        )
        content = b"same content"
        info1 = fm.save_upload(content, "file1.txt")
        info2 = fm.save_upload(content, "file2.txt")
        assert info1["file_id"] == info2["file_id"]  # Same hash = same entry

    def test_mark_processed(self, tmp_dir):
        fm = FileManager(
            upload_dir=str(tmp_dir / "raw"),
            processed_dir=str(tmp_dir / "processed"),
            embeddings_dir=str(tmp_dir / "emb"),
        )
        info = fm.save_upload(b"data", "test.txt")
        fm.mark_processed(info["file_id"])
        updated = fm.get_file_info(info["file_id"])
        assert updated["processed"] is True

    def test_storage_stats(self, tmp_dir):
        fm = FileManager(
            upload_dir=str(tmp_dir / "raw"),
            processed_dir=str(tmp_dir / "processed"),
            embeddings_dir=str(tmp_dir / "emb"),
        )
        fm.save_upload(b"doc1", "f1.txt")
        fm.save_upload(b"doc2", "f2.txt")
        stats = fm.get_storage_stats()
        assert stats["total_files"] == 2


class TestTextCleaner:
    def test_basic_cleaning(self):
        cleaner = TextCleaner()
        text = "Hello\x00 World\t\ttest"
        cleaned, stats = cleaner.clean(text)
        assert "\x00" not in cleaned
        assert stats.original_length > 0

    def test_ligature_replacement(self):
        cleaner = TextCleaner()
        text = "\ufb01ne-\ufb02ow"  # fi and fl ligatures
        cleaned, _ = cleaner.clean(text)
        assert "fi" in cleaned
        assert "fl" in cleaned

    def test_fix_pdf_hyphenation(self):
        cleaner = TextCleaner()
        text = "trans-\nformer attention"
        cleaned, _ = cleaner.clean(text)
        assert "transformer" in cleaned

    def test_stats_returned(self):
        cleaner = TextCleaner()
        _, stats = cleaner.clean("Hello World   \n\n\n Test")
        assert stats.original_length > 0
        assert stats.patterns_applied > 0


class TestTextNormalizer:
    def test_normalize_for_embedding(self):
        norm = TextNormalizer()
        text = "  Hello   World  "
        result = norm.normalize_for_embedding(text)
        assert result == result.strip()

    def test_normalize_for_sparse_retrieval(self):
        norm = TextNormalizer()
        text = "Machine Learning! Transformers?"
        result = norm.normalize_for_sparse_retrieval(text)
        assert result == result.lower()
        assert "!" not in result

    def test_number_normalization(self):
        norm = TextNormalizer(normalize_numbers=True)
        result = norm.normalize("trained on 1,000,000 examples")
        assert "1000000" in result or "1,000,000" not in result
