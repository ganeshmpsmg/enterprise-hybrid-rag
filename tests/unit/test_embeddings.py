"""Tests for embedding model and pipeline."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestEmbeddingModel:
    """Test EmbeddingModel with mocked sentence-transformers."""

    @pytest.fixture
    def mock_model(self):
        """Mock SentenceTransformer."""
        with patch("src.embeddings.embedding_model.EmbeddingModel._load_model"):
            from src.embeddings.embedding_model import EmbeddingModel

            model = EmbeddingModel(model_name="test-model", device="cpu")
            # Mock the internal _model
            mock_st = MagicMock()
            mock_st.get_sentence_embedding_dimension.return_value = 384
            mock_st.encode.return_value = np.random.rand(1, 384).astype(np.float32)
            model._model = mock_st
            yield model

    def test_embed_returns_array(self, mock_model):
        result = mock_model.embed("test text")
        assert isinstance(result, np.ndarray)

    def test_embed_batch_shape(self, mock_model):
        mock_model._model.encode.return_value = np.random.rand(3, 384).astype(
            np.float32
        )
        result = mock_model.embed_batch(["text1", "text2", "text3"])
        assert result.shape == (3, 384)

    def test_caching(self, mock_model):
        mock_model._model.encode.return_value = np.random.rand(1, 384).astype(
            np.float32
        )
        _ = mock_model.embed("cached text")
        _ = mock_model.embed("cached text")
        # encode should only be called once (second is from cache)
        assert mock_model._model.encode.call_count == 1

    def test_stats_tracked(self, mock_model):
        mock_model._model.encode.return_value = np.random.rand(1, 384).astype(
            np.float32
        )
        _ = mock_model.embed("text1")
        stats = mock_model.get_stats()
        assert stats["total_embedded"] > 0


class TestEmbeddingUtils:
    def test_cosine_similarity_identical(self):
        from src.embeddings.embedding_utils import cosine_similarity

        v = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from src.embeddings.embedding_utils import cosine_similarity

        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_normalize_embeddings(self):
        from src.embeddings.embedding_utils import normalize_embeddings

        emb = np.random.rand(10, 384).astype(np.float32)
        normed = normalize_embeddings(emb)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_top_k_indices(self):
        from src.embeddings.embedding_utils import top_k_indices

        scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3])
        indices = top_k_indices(scores, k=3)
        assert list(indices) == [1, 3, 2]  # Sorted by score descending
