"""Shared pytest fixtures for Enterprise RAG test suite."""

import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("VECTOR_STORE", "faiss")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def sample_ml_text():
    """Sample ML document text for testing."""
    return (
        """
    The Transformer model uses self-attention mechanisms to process sequences in parallel.
    Unlike RNNs which process tokens sequentially, transformers can attend to all positions simultaneously.
    Multi-head attention allows the model to jointly attend to information from different representation subspaces.
    BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep bidirectional representations.
    GPT (Generative Pre-trained Transformer) uses left-to-right language modeling for pre-training.
    The attention score is computed as: Attention(Q,K,V) = softmax(QK^T / sqrt(dk)) * V
    Gradient descent optimizes model parameters by computing gradients of the loss function.
    Batch normalization normalizes layer inputs to stabilize training and accelerate convergence.
    """
        * 10
    )  # Repeat to get enough text for chunking tests


@pytest.fixture(scope="session")
def sample_chunks(sample_ml_text):
    """Pre-computed chunks for testing."""
    from src.chunking.recursive_chunker import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    return chunker.chunk(sample_ml_text, "test_doc_001", {"file_name": "test.txt"})


@pytest.fixture
def tmp_dir():
    """Temporary directory that auto-cleans."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
