"""
Normalizer - Text normalization for consistent representation.
Handles case, special characters, abbreviations for ML domain.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Common ML/AI abbreviation expansions
ML_ABBREVIATIONS = {
    "ML": "machine learning",
    "DL": "deep learning",
    "NLP": "natural language processing",
    "CV": "computer vision",
    "RL": "reinforcement learning",
    "LLM": "large language model",
    "RAG": "retrieval augmented generation",
    "BERT": "bidirectional encoder representations from transformers",
    "GPT": "generative pre-trained transformer",
    "CNN": "convolutional neural network",
    "RNN": "recurrent neural network",
    "LSTM": "long short-term memory",
    "GAN": "generative adversarial network",
    "VAE": "variational autoencoder",
    "SGD": "stochastic gradient descent",
    "Adam": "adaptive moment estimation",
    "API": "application programming interface",
    "SVM": "support vector machine",
    "KNN": "k-nearest neighbors",
    "PCA": "principal component analysis",
    "TF-IDF": "term frequency inverse document frequency",
    "BM25": "best match 25",
    "FAISS": "facebook ai similarity search",
    "MRR": "mean reciprocal rank",
    "NDCG": "normalized discounted cumulative gain",
}


class TextNormalizer:
    """
    Normalizes text for consistent processing across the RAG pipeline.

    Normalization levels:
    - light: Fix whitespace and basic encoding only
    - standard: light + lowercase + special char handling
    - aggressive: standard + stemming + abbreviation expansion
    """

    def __init__(
        self,
        lowercase: bool = False,  # Keep case for RAG (important for proper nouns)
        expand_abbreviations: bool = False,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        normalize_numbers: bool = True,
    ):
        self.lowercase = lowercase
        self.expand_abbreviations = expand_abbreviations
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.normalize_numbers = normalize_numbers

    def normalize(self, text: str) -> str:
        """Apply normalization pipeline to text."""
        if not text or not text.strip():
            return ""

        # Fix encoding artifacts
        text = self._fix_encoding(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Expand abbreviations (query-time only, not for indexing)
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)

        # Lowercase (usually False for RAG to preserve named entities)
        if self.lowercase:
            text = text.lower()

        # Normalize numbers (e.g., "1,000" -> "1000")
        if self.normalize_numbers:
            text = self._normalize_numbers(text)

        # Remove punctuation (for sparse retrieval only)
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r"\b\d+\b", " ", text)

        # Final whitespace cleanup
        text = " ".join(text.split())

        return text.strip()

    def normalize_for_embedding(self, text: str) -> str:
        """
        Light normalization suitable for embedding generation.
        Preserves semantic content and case.
        """
        text = self._fix_encoding(text)
        text = self._normalize_whitespace(text)
        text = self._normalize_numbers(text)
        return text.strip()

    def normalize_for_sparse_retrieval(self, text: str) -> str:
        """
        Heavier normalization for BM25/TF-IDF.
        Lowercases and cleans punctuation for better term matching.
        """
        text = self.normalize_for_embedding(text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = " ".join(text.split())
        return text.strip()

    def normalize_query(self, query: str) -> str:
        """
        Normalize a search query.
        Expands abbreviations, fixes whitespace.
        """
        query = self._fix_encoding(query)
        query = self._normalize_whitespace(query)
        if self.expand_abbreviations:
            query = self._expand_abbreviations(query)
        return query.strip()

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Fix double-encoded UTF-8
        try:
            text = text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace characters."""
        text = re.sub(r"[\r\n]+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known ML abbreviations."""
        for abbr, expansion in ML_ABBREVIATIONS.items():
            # Match whole word only
            text = re.sub(
                rf"\b{re.escape(abbr)}\b",
                expansion,
                text,
                flags=re.IGNORECASE,
            )
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalize number formats."""
        # Remove commas from numbers: 1,000 -> 1000
        text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
        # Normalize scientific notation
        text = re.sub(r"(\d+)[eE]([+-]?\d+)", r"\1e\2", text)
        return text

    def batch_normalize(self, texts: list[str]) -> list[str]:
        """Normalize a batch of texts."""
        return [self.normalize(t) for t in texts]
