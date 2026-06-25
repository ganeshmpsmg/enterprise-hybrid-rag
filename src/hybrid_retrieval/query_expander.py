"""
Query Expander - Expands queries with synonyms, related terms, and paraphrases.
Improves recall by generating multiple query variations.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ML/AI domain synonym dictionary
ML_SYNONYMS = {
    "neural network": [
        "deep learning model",
        "artificial neural network",
        "ANN",
        "DNN",
    ],
    "transformer": ["attention-based model", "BERT", "GPT", "encoder-decoder"],
    "embedding": ["vector representation", "dense vector", "latent representation"],
    "fine-tuning": ["transfer learning", "domain adaptation", "model adaptation"],
    "retrieval": ["search", "information retrieval", "document retrieval"],
    "attention mechanism": [
        "self-attention",
        "multi-head attention",
        "scaled dot-product attention",
    ],
    "gradient descent": ["SGD", "optimizer", "backpropagation"],
    "overfitting": ["regularization", "generalization", "variance"],
    "accuracy": ["performance", "precision", "recall", "F1"],
    "language model": ["LLM", "GPT", "BERT", "text generation"],
    "classification": ["categorization", "labeling", "prediction"],
    "clustering": ["unsupervised learning", "k-means", "grouping"],
    "dimensionality reduction": ["PCA", "t-SNE", "UMAP", "compression"],
    "loss function": ["objective function", "cost function", "training loss"],
    "hyperparameter": ["learning rate", "batch size", "configuration"],
    "inference": ["prediction", "forward pass", "deployment"],
    "training": ["learning", "fitting", "optimization"],
    "dataset": ["corpus", "data collection", "training data"],
    "model": ["algorithm", "architecture", "network"],
    "feature": ["attribute", "input", "variable"],
}


class QueryExpander:
    """
    Expands queries using multiple strategies:
    1. Synonym expansion (domain-specific ML synonyms)
    2. Acronym expansion (ML/AI abbreviations)
    3. Sub-query decomposition (for complex queries)
    4. LLM-based expansion (if LLM available)
    """

    ML_ACRONYMS = {
        "NLP": "natural language processing",
        "CV": "computer vision",
        "RL": "reinforcement learning",
        "ML": "machine learning",
        "DL": "deep learning",
        "LLM": "large language model",
        "RAG": "retrieval augmented generation",
        "BERT": "bidirectional encoder representations transformers",
        "GPT": "generative pre-trained transformer",
        "CNN": "convolutional neural network",
        "RNN": "recurrent neural network",
        "LSTM": "long short-term memory",
        "GAN": "generative adversarial network",
        "VAE": "variational autoencoder",
        "SVM": "support vector machine",
        "KNN": "k nearest neighbors",
        "PCA": "principal component analysis",
        "FAISS": "facebook AI similarity search",
        "BM25": "best match 25",
        "MRR": "mean reciprocal rank",
        "NDCG": "normalized discounted cumulative gain",
    }

    def __init__(
        self,
        use_synonyms: bool = True,
        use_acronym_expansion: bool = True,
        use_decomposition: bool = False,
        max_expansions: int = 3,
    ):
        self.use_synonyms = use_synonyms
        self.use_acronym_expansion = use_acronym_expansion
        self.use_decomposition = use_decomposition
        self.max_expansions = max_expansions

    def expand(self, query: str) -> list[str]:
        """Generate expanded query variants."""
        expansions = [query]

        if self.use_acronym_expansion:
            expanded_acronyms = self._expand_acronyms(query)
            if expanded_acronyms != query:
                expansions.append(expanded_acronyms)

        if self.use_synonyms:
            synonym_queries = self._expand_synonyms(query)
            expansions.extend(synonym_queries[: self.max_expansions])

        if self.use_decomposition:
            sub_queries = self._decompose_query(query)
            expansions.extend(sub_queries)

        seen = set()
        unique = []
        for q in expansions:
            q_lower = q.lower().strip()
            if q_lower not in seen and q_lower:
                seen.add(q_lower)
                unique.append(q)

        return unique

    def _expand_acronyms(self, query: str) -> str:
        result = query
        for acronym, expansion in self.ML_ACRONYMS.items():
            pattern = rf"\b{re.escape(acronym)}\b"
            result = re.sub(
                pattern, f"{acronym} ({expansion})", result, flags=re.IGNORECASE
            )
        return result

    def _expand_synonyms(self, query: str) -> list[str]:
        query_lower = query.lower()
        expansions = []
        for term, synonyms in ML_SYNONYMS.items():
            if term in query_lower:
                for syn in synonyms[:2]:
                    expanded = re.sub(
                        re.escape(term), syn, query_lower, flags=re.IGNORECASE
                    )
                    if expanded != query_lower:
                        expansions.append(expanded)
        return expansions

    def _decompose_query(self, query: str) -> list[str]:
        sub_queries = []
        parts = re.split(r"\s+and\s+|\s+or\s+|\s*,\s*", query, flags=re.IGNORECASE)
        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if len(part.split()) >= 3:
                    sub_queries.append(part)
        return sub_queries


class QueryRewriter:
    """LLM-based query rewriting."""

    REWRITE_PROMPT = """You are a search query optimizer for machine learning documentation.
Given the user query, generate {n} alternative search queries that:
1. Preserve the original intent
2. Use different vocabulary (synonyms, technical terms)
3. Are concise and searchable

Original query: {query}

Return ONLY the alternative queries, one per line, no numbering or explanation."""

    def __init__(
        self,
        llm_service=None,
        n_rewrites: int = 2,
        fallback_expander: Optional[QueryExpander] = None,
    ):
        self.llm_service = llm_service
        self.n_rewrites = n_rewrites
        self.fallback = fallback_expander or QueryExpander()

    def rewrite(self, query: str) -> list[str]:
        if self.llm_service:
            try:
                return self._llm_rewrite(query)
            except Exception as e:
                logger.warning(f"LLM rewrite failed: {e}. Using fallback.")
        return self.fallback.expand(query)

    def _llm_rewrite(self, query: str) -> list[str]:
        prompt = self.REWRITE_PROMPT.format(query=query, n=self.n_rewrites)
        response = self.llm_service.generate(prompt, max_tokens=200, temperature=0.3)
        # Fixed E741 by renaming 'l' to 'line'
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return [query] + lines[: self.n_rewrites]
