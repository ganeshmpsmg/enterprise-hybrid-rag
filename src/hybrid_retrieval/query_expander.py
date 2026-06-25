import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ML/AI domain synonym dictionary
ML_SYNONYMS = {
    "neural network": ["deep learning model", "artificial neural network"],
    "transformer": ["attention-based model", "BERT"],
    "embedding": ["vector representation", "dense vector"],
    "fine-tuning": ["transfer learning", "domain adaptation"],
    "retrieval": ["search", "information retrieval"],
    "attention mechanism": ["self-attention", "multi-head attention"],
    "gradient descent": ["SGD", "optimizer"],
    "overfitting": ["regularization", "generalization"],
    "accuracy": ["performance", "precision"],
    "language model": ["LLM", "GPT"],
    "classification": ["categorization", "labeling"],
    "clustering": ["unsupervised learning", "k-means"],
    "dimensionality reduction": ["PCA", "t-SNE"],
    "loss function": ["objective function", "cost function"],
    "hyperparameter": ["learning rate", "batch size"],
    "inference": ["prediction", "forward pass"],
    "training": ["learning", "fitting"],
    "dataset": ["corpus", "data collection"],
    "model": ["algorithm", "architecture"],
    "feature": ["attribute", "input"],
}

class QueryExpander:
    """
    Expands queries using domain-specific ML synonyms, acronyms, and decomposition.
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

    def expand_for_hybrid(self, query: str) -> dict:
        """Generate query variants for hybrid retrieval."""
        expanded = self.expand(query)
        return {
            "dense": expanded,
            "sparse": [query],
        }

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

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in expansions:
            q_lower = q.lower().strip()
            if q_lower not in seen and q_lower:
                seen.add(q_lower)
                unique.append(q)
        return unique

    def _expand_acronyms(self, query: str) -> str:
        """Replaces acronyms with 'ACRONYM (full form)' format."""
        result = query
        for acr, full in self.ML_ACRONYMS.items():
            pattern = rf"\b{re.escape(acr)}\b"
            result = re.sub(
                pattern, 
                f"{acr} ({full})", 
                result, 
                flags=re.IGNORECASE
            )
        return result

    def _expand_synonyms(self, query: str) -> list[str]:
        query_lower = query.lower()
        expansions = []
        for term, synonyms in ML_SYNONYMS.items():
            if term in query_lower:
                for syn in synonyms[:2]:
                    expanded = re.sub(re.escape(term), syn, query_lower, flags=re.IGNORECASE)
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
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return [query] + lines[: self.n_rewrites]