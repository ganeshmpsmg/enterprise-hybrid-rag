import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# --- Dictionary Definitions ---
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
    """Expands queries using domain-specific ML synonyms, acronyms, and decomposition."""

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
        use_synonyms=True,
        use_acronym_expansion=True,
        use_decomposition=False,
        max_expansions=3,
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
        for acr, full in self.ML_ACRONYMS.items():
            pattern = rf"\b{re.escape(acr)}\b"
            result = re.sub(pattern, f"{acr} ({full})", result, flags=re.IGNORECASE)
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


# --- Pipeline Execution ---
class RAGPipeline:
    def __init__(self, query_expander, ranking_pipeline, answer_generator):
        self.query_expander = query_expander
        self.ranking_pipeline = ranking_pipeline
        self.answer_generator = answer_generator

    def run(self, query: str):
        """
        Executes the RAG pipeline with correct type handling.
        """
        # 1. Expand query using positional argument 'query'
        expanded_queries = self.query_expander.expand(query)
        primary_query = expanded_queries[0] if expanded_queries else query

        # 2. Retrieve and rerank
        ranked_results = self.ranking_pipeline.retrieve_and_rerank(
            query=primary_query,
            top_k=5,
        )

        # 3. Convert objects to dicts as expected by the generator
        ranked_results_dicts = [r.to_dict() for r in ranked_results]

        # 4. Generate answer
        rag_answer = self.answer_generator.generate(
            query=primary_query,
            ranked_results=ranked_results_dicts,
        )

        return RAGResponse(
            answer=rag_answer.answer,
            citations=rag_answer.citations,
        )
