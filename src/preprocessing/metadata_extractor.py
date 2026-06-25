"""
Metadata Extractor - Extracts rich metadata from documents for filtering and search.
Metadata enables precise filtering during retrieval (e.g., by author, topic, year).
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ML topic taxonomy for automatic classification
ML_TOPICS = {
    "deep_learning": ["neural network", "deep learning", "backpropagation", "activation function",
                      "dropout", "batch normalization", "convolutional", "recurrent"],
    "nlp": ["natural language", "text classification", "tokenization", "embedding", "transformer",
            "attention", "bert", "gpt", "language model", "sentiment"],
    "computer_vision": ["image classification", "object detection", "segmentation", "cnn",
                        "convolutional", "resnet", "vision transformer", "yolo"],
    "reinforcement_learning": ["reinforcement learning", "reward", "policy", "q-learning",
                                "markov", "agent", "environment", "exploration"],
    "optimization": ["gradient descent", "adam", "sgd", "learning rate", "optimizer",
                     "loss function", "convergence", "regularization"],
    "retrieval": ["retrieval", "search", "rag", "vector database", "embedding", "similarity",
                  "dense retrieval", "sparse retrieval", "bm25", "faiss"],
    "generative": ["generative", "gan", "vae", "diffusion", "llm", "gpt", "prompt",
                   "fine-tuning", "instruction tuning"],
    "evaluation": ["precision", "recall", "f1", "accuracy", "benchmark", "evaluation",
                   "metric", "performance", "ablation"],
}


@dataclass
class DocumentMetadata:
    """Rich metadata extracted from a document."""
    file_name: str
    file_type: str
    file_size_bytes: int
    doc_id: str
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    total_pages: int = 0
    word_count: int = 0
    char_count: int = 0
    section_count: int = 0
    has_references: bool = False
    has_abstract: bool = False
    has_equations: bool = False
    has_figures: bool = False
    has_tables: bool = False
    publication_year: Optional[int] = None
    extraction_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_path: str = ""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    language: str = "en"
    readability_score: float = 0.0
    content_quality: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "doc_id": self.doc_id,
            "title": self.title or "",
            "authors": self.authors,
            "abstract": self.abstract or "",
            "keywords": self.keywords,
            "topics": self.topics,
            "total_pages": self.total_pages,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "section_count": self.section_count,
            "has_references": self.has_references,
            "has_abstract": self.has_abstract,
            "has_equations": self.has_equations,
            "has_figures": self.has_figures,
            "has_tables": self.has_tables,
            "publication_year": self.publication_year,
            "extraction_timestamp": self.extraction_timestamp,
            "source_path": self.source_path,
            "arxiv_id": self.arxiv_id or "",
            "doi": self.doi or "",
            "language": self.language,
            "readability_score": self.readability_score,
            "content_quality": self.content_quality,
        }


class MetadataExtractor:
    """Extracts structured metadata from document text and file properties."""

    ARXIV_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
    DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
    SECTION_PATTERNS = [
        re.compile(r"^\s*\d+\.?\s+[A-Z][A-Za-z\s]+$", re.MULTILINE),
        re.compile(r"^\s*[A-Z][A-Z\s]{3,}$", re.MULTILINE),
    ]
    ABSTRACT_PATTERN = re.compile(
        r"(?:Abstract|ABSTRACT)\s*[:\-]?\s*(.{100,1000}?)(?:\n\n|\n[A-Z]|\d+\.)",
        re.DOTALL | re.IGNORECASE,
    )

    def extract(self, text, file_name, file_type, file_size_bytes, doc_id, existing_metadata=None, total_pages=1) -> DocumentMetadata:
        existing = existing_metadata or {}
        meta = DocumentMetadata(
            file_name=file_name,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            doc_id=doc_id,
            total_pages=total_pages,
            word_count=len(text.split()),
            char_count=len(text),
        )

        meta.title = existing.get("title") or self._extract_title(text)
        meta.authors = self._parse_authors(existing.get("author", ""))
        meta.abstract = self._extract_abstract(text)
        meta.has_abstract = meta.abstract is not None
        meta.publication_year = self._extract_year(text, existing)
        meta.topics = self._classify_topics(text)
        
        meta.content_quality = self._assess_quality(text, meta)
        meta.readability_score = self._compute_readability(text)
        return meta

    def _extract_title(self, text: str) -> Optional[str]:
        # Ambiguous variable 'l' fixed to 'line'
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:10]:
            if 10 < len(line) < 200 and not line.isupper():
                if "@" not in line and not re.search(r"\d{4}", line):
                    return line
        return None

    def _parse_authors(self, author_string: str) -> list[str]:
        if not author_string:
            return []
        authors = re.split(r";|\band\b|,(?=\s+[A-Z])", author_string)
        return [a.strip() for a in authors if a.strip()]

    def _extract_year(self, text: str, existing_meta: dict) -> Optional[int]:
        creation_date = existing_meta.get("creation_date", "")
        if creation_date:
            year_match = re.search(r"(20\d{2}|19\d{2})", str(creation_date))
            if year_match:
                return int(year_match.group(1))
        
        matches = self.YEAR_PATTERN.findall(text[:500])
        if matches:
            years = [int(y) for y in matches if 1990 <= int(y) <= datetime.now().year]
            if years:
                return max(years)
        return None

    def _classify_topics(self, text: str) -> list[str]:
        lower_text = text.lower()
        found_topics = []
        for topic, keywords in ML_TOPICS.items():
            score = sum(1 for kw in keywords if kw in lower_text)
            if score >= 2:
                found_topics.append((topic, score))
        found_topics.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in found_topics[:3]]

    def _assess_quality(self, text: str, meta: DocumentMetadata) -> str:
        score = 0
        if meta.word_count > 500:
            score += 2
        if meta.word_count > 2000:
            score += 2
        if meta.has_abstract:
            score += 2
        if meta.has_references:
            score += 1
        if meta.title:
            score += 1
        if len(meta.topics) > 0:
            score += 1
        if meta.section_count > 3:
            score += 1
        
        if score >= 7:
            return "high"
        if score >= 4:
            return "medium"
        return "low"

    def _compute_readability(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.split()) > 3]
        if not sentences:
            return 0.0
        avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        score = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_word_len / 5)
        return round(max(0.0, min(100.0, score)), 1)