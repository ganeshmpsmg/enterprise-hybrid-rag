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
    # File metadata
    file_name: str
    file_type: str
    file_size_bytes: int
    doc_id: str

    # Content metadata
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    # Structural metadata
    total_pages: int = 0
    word_count: int = 0
    char_count: int = 0
    section_count: int = 0
    has_references: bool = False
    has_abstract: bool = False
    has_equations: bool = False
    has_figures: bool = False
    has_tables: bool = False

    # Temporal metadata
    publication_year: Optional[int] = None
    extraction_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Source metadata
    source_path: str = ""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None

    # Quality metadata
    language: str = "en"
    readability_score: float = 0.0
    content_quality: str = "unknown"  # high | medium | low

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
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
    """
    Extracts structured metadata from document text and file properties.

    Extraction strategies:
    - Rule-based: Regex patterns for title, authors, year, arXiv ID, DOI
    - Heuristic: Section detection, content quality scoring
    - ML-based: Topic classification using keyword matching
    """

    # Patterns for academic paper metadata
    ARXIV_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
    DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
    YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
    EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")

    # Section header patterns
    SECTION_PATTERNS = [
        re.compile(r"^\s*\d+\.?\s+[A-Z][A-Za-z\s]+$", re.MULTILINE),
        re.compile(r"^\s*[A-Z][A-Z\s]{3,}$", re.MULTILINE),
    ]

    ABSTRACT_PATTERN = re.compile(
        r"(?:Abstract|ABSTRACT)\s*[:\-]?\s*(.{100,1000}?)(?:\n\n|\n[A-Z]|\d+\.)",
        re.DOTALL | re.IGNORECASE,
    )

    def extract(
        self,
        text: str,
        file_name: str,
        file_type: str,
        file_size_bytes: int,
        doc_id: str,
        existing_metadata: Optional[dict] = None,
        total_pages: int = 1,
    ) -> DocumentMetadata:
        """
        Extract comprehensive metadata from document text.

        Args:
            text: Full document text
            file_name: Original filename
            file_type: File extension
            file_size_bytes: File size
            doc_id: Document hash/ID
            existing_metadata: Pre-extracted metadata (e.g., from PDF header)
            total_pages: Total page count

        Returns:
            DocumentMetadata with all extracted fields
        """
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

        # ── Extract from existing PDF metadata first ──
        meta.title = existing.get("title") or self._extract_title(text)
        meta.authors = self._parse_authors(existing.get("author", ""))

        # ── Text-based extractions ────────────────────
        meta.abstract = self._extract_abstract(text)
        meta.has_abstract = meta.abstract is not None
        meta.arxiv_id = self._extract_arxiv_id(text)
        meta.doi = self._extract_doi(text)
        meta.publication_year = self._extract_year(text, existing)
        meta.keywords = self._extract_keywords(text)
        meta.topics = self._classify_topics(text)
        meta.section_count = self._count_sections(text)

        # ── Structural features ───────────────────────
        lower_text = text.lower()
        meta.has_references = bool(re.search(r"\breferences\b|\bbibliography\b", lower_text))
        meta.has_equations = bool(re.search(r"=\s*\d|\\frac|\\sum|\\int|\$[^$]+\$", text))
        meta.has_figures = bool(re.search(r"\bfigure\s+\d+|\bfig\.\s+\d+", lower_text))
        meta.has_tables = bool(re.search(r"\btable\s+\d+", lower_text))

        # ── Quality assessment ────────────────────────
        meta.content_quality = self._assess_quality(text, meta)
        meta.readability_score = self._compute_readability(text)

        logger.debug(
            f"Extracted metadata for {file_name}: "
            f"title='{(meta.title or '')[:50]}', "
            f"topics={meta.topics}, quality={meta.content_quality}"
        )
        return meta

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first few lines heuristically."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Title is usually the first non-short line that isn't all caps
        for line in lines[:10]:
            if 10 < len(line) < 200 and not line.isupper():
                # Skip lines that look like author lines (contain @ or commas with names)
                if "@" not in line and not re.search(r"\d{4}", line):
                    return line
        return None

    def _parse_authors(self, author_string: str) -> list[str]:
        """Parse author string into individual names."""
        if not author_string:
            return []
        # Split on common delimiters: semicolons, commas (but not within names), "and"
        authors = re.split(r";|\band\b|,(?=\s+[A-Z])", author_string)
        return [a.strip() for a in authors if a.strip()]

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract section."""
        match = self.ABSTRACT_PATTERN.search(text)
        if match:
            abstract = match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r"\s+", " ", abstract)
            return abstract[:1000]  # Cap at 1000 chars
        return None

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        match = self.ARXIV_PATTERN.search(text)
        return match.group(1) if match else None

    def _extract_doi(self, text: str) -> Optional[str]:
        match = self.DOI_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_year(self, text: str, existing_meta: dict) -> Optional[int]:
        """Extract publication year."""
        # Try existing metadata first
        creation_date = existing_meta.get("creation_date", "")
        if creation_date:
            year_match = re.search(r"(20\d{2}|19\d{2})", str(creation_date))
            if year_match:
                return int(year_match.group(1))
        # Look in first 500 chars of text
        matches = self.YEAR_PATTERN.findall(text[:500])
        if matches:
            years = [int(y) for y in matches if 1990 <= int(y) <= datetime.now().year]
            if years:
                return max(years)  # Most recent year in header area
        return None

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from 'Keywords:' section if present."""
        kw_match = re.search(
            r"(?:Keywords?|Index Terms?)\s*[:\-]\s*([^\n]{10,300})",
            text, re.IGNORECASE
        )
        if kw_match:
            kw_text = kw_match.group(1)
            keywords = re.split(r"[,;·•]", kw_text)
            return [k.strip().lower() for k in keywords if 2 < len(k.strip()) < 50][:20]
        return []

    def _classify_topics(self, text: str) -> list[str]:
        """Classify document topics using keyword matching."""
        lower_text = text.lower()
        found_topics = []
        for topic, keywords in ML_TOPICS.items():
            score = sum(1 for kw in keywords if kw in lower_text)
            if score >= 2:  # At least 2 keyword matches
                found_topics.append((topic, score))
        # Return top 3 topics sorted by score
        found_topics.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in found_topics[:3]]

    def _count_sections(self, text: str) -> int:
        """Count number of sections in the document."""
        count = 0
        for pattern in self.SECTION_PATTERNS:
            count += len(pattern.findall(text))
        return min(count, 50)  # Cap at 50 to avoid false positives

    def _assess_quality(self, text: str, meta: DocumentMetadata) -> str:
        """Heuristic content quality assessment."""
        score = 0
        if meta.word_count > 500: score += 2
        if meta.word_count > 2000: score += 2
        if meta.has_abstract: score += 2
        if meta.has_references: score += 1
        if meta.title: score += 1
        if len(meta.topics) > 0: score += 1
        if meta.section_count > 3: score += 1
        if score >= 7: return "high"
        if score >= 4: return "medium"
        return "low"

    def _compute_readability(self, text: str) -> float:
        """
        Compute a simple readability score (0-100).
        Based on average sentence length and word length.
        Higher = more readable.
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 3]
        if not sentences:
            return 0.0
        avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        # Flesch-like formula (simplified)
        score = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_word_len / 5)
        return round(max(0.0, min(100.0, score)), 1)
