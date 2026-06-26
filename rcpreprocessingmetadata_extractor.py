warning: in the working copy of 'src/preprocessing/metadata_extractor.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/src/preprocessing/metadata_extractor.py b/src/preprocessing/metadata_extractor.py[m
[1mindex a691066..b95e38c 100644[m
[1m--- a/src/preprocessing/metadata_extractor.py[m
[1m+++ b/src/preprocessing/metadata_extractor.py[m
[36m@@ -1,111 +1,17 @@[m
[31m-"""[m
[31m-Metadata Extractor - Extracts rich metadata from documents for filtering and search.[m
[31m-Metadata enables precise filtering during retrieval (e.g., by author, topic, year).[m
[31m-"""[m
[31m-[m
 import logging[m
 import re[m
[32m+[m[32mimport inspect[m
[32m+[m[32mimport os[m
 from dataclasses import dataclass, field[m
 from datetime import datetime[m
 from typing import Optional[m
 [m
[32m+[m[32m# Setup logger[m
 logger = logging.getLogger(__name__)[m
 [m
[31m-# ML topic taxonomy for automatic classification[m
[31m-ML_TOPICS = {[m
[31m-    "deep_learning": [[m
[31m-        "neural network",[m
[31m-        "deep learning",[m
[31m-        "backpropagation",[m
[31m-        "activation function",[m
[31m-        "dropout",[m
[31m-        "batch normalization",[m
[31m-        "convolutional",[m
[31m-        "recurrent",[m
[31m-    ],[m
[31m-    "nlp": [[m
[31m-        "natural language",[m
[31m-        "text classification",[m
[31m-        "tokenization",[m
[31m-        "embedding",[m
[31m-        "transformer",[m
[31m-        "attention",[m
[31m-        "bert",[m
[31m-        "gpt",[m
[31m-        "language model",[m
[31m-        "sentiment",[m
[31m-    ],[m
[31m-    "computer_vision": [[m
[31m-        "image classification",[m
[31m-        "object detection",[m
[31m-        "segmentation",[m
[31m-        "cnn",[m
[31m-        "convolutional",[m
[31m-        "resnet",[m
[31m-        "vision transformer",[m
[31m-        "yolo",[m
[31m-    ],[m
[31m-    "reinforcement_learning": [[m
[31m-        "reinforcement learning",[m
[31m-        "reward",[m
[31m-        "policy",[m
[31m-        "q-learning",[m
[31m-        "markov",[m
[31m-        "agent",[m
[31m-        "environment",[m
[31m-        "exploration",[m
[31m-    ],[m
[31m-    "optimization": [[m
[31m-        "gradient descent",[m
[31m-        "adam",[m
[31m-        "sgd",[m
[31m-        "learning rate",[m
[31m-        "optimizer",[m
[31m-        "loss function",[m
[31m-        "convergence",[m
[31m-        "regularization",[m
[31m-    ],[m
[31m-    "retrieval": [[m
[31m-        "retrieval",[m
[31m-        "search",[m
[31m-        "rag",[m
[31m-        "vector database",[m
[31m-        "embedding",[m
[31m-        "similarity",[m
[31m-        "dense retrieval",[m
[31m-        "sparse retrieval",[m
[31m-        "bm25",[m
[31m-        "faiss",[m
[31m-    ],[m
[31m-    "generative": [[m
[31m-        "generative",[m
[31m-        "gan",[m
[31m-        "vae",[m
[31m-        "diffusion",[m
[31m-        "llm",[m
[31m-        "gpt",[m
[31m-        "prompt",[m
[31m-        "fine-tuning",[m
[31m-        "instruction tuning",[m
[31m-    ],[m
[31m-    "evaluation": [[m
[31m-        "precision",[m
[31m-        "recall",[m
[31m-        "f1",[m
[31m-        "accuracy",[m
[31m-        "benchmark",[m
[31m-        "evaluation",[m
[31m-        "metric",[m
[31m-        "performance",[m
[31m-        "ablation",[m
[31m-    ],[m
[31m-}[m
[31m-[m
[31m-[m
 @dataclass[m
 class DocumentMetadata:[m
     """Rich metadata extracted from a document."""[m
[31m-[m
     file_name: str[m
     file_type: str[m
     file_size_bytes: int[m
[36m@@ -113,27 +19,10 @@[m [mclass DocumentMetadata:[m
     title: Optional[str] = None[m
     authors: list[str] = field(default_factory=list)[m
     abstract: Optional[str] = None[m
[31m-    keywords: list[str] = field(default_factory=list)[m
[31m-    topics: list[str] = field(default_factory=list)[m
[31m-    total_pages: int = 0[m
[31m-    word_count: int = 0[m
[31m-    char_count: int = 0[m
[31m-    section_count: int = 0[m
[31m-    has_references: bool = False[m
     has_abstract: bool = False[m
[31m-    has_equations: bool = False[m
[31m-    has_figures: bool = False[m
[31m-    has_tables: bool = False[m
[31m-    publication_year: Optional[int] = None[m
     extraction_timestamp: str = field([m
         default_factory=lambda: datetime.utcnow().isoformat()[m
     )[m
[31m-    source_path: str = ""[m
[31m-    arxiv_id: Optional[str] = None[m
[31m-    doi: Optional[str] = None[m
[31m-    language: str = "en"[m
[31m-    readability_score: float = 0.0[m
[31m-    content_quality: str = "unknown"[m
 [m
     def to_dict(self) -> dict:[m
         return {[m
[36m@@ -141,146 +30,46 @@[m [mclass DocumentMetadata:[m
             "file_type": self.file_type,[m
             "file_size_bytes": self.file_size_bytes,[m
             "doc_id": self.doc_id,[m
[31m-            "title": self.title or "",[m
[32m+[m[32m            "title": self.title,[m
             "authors": self.authors,[m
[31m-            "abstract": self.abstract or "",[m
[31m-            "keywords": self.keywords,[m
[31m-            "topics": self.topics,[m
[31m-            "total_pages": self.total_pages,[m
[31m-            "word_count": self.word_count,[m
[31m-            "char_count": self.char_count,[m
[31m-            "section_count": self.section_count,[m
[31m-            "has_references": self.has_references,[m
[32m+[m[32m            "abstract": self.abstract,[m
             "has_abstract": self.has_abstract,[m
[31m-            "has_equations": self.has_equations,[m
[31m-            "has_figures": self.has_figures,[m
[31m-            "has_tables": self.has_tables,[m
[31m-            "publication_year": self.publication_year,[m
             "extraction_timestamp": self.extraction_timestamp,[m
[31m-            "source_path": self.source_path,[m
[31m-            "arxiv_id": self.arxiv_id or "",[m
[31m-            "doi": self.doi or "",[m
[31m-            "language": self.language,[m
[31m-            "readability_score": self.readability_score,[m
[31m-            "content_quality": self.content_quality,[m
         }[m
 [m
[31m-[m
 class MetadataExtractor:[m
[31m-    """Extracts structured metadata from document text and file properties."""[m
[32m+[m[32m    """Extracts structured metadata from document text."""[m
 [m
[31m-    ARXIV_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)[m
[31m-    DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)[m
[31m-    YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")[m
[31m-    SECTION_PATTERNS = [[m
[31m-        re.compile(r"^\s*\d+\.?\s+[A-Z][A-Za-z\s]+$", re.MULTILINE),[m
[31m-        re.compile(r"^\s*[A-Z][A-Z\s]{3,}$", re.MULTILINE),[m
[31m-    ][m
     ABSTRACT_PATTERN = re.compile([m
         r"(?:Abstract|ABSTRACT)\s*[:\-]?\s*(.{100,1000}?)(?:\n\n|\n[A-Z]|\d+\.)",[m
         re.DOTALL | re.IGNORECASE,[m
     )[m
 [m
[32m+[m[32m    def __init__(self):[m
[32m+[m[32m        # Debugging: Log where this class is actually being loaded from[m
[32m+[m[32m        file_path = os.path.abspath(inspect.getfile(MetadataExtractor))[m
[32m+[m[32m        logger.warning(f"MetadataExtractor initialized from: {file_path}")[m
[32m+[m
     def extract([m
[31m-        self,[m
[31m-        text,[m
[31m-        file_name,[m
[31m-        file_type,[m
[31m-        file_size_bytes,[m
[31m-        doc_id,[m
[31m-        existing_metadata=None,[m
[31m-        total_pages=1,[m
[32m+[m[32m        self, text, file_name, file_type, file_size_bytes, doc_id, **kwargs[m
     ) -> DocumentMetadata:[m
[31m-        existing = existing_metadata or {}[m
[32m+[m[32m        """Main extraction entry point."""[m
         meta = DocumentMetadata([m
             file_name=file_name,[m
             file_type=file_type,[m
             file_size_bytes=file_size_bytes,[m
             doc_id=doc_id,[m
[31m-            total_pages=total_pages,[m
[31m-            word_count=len(text.split()),[m
[31m-            char_count=len(text),[m
         )[m
[31m-[m
[31m-        meta.title = existing.get("title") or self._extract_title(text)[m
[31m-        meta.authors = self._parse_authors(existing.get("author", ""))[m
[32m+[m[41m        [m
[32m+[m[32m        # Explicitly calling the method defined in this class[m
         meta.abstract = self._extract_abstract(text)[m
         meta.has_abstract = meta.abstract is not None[m
[31m-        meta.publication_year = self._extract_year(text, existing)[m
[31m-        meta.topics = self._classify_topics(text)[m
[31m-[m
[31m-        meta.content_quality = self._assess_quality(text, meta)[m
[31m-        meta.readability_score = self._compute_readability(text)[m
[32m+[m[41m        [m
         return meta[m
 [m
[31m-    def _extract_title(self, text: str) -> Optional[str]:[m
[31m-        # Ambiguous variable 'l' fixed to 'line'[m
[31m-        lines = [line.strip() for line in text.split("\n") if line.strip()][m
[31m-        for line in lines[:10]:[m
[31m-            if 10 < len(line) < 200 and not line.isupper():[m
[31m-                if "@" not in line and not re.search(r"\d{4}", line):[m
[31m-                    return line[m
[31m-        return None[m
[31m-[m
[31m-    def _parse_authors(self, author_string: str) -> list[str]:[m
[31m-        if not author_string:[m
[31m-            return [][m
[31m-        authors = re.split(r";|\band\b|,(?=\s+[A-Z])", author_string)[m
[31m-        return [a.strip() for a in authors if a.strip()][m
[31m-[m
[31m-    def _extract_year(self, text: str, existing_meta: dict) -> Optional[int]:[m
[31m-        creation_date = existing_meta.get("creation_date", "")[m
[31m-        if creation_date:[m
[31m-            year_match = re.search(r"(20\d{2}|19\d{2})", str(creation_date))[m
[31m-            if year_match:[m
[31m-                return int(year_match.group(1))[m
[31m-[m
[31m-        matches = self.YEAR_PATTERN.findall(text[:500])[m
[31m-        if matches:[m
[31m-            years = [int(y) for y in matches if 1990 <= int(y) <= datetime.now().year][m
[31m-            if years:[m
[31m-                return max(years)[m
[31m-        return None[m
[31m-[m
[31m-    def _classify_topics(self, text: str) -> list[str]:[m
[31m-        lower_text = text.lower()[m
[31m-        found_topics = [][m
[31m-        for topic, keywords in ML_TOPICS.items():[m
[31m-            score = sum(1 for kw in keywords if kw in lower_text)[m
[31m-            if score >= 2:[m
[31m-                found_topics.append((topic, score))[m
[31m-        found_topics.sort(key=lambda x: x[1], reverse=True)[m
[31m-        return [t[0] for t in found_topics[:3]][m
[31m-[m
[31m-    def _assess_quality(self, text: str, meta: DocumentMetadata) -> str:[m
[31m-        score = 0[m
[31m-        if meta.word_count > 500:[m
[31m-            score += 2[m
[31m-        if meta.word_count > 2000:[m
[31m-            score += 2[m
[31m-        if meta.has_abstract:[m
[31m-            score += 2[m
[31m-        if meta.has_references:[m
[31m-            score += 1[m
[31m-        if meta.title:[m
[31m-            score += 1[m
[31m-        if len(meta.topics) > 0:[m
[31m-            score += 1[m
[31m-        if meta.section_count > 3:[m
[31m-            score += 1[m
[31m-[m
[31m-        if score >= 7:[m
[31m-            return "high"[m
[31m-        if score >= 4:[m
[31m-            return "medium"[m
[31m-        return "low"[m
[31m-[m
[31m-    def _compute_readability(self, text: str) -> float:[m
[31m-        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.split()) > 3][m
[31m-        if not sentences:[m
[31m-            return 0.0[m
[31m-        avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)[m
[31m-        words = text.split()[m
[31m-        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)[m
[31m-        score = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_word_len / 5)[m
[31m-        return round(max(0.0, min(100.0, score)), 1)[m
[32m+[m[32m    def _extract_abstract(self, text: str) -> Optional[str]:[m
[32m+[m[32m        """Extracts abstract text using regex pattern."""[m
[32m+[m[32m        match = self.ABSTRACT_PATTERN.search(text)[m
[32m+[m[32m        if match:[m
[32m+[m[32m            return match.group(1).strip()[m
[32m+[m[32m        return None[m
\ No newline at end of file[m
