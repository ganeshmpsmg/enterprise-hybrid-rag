import logging
import re
import inspect
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Setup logger
logger = logging.getLogger(__name__)

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
    has_abstract: bool = False
    extraction_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "doc_id": self.doc_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "has_abstract": self.has_abstract,
            "extraction_timestamp": self.extraction_timestamp,
        }

class MetadataExtractor:
    """Extracts structured metadata from document text."""

    ABSTRACT_PATTERN = re.compile(
        r"(?:Abstract|ABSTRACT)\s*[:\-]?\s*(.{100,1000}?)(?:\n\n|\n[A-Z]|\d+\.)",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self):
        # Debugging: Log where this class is actually being loaded from
        file_path = os.path.abspath(inspect.getfile(MetadataExtractor))
        logger.warning(f"MetadataExtractor initialized from: {file_path}")

    def extract(
        self, text, file_name, file_type, file_size_bytes, doc_id, **kwargs
    ) -> DocumentMetadata:
        """Main extraction entry point."""
        meta = DocumentMetadata(
            file_name=file_name,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            doc_id=doc_id,
        )
        
        # Explicitly calling the method defined in this class
        meta.abstract = self._extract_abstract(text)
        meta.has_abstract = meta.abstract is not None
        
        return meta

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extracts abstract text using regex pattern."""
        match = self.ABSTRACT_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None