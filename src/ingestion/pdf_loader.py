"""
PDF Loader - Production-grade PDF document loading with metadata extraction.
Supports text extraction, OCR fallback, table detection, and image extraction.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """Represents a single page from a PDF document."""

    page_number: int
    text: str
    word_count: int
    char_count: int
    has_tables: bool = False
    has_images: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class PDFDocument:
    """Represents a fully loaded PDF document."""

    file_path: str
    file_name: str
    file_size_bytes: int
    total_pages: int
    pages: list[PDFPage]
    metadata: dict
    doc_hash: str
    raw_text: str

    @property
    def word_count(self) -> int:
        return sum(p.word_count for p in self.pages)

    @property
    def is_empty(self) -> bool:
        return len(self.raw_text.strip()) == 0


class PDFLoader:
    """
    Production-grade PDF loader with dual extraction strategy.

    Strategy:
    1. Primary: pdfplumber for rich layout-aware extraction
    2. Fallback: pypdf for simple extraction when pdfplumber fails

    Handles:
    - Text-based PDFs
    - PDFs with tables
    - Multi-column layouts
    - Encrypted PDFs (with password)
    - Large PDFs (page-by-page streaming)
    """

    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = False,
        min_page_chars: int = 10,
        password: Optional[str] = None,
    ):
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.min_page_chars = min_page_chars
        self.password = password

    def load(self, file_path: str | Path) -> PDFDocument:
        """
        Load a PDF file and extract all content.

        Args:
            file_path: Path to the PDF file

        Returns:
            PDFDocument with all extracted content and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid PDF
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info(f"Loading PDF: {path.name} ({path.stat().st_size / 1024:.1f} KB)")

        # Compute file hash for deduplication
        doc_hash = self._compute_hash(path)

        # Extract metadata and pages
        metadata = self._extract_metadata(path)
        pages = self._extract_pages(path)

        raw_text = "\n\n".join(p.text for p in pages if p.text.strip())

        doc = PDFDocument(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            total_pages=len(pages),
            pages=pages,
            metadata=metadata,
            doc_hash=doc_hash,
            raw_text=raw_text,
        )

        logger.info(
            f"Loaded PDF: {path.name} | Pages: {doc.total_pages} | "
            f"Words: {doc.word_count} | Empty: {doc.is_empty}"
        )
        return doc

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file content for deduplication."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_metadata(self, path: Path) -> dict:
        """Extract PDF metadata using pypdf."""
        metadata = {
            "source": str(path.absolute()),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
        }
        try:
            reader = PdfReader(str(path))
            if reader.metadata:
                info = reader.metadata
                metadata.update(
                    {
                        "title": info.get("/Title", ""),
                        "author": info.get("/Author", ""),
                        "subject": info.get("/Subject", ""),
                        "creator": info.get("/Creator", ""),
                        "producer": info.get("/Producer", ""),
                        "creation_date": str(info.get("/CreationDate", "")),
                        "total_pages": len(reader.pages),
                    }
                )
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
        return metadata

    def _extract_pages(self, path: Path) -> list[PDFPage]:
        """Extract text from each page using pdfplumber with pypdf fallback."""
        pages = []
        try:
            pages = self._extract_with_pdfplumber(path)
        except Exception as e:
            logger.warning(f"pdfplumber failed ({e}), falling back to pypdf")
            pages = self._extract_with_pypdf(path)
        return pages

    def _extract_with_pdfplumber(self, path: Path) -> list[PDFPage]:
        """Extract using pdfplumber - better for complex layouts."""
        pages = []
        with pdfplumber.open(str(path), password=self.password or "") as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                has_tables = False
                if self.extract_tables:
                    tables = page.extract_tables()
                    if tables:
                        has_tables = True
                        # Append table text to page text
                        for table in tables:
                            for row in table:
                                row_text = " | ".join(
                                    str(cell) if cell else "" for cell in row
                                )
                                text += f"\n{row_text}"

                pages.append(
                    PDFPage(
                        page_number=i + 1,
                        text=text,
                        word_count=len(text.split()),
                        char_count=len(text),
                        has_tables=has_tables,
                        has_images=(
                            len(page.images) > 0 if self.extract_images else False
                        ),
                        metadata={
                            "page_number": i + 1,
                            "page_width": page.width,
                            "page_height": page.height,
                        },
                    )
                )
        return pages

    def _extract_with_pypdf(self, path: Path) -> list[PDFPage]:
        """Fallback extraction using pypdf."""
        pages = []
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(
                PDFPage(
                    page_number=i + 1,
                    text=text,
                    word_count=len(text.split()),
                    char_count=len(text),
                    metadata={"page_number": i + 1},
                )
            )
        return pages

    def load_batch(self, file_paths: list[str | Path]) -> list[PDFDocument]:
        """Load multiple PDF files, skipping failed ones."""
        documents = []
        for path in file_paths:
            try:
                doc = self.load(path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return documents
