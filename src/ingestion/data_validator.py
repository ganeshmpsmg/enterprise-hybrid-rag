"""
Data Validator - Validates uploaded documents for quality, size, and format.
Prevents corrupted, empty, or malicious files from entering the pipeline.
"""
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    file_size_mb: float
    detected_type: str
    page_count: Optional[int] = None

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class DocumentValidator:
    """
    Validates documents before ingestion.

    Checks:
    1. File existence and readability
    2. File size limits
    3. File extension whitelist
    4. MIME type validation
    5. Content quality (non-empty, sufficient text)
    6. Duplicate detection (via hash)
    """

    def __init__(
        self,
        max_file_size_mb: float = 50.0,
        min_text_length: int = 50,
        allowed_extensions: Optional[set] = None,
    ):
        self.max_file_size_mb = max_file_size_mb
        self.min_text_length = min_text_length
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS
        self._known_hashes: set[str] = set()  # For duplicate detection

    def validate(self, file_path: str | Path) -> ValidationResult:
        """
        Validate a document file.

        Returns ValidationResult with is_valid flag and any errors/warnings.
        """
        path = Path(file_path)
        errors = []
        warnings = []

        # ── 1. File existence ──────────────────────────
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"File not found: {file_path}"],
                warnings=[],
                file_size_mb=0.0,
                detected_type="unknown",
            )

        # ── 2. File size ───────────────────────────────
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            errors.append(
                f"File too large: {size_mb:.1f} MB > {self.max_file_size_mb} MB limit"
            )
        if size_bytes == 0:
            errors.append("File is empty (0 bytes)")

        # ── 3. Extension check ─────────────────────────
        ext = path.suffix.lower()
        if ext not in self.allowed_extensions:
            errors.append(
                f"Unsupported extension: '{ext}'. "
                f"Allowed: {sorted(self.allowed_extensions)}"
            )

        # ── 4. MIME type ───────────────────────────────
        detected_type, _ = mimetypes.guess_type(str(path))
        detected_type = detected_type or "application/octet-stream"
        if detected_type not in ALLOWED_MIME_TYPES and ext in self.allowed_extensions:
            warnings.append(f"Unexpected MIME type: {detected_type}")

        # ── 5. Duplicate detection ─────────────────────
        if errors:  # Skip expensive hash if already invalid
            pass
        else:
            try:
                import hashlib
                sha256 = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
                doc_hash = sha256.hexdigest()
                if doc_hash in self._known_hashes:
                    warnings.append(f"Duplicate document detected (hash: {doc_hash[:8]}...)")
                else:
                    self._known_hashes.add(doc_hash)
            except Exception as e:
                warnings.append(f"Could not compute hash: {e}")

        # ── 6. Content quality (PDF-specific) ──────────
        page_count = None
        if ext == ".pdf" and not errors:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                page_count = len(reader.pages)
                sample_text = ""
                for page in reader.pages[:3]:
                    sample_text += page.extract_text() or ""
                if len(sample_text.strip()) < self.min_text_length:
                    warnings.append(
                        "PDF appears to have very little extractable text. "
                        "May be scanned/image-based (OCR not supported)."
                    )
            except Exception as e:
                warnings.append(f"Could not inspect PDF content: {e}")

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            file_size_mb=round(size_mb, 2),
            detected_type=detected_type,
            page_count=page_count,
        )

        if is_valid:
            logger.info(f"Validation passed: {path.name} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"Validation failed: {path.name} | Errors: {errors}")

        return result

    def validate_batch(
        self, file_paths: list[str | Path]
    ) -> dict[str, ValidationResult]:
        """Validate multiple files and return a map of path -> result."""
        results = {}
        for fp in file_paths:
            results[str(fp)] = self.validate(fp)
        valid_count = sum(1 for r in results.values() if r.is_valid)
        logger.info(f"Batch validation: {valid_count}/{len(file_paths)} valid")
        return results
