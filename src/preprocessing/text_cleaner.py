"""
Text Cleaner - Production-grade text cleaning for ML documents.
Handles common PDF extraction artifacts, Unicode issues, and noise.
"""
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Statistics from a cleaning operation."""
    original_length: int
    cleaned_length: int
    chars_removed: int
    patterns_applied: int

    @property
    def reduction_pct(self) -> float:
        if self.original_length == 0:
            return 0.0
        return round((self.chars_removed / self.original_length) * 100, 1)


class TextCleaner:
    """
    Cleans raw text extracted from PDFs and other document formats.

    Pipeline (in order):
    1. Unicode normalization (NFC)
    2. Fix common PDF extraction artifacts (ligatures, hyphens)
    3. Remove control characters and null bytes
    4. Fix excessive whitespace
    5. Remove headers/footers patterns
    6. Fix line break artifacts from PDF columns
    7. Optional: Remove references section
    8. Optional: Remove equations/formulas
    """

    # Ligature map for common PDF ligatures
    LIGATURE_MAP = {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st",
        "\ufb06": "st", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "--",
        "\u00a0": " ",   # Non-breaking space
        "\u200b": "",    # Zero-width space
        "\u200c": "",    # Zero-width non-joiner
        "\u200d": "",    # Zero-width joiner
        "\ufeff": "",    # BOM
    }

    # Patterns that are typically noise in academic/ML papers
    NOISE_PATTERNS = [
        # Page numbers: "- 5 -" or "Page 5 of 12"
        (r"-\s*\d+\s*-", " "),
        (r"[Pp]age\s+\d+\s+of\s+\d+", " "),
        # Copyright/watermark lines
        (r"©\s*\d{4}.*?(?:\n|$)", " "),
        (r"arXiv:\d{4}\.\d{4,5}v\d+", " "),
        # Repeated dashes/underscores (section separators)
        (r"[-_=]{3,}", " "),
        # Email addresses (often noise in extracted text)
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        # URLs
        (r"https?://\S+", "[URL]"),
        # Multiple exclamation/question marks
        (r"[!?]{2,}", "!"),
    ]

    def __init__(
        self,
        unicode_normalize: bool = True,
        fix_ligatures: bool = True,
        remove_headers_footers: bool = True,
        fix_hyphenation: bool = True,
        remove_references: bool = False,
        min_line_length: int = 3,
    ):
        self.unicode_normalize = unicode_normalize
        self.fix_ligatures = fix_ligatures
        self.remove_headers_footers = remove_headers_footers
        self.fix_hyphenation = fix_hyphenation
        self.remove_references = remove_references
        self.min_line_length = min_line_length

    def clean(self, text: str) -> tuple[str, CleaningStats]:
        """
        Clean text through the full cleaning pipeline.

        Args:
            text: Raw extracted text

        Returns:
            Tuple of (cleaned_text, CleaningStats)
        """
        original = text
        patterns_applied = 0

        # ── 1. Unicode normalization ───────────────────
        if self.unicode_normalize:
            text = unicodedata.normalize("NFC", text)
            patterns_applied += 1

        # ── 2. Fix ligatures ───────────────────────────
        if self.fix_ligatures:
            for ligature, replacement in self.LIGATURE_MAP.items():
                text = text.replace(ligature, replacement)
            patterns_applied += 1

        # ── 3. Remove control characters ──────────────
        text = self._remove_control_chars(text)
        patterns_applied += 1

        # ── 4. Fix PDF hyphenation (word- \nbreak) ──────
        if self.fix_hyphenation:
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
            patterns_applied += 1

        # ── 5. Remove headers/footers ──────────────────
        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)
            patterns_applied += 1

        # ── 6. Apply noise patterns ────────────────────
        for pattern, replacement in self.NOISE_PATTERNS:
            text = re.sub(pattern, replacement, text)
        patterns_applied += len(self.NOISE_PATTERNS)

        # ── 7. Fix whitespace ──────────────────────────
        text = self._normalize_whitespace(text)
        patterns_applied += 1

        # ── 8. Remove references section (optional) ────
        if self.remove_references:
            text = self._remove_references_section(text)
            patterns_applied += 1

        stats = CleaningStats(
            original_length=len(original),
            cleaned_length=len(text),
            chars_removed=len(original) - len(text),
            patterns_applied=patterns_applied,
        )
        logger.debug(
            f"Cleaned text: {stats.original_length} -> {stats.cleaned_length} chars "
            f"({stats.reduction_pct}% reduction)"
        )
        return text, stats

    def clean_batch(self, texts: list[str]) -> list[tuple[str, CleaningStats]]:
        """Clean a batch of texts."""
        return [self.clean(t) for t in texts]

    def _remove_control_chars(self, text: str) -> str:
        """Remove null bytes and other control characters, keeping newlines and tabs."""
        return "".join(
            c for c in text
            if c == "\n" or c == "\t" or not unicodedata.category(c).startswith("C")
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize various whitespace issues."""
        # Replace tabs with spaces
        text = text.replace("\t", " ")
        # Collapse multiple spaces into one
        text = re.sub(r" {2,}", " ", text)
        # Normalize multiple newlines (max 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip each line
        lines = [line.rstrip() for line in text.split("\n")]
        # Remove very short lines (likely noise) but keep empty lines as paragraph separators
        lines = [
            line for line in lines
            if len(line.strip()) >= self.min_line_length or line.strip() == ""
        ]
        return "\n".join(lines).strip()

    def _remove_headers_footers(self, text: str) -> str:
        """
        Remove repeated header/footer patterns.
        Heuristic: lines appearing 3+ times are likely headers/footers.
        """
        lines = text.split("\n")
        from collections import Counter
        line_counts = Counter(line.strip() for line in lines if line.strip())
        repeated = {line for line, count in line_counts.items() if count >= 3 and len(line) < 100}
        filtered = [line for line in lines if line.strip() not in repeated]
        return "\n".join(filtered)

    def _remove_references_section(self, text: str) -> str:
        """Remove everything after 'References' or 'Bibliography' section."""
        patterns = [
            r"\n\s*References\s*\n",
            r"\n\s*Bibliography\s*\n",
            r"\n\s*REFERENCES\s*\n",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return text[:match.start()].strip()
        return text
