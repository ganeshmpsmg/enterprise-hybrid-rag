"""
File Manager - Handles file storage, retrieval, and lifecycle management.
Manages the transition from raw uploads to processed documents.
"""

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FileManager:
    """
    Manages file storage for the RAG pipeline.

    Responsibilities:
    - Save uploaded files to raw storage with unique naming
    - Track file metadata in a simple registry
    - Move files between pipeline stages (raw -> processed)
    - Clean up temporary and orphaned files
    - Provide file lookup by ID or hash
    """

    def __init__(
        self,
        upload_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        embeddings_dir: str = "data/embeddings",
    ):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self._registry: dict[str, dict] = {}  # file_id -> metadata

        # Ensure directories exist
        for d in [self.upload_dir, self.processed_dir, self.embeddings_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        file_content: bytes,
        original_filename: str,
        content_type: str = "application/octet-stream",
    ) -> dict:
        """
        Save an uploaded file to raw storage.

        Args:
            file_content: Raw bytes of the uploaded file
            original_filename: Original filename from the client
            content_type: MIME type of the file

        Returns:
            dict with file_id, stored_path, file_hash, and metadata
        """
        # Generate unique ID and hash
        file_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Check for duplicates by hash
        existing = self._find_by_hash(file_hash)
        if existing:
            logger.info(
                f"Duplicate file detected: {original_filename} (hash: {file_hash[:8]})"
            )
            return existing

        # Create safe filename: {file_id}_{original_name}
        safe_name = self._sanitize_filename(original_filename)
        stored_name = f"{file_id[:8]}_{safe_name}"
        stored_path = self.upload_dir / stored_name

        # Write file
        stored_path.write_bytes(file_content)

        file_info = {
            "file_id": file_id,
            "original_filename": original_filename,
            "stored_filename": stored_name,
            "stored_path": str(stored_path.absolute()),
            "file_hash": file_hash,
            "file_size_bytes": len(file_content),
            "content_type": content_type,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "stage": "raw",
            "processed": False,
        }

        self._registry[file_id] = file_info
        logger.info(
            f"Saved upload: {original_filename} -> {stored_name} (ID: {file_id[:8]})"
        )
        return file_info

    def save_upload_from_path(self, source_path: str | Path) -> dict:
        """Save an existing file to raw storage."""
        path = Path(source_path)
        return self.save_upload(
            file_content=path.read_bytes(),
            original_filename=path.name,
            content_type="application/octet-stream",
        )

    def mark_processed(self, file_id: str) -> Optional[dict]:
        """Mark a file as successfully processed."""
        if file_id in self._registry:
            self._registry[file_id]["processed"] = True
            self._registry[file_id]["stage"] = "processed"
            self._registry[file_id][
                "processed_timestamp"
            ] = datetime.utcnow().isoformat()
            return self._registry[file_id]
        return None

    def get_file_info(self, file_id: str) -> Optional[dict]:
        """Retrieve file metadata by ID."""
        return self._registry.get(file_id)

    def get_file_path(self, file_id: str) -> Optional[Path]:
        """Get the stored path for a file ID."""
        info = self._registry.get(file_id)
        if info:
            return Path(info["stored_path"])
        return None

    def list_files(self, stage: Optional[str] = None) -> list[dict]:
        """List all tracked files, optionally filtered by stage."""
        files = list(self._registry.values())
        if stage:
            files = [f for f in files if f.get("stage") == stage]
        return sorted(files, key=lambda x: x["upload_timestamp"], reverse=True)

    def delete_file(self, file_id: str) -> bool:
        """Delete a file from storage and registry."""
        info = self._registry.get(file_id)
        if not info:
            return False
        path = Path(info["stored_path"])
        if path.exists():
            path.unlink()
            logger.info(
                f"Deleted file: {info['original_filename']} (ID: {file_id[:8]})"
            )
        del self._registry[file_id]
        return True

    def cleanup_raw_files(self, keep_processed: bool = True) -> int:
        """
        Clean up raw upload files.

        Args:
            keep_processed: If True, only delete files that have been processed

        Returns:
            Number of files deleted
        """
        deleted = 0
        for file_id, info in list(self._registry.items()):
            if keep_processed and not info.get("processed"):
                continue
            if self.delete_file(file_id):
                deleted += 1
        logger.info(f"Cleaned up {deleted} raw files")
        return deleted

    def get_storage_stats(self) -> dict:
        """Return storage statistics."""
        total_files = len(self._registry)
        processed = sum(1 for f in self._registry.values() if f.get("processed"))
        total_bytes = sum(
            Path(f["stored_path"]).stat().st_size
            for f in self._registry.values()
            if Path(f["stored_path"]).exists()
        )
        return {
            "total_files": total_files,
            "processed_files": processed,
            "pending_files": total_files - processed,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "upload_dir": str(self.upload_dir),
            "processed_dir": str(self.processed_dir),
        }

    def _sanitize_filename(self, filename: str) -> str:
        """Remove unsafe characters from filename."""
        import re

        # Keep only alphanumeric, dots, hyphens, underscores
        safe = re.sub(r"[^\w\-.]", "_", filename)
        # Collapse multiple underscores
        safe = re.sub(r"_+", "_", safe)
        return safe[:100]  # Limit length

    def _find_by_hash(self, file_hash: str) -> Optional[dict]:
        """Find a file by its content hash (for deduplication)."""
        for info in self._registry.values():
            if info.get("file_hash") == file_hash:
                return info
        return None
