"""
API Schemas - Pydantic models for request/response validation.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class FileTypeEnum(str, Enum):
    pdf = "pdf"
    docx = "docx"
    txt = "txt"
    md = "md"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    filter_file_type: Optional[FileTypeEnum] = Field(
        None, description="Filter by file type"
    )
    filter_doc_id: Optional[str] = Field(None, description="Filter by document ID")
    filter_topics: Optional[list[str]] = Field(None, description="Filter by ML topics")
    use_reranking: bool = Field(
        default=True, description="Apply cross-encoder reranking"
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_metadata: Optional[dict[str, Any]] = Field(None)
    conversation_history: Optional[list[dict[str, str]]] = Field(None)
    stream: bool = Field(default=False)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    retrieval_type: str = Field(default="hybrid", pattern="^(hybrid|dense|sparse)$")
    filter_metadata: Optional[dict[str, Any]] = Field(None)


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    rank: int
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total_results: int
    latency_ms: float
    retrieval_type: str = "hybrid"


class Citation(BaseModel):
    source_number: int
    file_name: str
    page_number: Optional[int] = None
    title: Optional[str] = None
    score: float = 0.0


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    context_chunks_used: int
    total_latency_ms: float
    model: str


class UploadResponse(BaseModel):
    file_id: str
    file_name: str
    file_size_mb: float
    chunks_indexed: int
    processing_time_sec: float
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_status: str
    total_documents: int
    total_chunks: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: int
