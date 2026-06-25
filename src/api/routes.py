"""
API Routes - FastAPI route handlers for all RAG endpoints.
Implements: /upload, /search, /retrieve, /ask, /health
"""
print("=" * 50)
print("ROUTES.PY LOADED")
print(__file__)
print("=" * 50)
import io
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AskRequest, AskResponse, ChunkResult, Citation,
    ErrorResponse, HealthResponse, RetrieveRequest,
    SearchRequest, SearchResponse, UploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Dependency injection ─────────────────────────────────────
# These are set by main.py on startup
_pipeline = None
_file_manager = None
_ingestion_service = None
_vector_store = None
_app_start_time = time.time()


def get_pipeline():
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    return _pipeline


def get_ingestion():
    if _ingestion_service is None:
        raise HTTPException(503, "Ingestion service not initialized")
    return _ingestion_service


# ── Health Check ─────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint.
    Returns system status, vector store stats, and uptime.
    """
    uptime = time.time() - _app_start_time
    vs_stats = {}
    total_chunks = 0

    if _vector_store:
        try:
            vs_stats = _vector_store.get_stats()
            total_chunks = vs_stats.get("total_vectors", 0)
        except Exception:
            pass

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store_status="connected" if _vector_store else "disconnected",
        total_documents=vs_stats.get("unique_documents", 0),
        total_chunks=total_chunks,
        uptime_seconds=round(uptime, 1),
    )


# ── Document Upload ───────────────────────────────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, TXT, MD)"),
):
    """
    Upload and ingest a document into the RAG system.

    Process:
    1. Validate file (type, size)
    2. Extract text and metadata
    3. Clean and chunk text
    4. Generate embeddings
    5. Index in vector store and build BM25 index

    Supported formats: PDF, DOCX, TXT, Markdown
    Max file size: 50MB
    """
    t0 = time.perf_counter()

    if not file.filename:
        raise HTTPException(400, "No filename provided")

    # Validate extension
    allowed_ext = {".pdf", ".docx", ".txt", ".md", ".markdown"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Supported: {allowed_ext}",
        )

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    max_size = 50 * 1024 * 1024  # 50MB
    if len(content) > max_size:
        raise HTTPException(413, f"File too large: {len(content)/(1024*1024):.1f}MB > 50MB limit")

    # Process document
    svc = get_ingestion()
    try:
        result = svc.ingest_bytes(
            content=content,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
        )
        elapsed = time.perf_counter() - t0
        return UploadResponse(
            file_id=result["file_id"],
            file_name=file.filename,
            file_size_mb=round(len(content) / (1024 * 1024), 3),
            chunks_indexed=result["chunks_indexed"],
            processing_time_sec=round(elapsed, 3),
            message=f"Successfully ingested {result['chunks_indexed']} chunks",
        )
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


# ── Hybrid Search ─────────────────────────────────────────────
@router.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(request: SearchRequest):
    """
    Hybrid search: dense + sparse + RRF fusion.

    Returns ranked document chunks without LLM generation.
    Use this endpoint when you want raw retrieval results.
    """
    pipeline = get_pipeline()
    t0 = time.perf_counter()

    # Build metadata filter
    filter_meta = {}
    if request.filter_file_type:
        filter_meta["file_type"] = request.filter_file_type.value
    if request.filter_doc_id:
        filter_meta["doc_id"] = request.filter_doc_id
    if request.filter_topics:
        filter_meta["topics"] = request.filter_topics

    try:
        # Hybrid retrieval
        hybrid_results = pipeline.hybrid_retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 3 if request.use_reranking else request.top_k,
            filter_metadata=filter_meta or None,
        )

        results = [r.to_dict() for r in hybrid_results]

        # Optional reranking
        if request.use_reranking and results:
            reranked = pipeline.ranking_pipeline.reranker.rerank(
                query=request.query,
                candidates=results,
                top_k=request.top_k,
            )
            results = reranked
        else:
            results = results[:request.top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        chunk_results = [
            ChunkResult(
                chunk_id=r.get("chunk_id", ""),
                doc_id=r.get("doc_id", ""),
                content=r.get("content", ""),
                score=r.get("rerank_score", r.get("score", 0.0)),
                rank=i,
                metadata=r.get("metadata", {}),
            )
            for i, r in enumerate(results)
        ]

        return SearchResponse(
            query=request.query,
            results=chunk_results,
            total_results=len(chunk_results),
            latency_ms=round(elapsed_ms, 2),
            retrieval_type="hybrid_reranked" if request.use_reranking else "hybrid",
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(500, f"Search failed: {str(e)}")


# ── Ask (Full RAG) ────────────────────────────────────────────
@router.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    print("ASK FUNCTION ENTERED")

    """
    Full RAG pipeline: retrieval + reranking + LLM answer generation.
    """
    
    pipeline = get_pipeline()

    try:
        print("\n" + "=" * 60)
        print("ASK ENDPOINT CALLED")
        print(
            "BM25 RETRIEVER ID:",
            id(pipeline.hybrid_retriever.sparse_retriever)
        )
        print(
            "BM25 EXISTS:",
            pipeline.hybrid_retriever.sparse_retriever._bm25 is not None
        )
        print(
            "CORPUS SIZE:",
            len(pipeline.hybrid_retriever.sparse_retriever._corpus)
        )
        print("=" * 60 + "\n")

        rag_response = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            conversation_history=request.conversation_history,
        )

        citations = [
            Citation(
                source_number=c.get("source_number", i + 1),
                file_name=c.get("file_name", ""),
                page_number=c.get("page_number"),
                title=c.get("title"),
                score=c.get("score", 0.0),
            )
            for i, c in enumerate(rag_response.citations)
        ]

        return AskResponse(
            query=request.query,
            answer=rag_response.answer,
            citations=citations,
            context_chunks_used=rag_response.context_chunks_used,
            total_latency_ms=rag_response.total_latency_ms,
            model=rag_response.pipeline_metadata.get(
                "model",
                "unknown"
            ),
        )

    except Exception as e:
        import traceback

        traceback.print_exc()

        print(
            "BM25 EXISTS AT ERROR:",
            pipeline.hybrid_retriever.sparse_retriever._bm25 is not None
        )

        print(
            "CORPUS SIZE AT ERROR:",
            len(
                pipeline.hybrid_retriever.sparse_retriever._corpus
            )
        )

        print("RAG PIPELINE ERROR:", repr(e))

        logger.error(
            f"Ask failed: {e}",
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )


# ── Stats ────────────────────────────────────────────────────
@router.get("/stats", tags=["System"])
async def get_stats():
    """Return pipeline performance statistics."""
    pipeline = get_pipeline()
    return pipeline.get_stats()
