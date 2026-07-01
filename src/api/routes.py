import asyncio
import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.api.schemas import HealthResponse

# Global placeholders injected at runtime by main.py
_pipeline = None
_ingestion_service = None
_initialization_error = None
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str


router = APIRouter()
ASK_TIMEOUT = int(os.getenv("ASK_TIMEOUT", "60"))


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives file, reads bytes, and triggers the ingestion service.
    Removed doc_id as it is not supported by the current service signature.
    """
    global _ingestion_service

    if _ingestion_service is None:
        raise HTTPException(
            status_code=503,
            detail=_initialization_error or "Ingestion service not initialized.",
        )

    try:
        content = await file.read()

        # Offload synchronous ingestion to threadpool
        # Note: doc_id argument removed to match ingest_bytes signature
        result = await run_in_threadpool(
            _ingestion_service.ingest_bytes,
            content=content,
            filename=file.filename,
            content_type=file.content_type,
        )
        return {"status": "success", "detail": result}
    except Exception as e:
        logger.exception("Error processing file upload")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_question(request: QueryRequest):
    """Processes a RAG query using the synchronous pipeline in a threadpool."""
    global _pipeline

    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=_initialization_error or "RAG pipeline is not initialized.",
        )

    try:
        fut = run_in_threadpool(_pipeline.run, query=request.query)
        response = await asyncio.wait_for(fut, timeout=ASK_TIMEOUT)
        return response.to_dict() if hasattr(response, "to_dict") else response
    except Exception as e:
        logger.exception("Error processing /ask request")
        if isinstance(e, asyncio.TimeoutError):
            raise HTTPException(
                status_code=504, detail=f"RAG pipeline timed out after {ASK_TIMEOUT}s"
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health():
    global _pipeline, _ingestion_service
    status = "ok" if _pipeline is not None else "degraded"
    vector_store_status = "initialized" if _pipeline is not None else "uninitialized"
    return {
        "status": status,
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "vector_store_status": vector_store_status,
        "total_documents": 0,
        "total_chunks": 0,
        "uptime_seconds": 0.0,
    }


@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.get("/debug")
async def debug():
    global _pipeline, _ingestion_service
    return {
        "pipeline_initialized": _pipeline is not None,
        "ingestion_initialized": _ingestion_service is not None,
        "initialization_error": _initialization_error,
    }
