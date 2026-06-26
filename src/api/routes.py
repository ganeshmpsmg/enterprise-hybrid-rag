import logging
import traceback
from fastapi import APIRouter, HTTPException, UploadFile, File
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

# Global placeholders injected at runtime by main.py
_pipeline = None
_ingestion_service = None
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Receives file, reads bytes, and triggers the ingestion service.
    Removed doc_id as it is not supported by the current service signature.
    """
    global _ingestion_service
    
    if _ingestion_service is None:
        raise HTTPException(status_code=500, detail="Ingestion service not initialized.")

    try:
        content = await file.read()
        
        # Offload synchronous ingestion to threadpool
        # Note: doc_id argument removed to match ingest_bytes signature
        result = await run_in_threadpool(
            _ingestion_service.ingest_bytes,
            content=content,
            filename=file.filename,
            content_type=file.content_type
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
        raise HTTPException(status_code=500, detail="RAG pipeline is not initialized.")

    try:
        response = await run_in_threadpool(_pipeline.run, query=request.query)
        return response.to_dict() if hasattr(response, "to_dict") else response
    except Exception as e:
        logger.exception("Error processing /ask request")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ping")
async def ping():
    return {"status": "ok"}

@router.get("/debug")
async def debug():
    global _pipeline, _ingestion_service
    return {
        "pipeline_initialized": _pipeline is not None,
        "ingestion_initialized": _ingestion_service is not None,
    }