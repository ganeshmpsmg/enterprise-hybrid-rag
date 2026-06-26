import logging
import traceback
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

# Global placeholders injected at runtime by main.py
_pipeline = None
_ingestion_service = None
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

router = APIRouter()

@router.post("/ask")
async def ask_question(request: QueryRequest):
    """Processes a RAG query using the synchronous pipeline in a threadpool."""
    global _pipeline
    
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline is not initialized.")

    try:
        # Offload synchronous execution to avoid blocking the event loop
        response = await run_in_threadpool(_pipeline.run, query=request.query)
        return response.to_dict() if hasattr(response, "to_dict") else response
    except Exception as e:
        logger.exception("Error processing /ask request")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ping")
async def ping():
    """Simple health check endpoint."""
    return {"status": "ok"}

@router.get("/debug")
async def debug():
    """Diagnostic endpoint to inspect startup status."""
    global _pipeline, _ingestion_service
    try:
        return {
            "pipeline_initialized": _pipeline is not None,
            "ingestion_initialized": _ingestion_service is not None,
        }
    except Exception:
        return {"traceback": traceback.for