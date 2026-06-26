from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
import logging
from pydantic import BaseModel

# Global placeholder; main.py will inject the pipeline instance here
_pipeline = None
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

router = APIRouter()

@router.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Processes a RAG query. Uses run_in_threadpool to offload 
    the synchronous .run() call.
    """
    global _pipeline
    
    if _pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline is not initialized."
        )

    try:
        # Offload synchronous execution to threadpool to avoid blocking FastAPI
        response = await run_in_threadpool(
            _pipeline.run,
            query=request.query,
        )

        # Return the dictionary format as required by the RAGResponse dataclass
        if hasattr(response, "to_dict"):
            return response.to_dict()
        
        return response

    except Exception as e:
        # Log the full traceback for your backend logs
        logger.exception("Error processing /ask request")
        
        # Return the specific error to the client for debugging
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )