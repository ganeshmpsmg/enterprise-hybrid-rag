from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
import logging

# Define these as globals; they will be populated by main.py
_pipeline = None
logger = logging.getLogger(__name__)

# Replace this with the correct relative import if QueryRequest is in a models.py file
# e.g., from ..models import QueryRequest
from pydantic import BaseModel
class QueryRequest(BaseModel):
    query: str

router = APIRouter()

@router.post("/ask")
async def ask_question(request: QueryRequest):
    global _pipeline
    
    if _pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline is not initialized."
        )

    try:
        # Offload the synchronous RAG pipeline to a threadpool
        response = await run_in_threadpool(
            _pipeline.run,
            query=request.query,
        )

        # Return the dictionary format as required by your RAGResponse dataclass
        if hasattr(response, "to_dict"):
            return response.to_dict()
        
        return response

    except Exception:
        # Use logger.exception to capture the full traceback in your logs
        logger.exception("Error processing /ask request")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer."
        )