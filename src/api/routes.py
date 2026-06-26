import logging
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel

# These placeholders are populated by main.py at runtime via dependency injection
_ingestion_service = None
_pipeline = None

router = APIRouter()
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    """
    Handles PDF uploads. 
    Reads file into memory and offloads ingestion to BackgroundTasks 
    to prevent 502 Gateway Timeouts.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    if _ingestion_service is None:
        raise HTTPException(status_code=500, detail="Ingestion service is not initialized.")

    # 1. Read the file bytes while the request is still active.
    # This prevents the 'UploadFile' from closing before the background task runs.
    content = await file.read()
    
    # 2. Offload the heavy embedding/indexing work to the background.
    # Change 'ingest_bytes' to 'process' if that is the actual method name in your class.
    background_tasks.add_task(_ingestion_service.ingest_bytes, content, file.filename)
    
    logger.info(f"Ingestion task queued for file: {file.filename}")
    
    return {
        "status": "Accepted",
        "message": f"File {file.filename} is being processed in the background.",
        "filename": file.filename
    }

@router.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Handles user questions using the initialized RAG pipeline.
    Matches the /api/v1/ask endpoint expected by your frontend.
    """
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline is not initialized.")
    
    try:
        # Assuming your RAGPipeline has a method named 'generate' or similar
        response = await _pipeline.generate(request.query)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Simple status check for debugging deployment."""
    return {
        "status": "online", 
        "services_initialized": {
            "ingestion": _ingestion_service is not None, 
            "pipeline": _pipeline is not None
        }
    }