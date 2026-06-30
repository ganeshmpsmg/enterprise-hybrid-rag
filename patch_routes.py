from pathlib import Path
p = Path('src/api/routes.py')
s = p.read_text(encoding='utf-8')
old_imports = "import logging\nimport os\nfrom fastapi import APIRouter, HTTPException, UploadFile, File\nfrom starlette.concurrency import run_in_threadpool\nfrom pydantic import BaseModel\n"
new_imports = "import logging\nimport os\nimport asyncio\nfrom fastapi import APIRouter, HTTPException, UploadFile, File\nfrom starlette.concurrency import run_in_threadpool\nfrom pydantic import BaseModel\n"
if old_imports in s:
    s = s.replace(old_imports, new_imports)

# insert ASK_TIMEOUT after router = APIRouter()
if 'router = APIRouter()' in s and 'ASK_TIMEOUT' not in s:
    s = s.replace('router = APIRouter()\n', 'router = APIRouter()\nASK_TIMEOUT = int(os.getenv("ASK_TIMEOUT", "60"))\n')

# replace ask_question function body
start_marker = '\n@router.post("/ask")\nasync def ask_question(request: QueryRequest):'
if start_marker in s:
    start_idx = s.index(start_marker)
    # find the next decorator for /health which marks end of function
    end_marker = '\n@router.get("/health", response_model=HealthResponse)'
    end_idx = s.index(end_marker)
    new_func = '''\n@router.post("/ask")\nasync def ask_question(request: QueryRequest):\n    """Processes a RAG query using the synchronous pipeline in a threadpool."""\n    global _pipeline\n    \n    if _pipeline is None:\n        raise HTTPException(\n            status_code=503,\n            detail=_initialization_error or "RAG pipeline is not initialized.",\n        )\n\n    try:\n        fut = run_in_threadpool(_pipeline.run, query=request.query)\n        response = await asyncio.wait_for(fut, timeout=ASK_TIMEOUT)\n        return response.to_dict() if hasattr(response, "to_dict") else response\n    except Exception as e:\n        logger.exception("Error processing /ask request")\n        if isinstance(e, asyncio.TimeoutError):\n            raise HTTPException(status_code=504, detail=f"RAG pipeline timed out after {ASK_TIMEOUT}s")\n        raise HTTPException(status_code=500, detail=str(e))\n'''
    s = s[:start_idx] + new_func + s[end_idx:]
    p.write_text(s, encoding='utf-8')
    print('patched')
else:
    print('ask marker not found')
