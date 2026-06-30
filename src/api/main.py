"""
FastAPI Application Entry Point - Enterprise Hybrid RAG System.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from src.api.routes import router
import src.api.routes as route_module

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Application factory ──────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize all pipeline components on startup."""
        logger.info("Starting Enterprise Hybrid RAG System...")
        t0 = time.time()

        route_module._pipeline = None
        route_module._ingestion_service = None
        route_module._initialization_error = None

        try:
            await _initialize_pipeline()
        except Exception as exc:
            route_module._initialization_error = str(exc)
            logger.exception("Pipeline initialization failed; starting in degraded mode")

        elapsed = time.time() - t0
        logger.info(f"Pipeline initialization completed in {elapsed:.2f}s")
        yield  # Application runs here
        logger.info("Shutting down RAG system...")

    app = FastAPI(
        title="Enterprise Hybrid RAG API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
    )

    # ── Middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=int(os.getenv("RATE_LIMIT_MAX", "100")),
        window_seconds=60,
    )

    # ── Routes ──────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "status": "ok",
            "message": "Enterprise Hybrid RAG backend is running. Use /api/v1/health or /docs.",
        }

    # ── Global Exception Handler ────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


async def _initialize_pipeline():
    """Wire up all pipeline dependencies."""
    from src.embeddings.embedding_model import EmbeddingModel
    from src.embeddings.embedding_pipeline import EmbeddingPipeline
    from src.vectorstore.chroma_manager import ChromaManager
    from src.vectorstore.faiss_manager import FAISSManager
    from src.dense_retrieval.dense_retriever import DenseRetriever
    from src.sparse_retrieval.bm25_retriever import BM25Retriever
    from src.hybrid_retrieval.hybrid_retriever import HybridRetriever
    from src.hybrid_retrieval.query_expander import QueryExpander
    from src.reranker.reranker import Reranker
    from src.reranker.ranking_pipeline import RankingPipeline
    from src.llm.llm_service import LLMService
    from src.llm.answer_generator import AnswerGenerator
    from src.llm.rag_pipeline import RAGPipeline
    from src.vectorstore.index_builder import IndexBuilder
    from src.utils.ingestion_service import IngestionService

    # 1. Initialize Core Models
    embed_model = EmbeddingModel(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    embed_pipeline = EmbeddingPipeline(model=embed_model)

    # 2. Initialize Vector Store
    v_type = os.getenv("VECTOR_STORE", "chroma")
    if v_type == "faiss":
        vector_store = FAISSManager(dimension=384)
    else:
        vector_store = ChromaManager(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "data/embeddings/chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION", "ml_documents"),
        )
    
    # 3. Initialize Retrievers
    sparse_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever(
        dense_retriever=DenseRetriever(vector_store=vector_store, embedding_pipeline=embed_pipeline),
        sparse_retriever=sparse_retriever,
        dense_top_k=int(os.getenv("DENSE_TOP_K", "20")),
        sparse_top_k=int(os.getenv("SPARSE_TOP_K", "20")),
        rrf_k=int(os.getenv("RRF_K", "60")),
    )

    # 4. Initialize RAG Pipeline
    rag_pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        ranking_pipeline=RankingPipeline(
            hybrid_retriever=hybrid_retriever,
            reranker=Reranker(model_name=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"), top_k=5)
        ),
        answer_generator=AnswerGenerator(llm_service=LLMService(provider=os.getenv("LLM_PROVIDER", "openai"))),
        query_expander=QueryExpander(),
    )

    # 5. Inject dependencies into route module
    route_module._vector_store = vector_store
    route_module._pipeline = rag_pipeline
    route_module._ingestion_service = IngestionService(index_builder=IndexBuilder(vector_store=vector_store, embedding_pipeline=embed_pipeline), sparse_retriever=sparse_retriever)
    
    logger.info("All pipeline components initialized successfully")

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
