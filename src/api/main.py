"""
FastAPI Application Entry Point - Enterprise Hybrid RAG System.
Handles startup, dependency wiring, and application lifecycle.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from src.api.routes import router
import src.api.routes as route_module

load_dotenv()

logger = logging.getLogger(__name__)


# ── Application factory ──────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize all pipeline components on startup."""
        logger.info("Starting Enterprise Hybrid RAG System...")
        t0 = time.time()

        try:
            await _initialize_pipeline()
            elapsed = time.time() - t0
            logger.info(f"Pipeline initialized in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            # Don't crash - API will return 503 for pipeline-dependent endpoints

        yield  # Application runs here

        logger.info("Shutting down RAG system...")

    app = FastAPI(
        title="Enterprise Hybrid RAG API",
        description="""
Production-grade Hybrid RAG Search System for Machine Learning Documents.

## Features
- **Hybrid Retrieval**: Dense (semantic) + Sparse (BM25) + RRF fusion
- **Cross-Encoder Reranking**: ms-marco-MiniLM-L-6-v2
- **Query Expansion**: Synonym and LLM-based query expansion
- **Multi-format Ingestion**: PDF, DOCX, TXT, Markdown
- **Multi-vector Store**: FAISS, ChromaDB, Qdrant

## Endpoints
- `POST /upload` - Ingest documents
- `POST /search` - Hybrid search (no LLM)
- `POST /retrieve` - Raw retrieval
- `POST /ask` - Full RAG Q&A
- `GET /health` - Health check
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (order matters - outermost first) ──
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

    # Convenience aliases without prefix
    app.include_router(router)

    # ── Exception handlers ────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


async def _initialize_pipeline():
    """
    Wire up all pipeline dependencies.
    Uses environment variables for configuration.
    """
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

    # Configuration from environment
    embed_model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    vector_store_type = os.getenv("VECTOR_STORE", "chroma")
    llm_provider = os.getenv("LLM_PROVIDER", "openai")

    # ── Embedding model ─────────────────────────
    logger.info(f"Loading embedding model: {embed_model_name}")
    embed_model = EmbeddingModel(model_name=embed_model_name)
    embed_pipeline = EmbeddingPipeline(model=embed_model)

    # ── Vector store ────────────────────────────
    logger.info(f"Initializing vector store: {vector_store_type}")
    if vector_store_type == "faiss":
        vector_store = FAISSManager(dimension=384)
    elif vector_store_type == "qdrant":
        from src.vectorstore.qdrant_manager import QdrantManager

        vector_store = QdrantManager(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            dimension=384,
        )
    else:  # chroma (default)
        vector_store = ChromaManager(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "data/embeddings/chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION", "ml_documents"),
        )

    route_module._vector_store = vector_store

    # ── Retrievers ─────────────────────────────
    dense_retriever = DenseRetriever(
        vector_store=vector_store,
        embedding_pipeline=embed_pipeline,
    )
    sparse_retriever = BM25Retriever()

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_top_k=int(os.getenv("DENSE_TOP_K", "20")),
        sparse_top_k=int(os.getenv("SPARSE_TOP_K", "20")),
        rrf_k=int(os.getenv("RRF_K", "60")),
    )

    # ── Reranker ───────────────────────────────
    reranker = Reranker(model_name=reranker_model, top_k=5)
    ranking_pipeline = RankingPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    # ── LLM + Answer generator ─────────────────
    llm_service = LLMService(provider=llm_provider)
    answer_generator = AnswerGenerator(llm_service=llm_service)

    # ── Full RAG pipeline ──────────────────────
    rag_pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        ranking_pipeline=ranking_pipeline,
        answer_generator=answer_generator,
        query_expander=QueryExpander(),
    )

    # ── Ingestion service ─────────────────────
    index_builder = IndexBuilder(
        vector_store=vector_store,
        embedding_pipeline=embed_pipeline,
    )

    from src.utils.ingestion_service import IngestionService

    ingestion_svc = IngestionService(
        index_builder=index_builder,
        sparse_retriever=sparse_retriever,
    )

    # ── Inject into route module ───────────────
    route_module._pipeline = rag_pipeline
    route_module._ingestion_service = ingestion_svc
    route_module._app_start_time = time.time()

    logger.info("All pipeline components initialized successfully")


# Create the app instance
app = create_app()


def main():
    """Entry point for running the server."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=int(os.getenv("API_WORKERS", "1")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
