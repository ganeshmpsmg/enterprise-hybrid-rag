"""
RAG Pipeline - Complete end-to-end Retrieval-Augmented Generation pipeline.

Full pipeline:
Query -> Expansion -> Hybrid Retrieval -> Reranking -> Context Building -> LLM -> Answer
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.hybrid_retrieval.hybrid_retriever import HybridRetriever
from src.hybrid_retrieval.query_expander import QueryExpander
from src.llm.answer_generator import AnswerGenerator, RAGAnswer
from src.reranker.ranking_pipeline import RankingPipeline

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG system response with full pipeline metadata."""
    query: str
    answer: str
    citations: list[dict]
    context_chunks_used: int
    pipeline_metadata: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "context_chunks_used": self.context_chunks_used,
            "pipeline_metadata": self.pipeline_metadata,
            "total_latency_ms": self.total_latency_ms,
        }


class RAGPipeline:
    """
    Enterprise RAG Pipeline.

    Stages:
    1. Query Expansion   - Generate query variants for better recall
    2. Hybrid Retrieval  - Dense + Sparse + RRF fusion
    3. Re-ranking        - Cross-encoder precise scoring
    4. Context Building  - Deduplicate, order, trim
    5. LLM Generation   - GPT/Llama/Mistral answer generation

    Configuration:
    - All stages are configurable and optional
    - Fallback to simpler retrieval if stages fail
    - Full pipeline latency and step-level timing tracked
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        ranking_pipeline: RankingPipeline,
        answer_generator: AnswerGenerator,
        query_expander: Optional[QueryExpander] = None,
        use_query_expansion: bool = True,
        hybrid_top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.ranking_pipeline = ranking_pipeline
        self.answer_generator = answer_generator
        self.query_expander = query_expander or QueryExpander()
        self.use_query_expansion = use_query_expansion
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k
        self._pipeline_calls = 0

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
        conversation_history: Optional[list[dict]] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            query: User's natural language question
            top_k: Number of final results to use for generation
            filter_metadata: Optional metadata filter (e.g., filter by document)
            conversation_history: Optional previous conversation turns

        Returns:
            RAGResponse with answer, citations, and pipeline metadata
        """
        pipeline_start = time.perf_counter()
        timings = {}
        final_k = top_k or self.rerank_top_k

        # ── Stage 1: Query Expansion ──────────────────
        t = time.perf_counter()
        if self.use_query_expansion:
            expanded_queries = self.query_expander.expand(query)
            primary_query = expanded_queries[0]  # Use original as primary
            logger.debug(f"Query expanded: {len(expanded_queries)} variants")
        else:
            primary_query = query
            expanded_queries = [query]
        timings["query_expansion_ms"] = round((time.perf_counter() - t) * 1000, 2)

        # ── Stage 2: Hybrid Retrieval ─────────────────
        t = time.perf_counter()
        try:
            hybrid_results = self.hybrid_retriever.retrieve(
                query=primary_query,
                top_k=self.hybrid_top_k,
                filter_metadata=filter_metadata,
            )
        except Exception as e:
            
            import traceback
            traceback.print_exc()

            logger.error(f"Hybrid retrieval failed: {repr(e)}")

            raise
        timings["hybrid_retrieval_ms"] = round((time.perf_counter() - t) * 1000, 2)

        if not hybrid_results:
            return RAGResponse(
                query=query,
                answer="No relevant documents found for your query.",
                citations=[],
                context_chunks_used=0,
                pipeline_metadata={"timings": timings, "stage_failed": "retrieval"},
                total_latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 2),
            )

        # ── Stage 3: Reranking ────────────────────────
        t = time.perf_counter()
        try:
            candidates = [r.to_dict() for r in hybrid_results]
            reranked = self.ranking_pipeline.reranker.rerank(
                query=primary_query,
                candidates=candidates,
                top_k=final_k,
            )
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using hybrid results.")
            reranked = candidates[:final_k]
        timings["reranking_ms"] = round((time.perf_counter() - t) * 1000, 2)

        # ── Stage 4 & 5: Context + LLM Generation ────
        t = time.perf_counter()
        try:
            rag_answer: RAGAnswer = self.answer_generator.generate(
                query=query,
                ranked_results=reranked,
                conversation_history=conversation_history,
            )
        except Exception as e:
            
            import traceback
            traceback.print_exc()

            logger.error(f"Answer generation failed: {repr(e)}")

            raise
        timings["generation_ms"] = round((time.perf_counter() - t) * 1000, 2)

        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        self._pipeline_calls += 1

        return RAGResponse(
            query=query,
            answer=rag_answer.answer,
            citations=rag_answer.citations,
            context_chunks_used=rag_answer.context_chunks_used,
            pipeline_metadata={
                "timings": timings,
                "expanded_queries": expanded_queries,
                "hybrid_results_count": len(hybrid_results),
                "reranked_count": len(reranked),
                "model": rag_answer.model,
            },
            total_latency_ms=total_ms,
        )

    def _empty_response(self, query: str, error: str) -> RAGResponse:
        return RAGResponse(
            query=query,
            answer=f"An error occurred while processing your query. Please try again.",
            citations=[],
            context_chunks_used=0,
            pipeline_metadata={"error": error},
            total_latency_ms=0.0,
        )

    def get_stats(self) -> dict:
        return {
            "pipeline_calls": self._pipeline_calls,
            "hybrid_retriever": self.hybrid_retriever.get_stats(),
            "answer_generator": self.answer_generator.get_stats(),
        }
