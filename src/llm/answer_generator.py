"""
Answer Generator - Orchestrates LLM answer generation for the RAG pipeline.
Takes context and query, returns structured answer with citations.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.llm.context_builder import ContextBuilder
from src.llm.llm_service import LLMService
from src.llm.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Structured answer from the RAG pipeline."""
    query: str
    answer: str
    citations: list[dict]
    context_chunks_used: int
    generation_time_ms: float
    model: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "context_chunks_used": self.context_chunks_used,
            "generation_time_ms": self.generation_time_ms,
            "model": self.model,
            "metadata": self.metadata,
        }


class AnswerGenerator:
    """
    Generates answers using LLM with retrieved context.

    Pipeline:
    ranked_results -> ContextBuilder -> PromptBuilder -> LLMService -> RAGAnswer
    """

    def __init__(
        self,
        llm_service: LLMService,
        prompt_builder: Optional[PromptBuilder] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.llm = llm_service
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.context_builder = context_builder or ContextBuilder()
        self._total_answers = 0
        self._total_time = 0.0

    def generate(
        self,
        query: str,
        ranked_results: list[dict],
        conversation_history: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> RAGAnswer:
        """
        Generate an answer from ranked retrieval results.

        Args:
            query: User's question
            ranked_results: Ranked retrieval results (each with 'content' and 'metadata')
            conversation_history: Optional previous conversation turns
            max_tokens: Override default token limit

        Returns:
            RAGAnswer with answer text and citations
        """
        t0 = time.perf_counter()

        # 1. Build context
        context_chunks, citations = self.context_builder.build(ranked_results)

        if not context_chunks:
            return RAGAnswer(
                query=query,
                answer="I could not find relevant information in the provided documents to answer your question.",
                citations=[],
                context_chunks_used=0,
                generation_time_ms=0.0,
                model=self.llm.provider,
            )

        # 2. Build prompt
        if conversation_history:
            messages = self.prompt_builder.build_conversational(
                query, context_chunks, conversation_history
            )
        else:
            prompt = self.prompt_builder.build(query, context_chunks)
            messages = prompt["messages"]

        # 3. Generate answer
        answer_text = self.llm.generate(messages=messages, max_tokens=max_tokens)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_answers += 1
        self._total_time += elapsed_ms

        logger.info(
            f"Generated answer for '{query[:50]}...' | "
            f"context={len(context_chunks)} chunks | "
            f"{elapsed_ms:.1f}ms"
        )

        return RAGAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            context_chunks_used=len(context_chunks),
            generation_time_ms=round(elapsed_ms, 2),
            model=self.llm.provider,
            metadata={
                "context_chars": sum(len(c.get("content", "")) for c in context_chunks),
                "top_score": ranked_results[0].get("final_score", ranked_results[0].get("score", 0)) if ranked_results else 0,
            },
        )

    def get_stats(self) -> dict:
        return {
            "total_answers": self._total_answers,
            "avg_generation_ms": round(self._total_time / max(self._total_answers, 1), 2),
            **self.llm.get_stats(),
        }
