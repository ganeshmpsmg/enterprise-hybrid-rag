"""
RAG Pipeline - Complete end-to-end Retrieval-Augmented Generation pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

# Importing necessary components
from src.hybrid_retrieval.hybrid_retriever import HybridRetriever
from src.hybrid_retrieval.query_expander import QueryExpander
from src.llm.answer_generator import AnswerGenerator
from src.reranker.ranking_pipeline import RankingPipeline

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    answer: str
    citations: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "answer": self.answer,
            "citations": self.citations,
        }

class RAGPipeline:
    def __init__(
        self, 
        hybrid_retriever: HybridRetriever, 
        ranking_pipeline: RankingPipeline, 
        answer_generator: AnswerGenerator, 
        query_expander: QueryExpander
    ):
        self.hybrid_retriever = hybrid_retriever
        self.ranking_pipeline = ranking_pipeline
        self.answer_generator = answer_generator
        self.query_expander = query_expander

    def run(self, query: str) -> RAGResponse:
        """
        Executes the pipeline and returns a RAGResponse object.
        """
        # 1. Expand query
        expanded_queries = self.query_expander.expand(query)

        primary_query = expanded_queries[0] if expanded_queries else query

        # 2. Retrieve and rerank
        
        ranked_results = self.ranking_pipeline.retrieve_and_rerank(
            query=primary_query,
            top_k=5,
        )

        # Convert RankedResult objects to dictionaries
        ranked_results = [r.to_dict() for r in ranked_results]

        # 3. Generate answer
        rag_answer = self.answer_generator.generate(
            query=primary_query,
            ranked_results=ranked_results,
        )
        
        # 4. Return as RAGResponse
        return RAGResponse(
            answer=rag_answer.answer,
            citations=rag_answer.citations
        )