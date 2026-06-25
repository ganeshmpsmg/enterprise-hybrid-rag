"""Tests for RAG pipeline components."""


class TestPromptBuilder:
    def test_build_prompt(self):
        from src.llm.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        chunks = [
            {
                "content": "Transformers use attention mechanisms.",
                "metadata": {"file_name": "paper.pdf"},
            },
            {
                "content": "BERT is bidirectional.",
                "metadata": {"file_name": "bert.pdf", "page_number": 3},
            },
        ]
        result = builder.build("What is attention?", chunks)
        assert "system" in result
        assert "user" in result
        assert "messages" in result
        assert "What is attention?" in result["user"]
        assert "[Source 1]" in result["user"]
        assert "[Source 2]" in result["user"]

    def test_context_truncation(self):
        from src.llm.prompt_builder import PromptBuilder, PromptConfig

        config = PromptConfig(system_prompt="You are helpful.", max_context_length=100)
        builder = PromptBuilder(config=config)
        chunks = [
            {"content": "x" * 200, "metadata": {}},
            {"content": "y" * 200, "metadata": {}},  # Should be truncated
        ]
        result = builder.build("query", chunks)
        # Should only include first chunk
        assert "[Source 2]" not in result["user"]


class TestContextBuilder:
    def test_deduplication(self):
        from src.llm.context_builder import ContextBuilder

        builder = ContextBuilder()
        chunks = [
            {
                "chunk_id": "1",
                "content": "Same content here is repeated.",
                "score": 0.9,
                "metadata": {},
            },
            {
                "chunk_id": "2",
                "content": "Same content here is repeated.",
                "score": 0.8,
                "metadata": {},
            },
        ]
        selected, _ = builder.build(chunks)
        assert len(selected) == 1

    def test_budget_respected(self):
        from src.llm.context_builder import ContextBuilder

        builder = ContextBuilder(max_context_chars=100)
        chunks = [
            {
                "chunk_id": f"{i}",
                "content": "x" * 80,
                "score": float(1 / i),
                "metadata": {},
            }
            for i in range(1, 5)
        ]
        selected, _ = builder.build(chunks)
        total = sum(len(c["content"]) for c in selected)
        assert total <= 200  # With buffer

    def test_citations_extracted(self):
        from src.llm.context_builder import ContextBuilder

        builder = ContextBuilder()
        chunks = [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "content": "text",
                "score": 0.9,
                "metadata": {"file_name": "paper.pdf", "page_number": 5},
            },
        ]
        _, citations = builder.build(chunks)
        assert len(citations) == 1
        assert citations[0]["file_name"] == "paper.pdf"
        assert citations[0]["page_number"] == 5


class TestQueryExpander:
    def test_returns_original_query(self):
        from src.hybrid_retrieval.query_expander import QueryExpander

        expander = QueryExpander()
        results = expander.expand("neural networks")
        assert "neural networks" in results

    def test_acronym_expansion(self):
        from src.hybrid_retrieval.query_expander import QueryExpander

        expander = QueryExpander(use_acronym_expansion=True)
        results = expander.expand("What is BERT?")
        # One of the results should have expanded acronym
        assert any(
            "bidirectional" in r.lower() or "encoder" in r.lower() for r in results
        )

    def test_hybrid_expansion_format(self):
        from src.hybrid_retrieval.query_expander import QueryExpander

        expander = QueryExpander()
        result = expander.expand_for_hybrid("transformer attention")
        assert "dense" in result
        assert "sparse" in result
