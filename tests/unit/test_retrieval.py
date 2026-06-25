"""Tests for retrieval components: BM25, Dense, Hybrid, RRF."""
import pytest


# ── BM25 Tests ──────────────────────────────────────────────
class TestBM25Retriever:
    @pytest.fixture
    def bm25_with_data(self):
        from src.sparse_retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        corpus = [
            "transformer attention mechanism neural network deep learning",
            "gradient descent optimizer learning rate backpropagation",
            "convolutional neural network image classification features",
            "BERT language model pre-training fine-tuning NLP",
            "reinforcement learning reward policy agent environment",
        ]
        chunk_ids = [f"chunk_{i}" for i in range(len(corpus))]
        metadatas = [{"doc_id": f"doc_{i}", "file_name": f"paper_{i}.pdf"} for i in range(len(corpus))]
        bm25.fit(corpus, chunk_ids, metadatas)
        return bm25

    def test_search_returns_results(self, bm25_with_data):
        results = bm25_with_data.search("transformer attention", top_k=3)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "chunk_0"

    def test_search_top_k_respected(self, bm25_with_data):
        results = bm25_with_data.search("neural network learning", top_k=2)
        assert len(results) <= 2

    def test_search_returns_scores(self, bm25_with_data):
        results = bm25_with_data.search("BERT language model", top_k=5)
        for r in results:
            assert "score" in r
            assert r["score"] > 0

    def test_search_empty_query_returns_empty(self, bm25_with_data):
        results = bm25_with_data.search("", top_k=5)
        assert results == []

    def test_metadata_filter(self, bm25_with_data):
        results = bm25_with_data.search("neural network", top_k=5, filter_metadata={"doc_id": "doc_0"})
        assert all(r["doc_id"] == "doc_0" for r in results)

    def test_stats(self, bm25_with_data):
        stats = bm25_with_data.get_stats()
        assert stats["indexed"] is True
        assert stats["total_documents"] == 5


# ── RRF Tests ────────────────────────────────────────────────
class TestRRF:
    def test_basic_fusion(self):
        from src.hybrid_retrieval.rrf import reciprocal_rank_fusion
        dense = [
            {"chunk_id": "A", "score": 0.9, "content": "a"},
            {"chunk_id": "B", "score": 0.7, "content": "b"},
        ]
        sparse = [
            {"chunk_id": "B", "score": 25.0, "content": "b"},
            {"chunk_id": "C", "score": 20.0, "content": "c"},
        ]
        result = reciprocal_rank_fusion([dense, sparse], k=60)
        chunk_ids = [r["chunk_id"] for r in result]
        assert "B" in chunk_ids  # B appears in both -> higher score
        assert len(result) == 3

    def test_single_ranker(self):
        from src.hybrid_retrieval.rrf import reciprocal_rank_fusion
        results = [{"chunk_id": "X", "score": 1.0, "content": "x"}]
        fused = reciprocal_rank_fusion([results], k=60)
        assert len(fused) == 1
        assert fused[0]["chunk_id"] == "X"

    def test_scores_are_positive(self):
        from src.hybrid_retrieval.rrf import reciprocal_rank_fusion
        r1 = [{"chunk_id": f"d{i}", "score": float(10-i), "content": ""} for i in range(5)]
        r2 = [{"chunk_id": f"d{i}", "score": float(i), "content": ""} for i in range(5)]
        fused = reciprocal_rank_fusion([r1, r2])
        for item in fused:
            assert item["score"] > 0

    def test_rrf_k_parameter_effect(self):
        from src.hybrid_retrieval.rrf import reciprocal_rank_fusion
        results = [{"chunk_id": "A", "score": 1.0, "content": ""}]
        fused_k60 = reciprocal_rank_fusion([results], k=60)
        fused_k1 = reciprocal_rank_fusion([results], k=1)
        # Lower k -> higher score for top results
        assert fused_k1[0]["score"] > fused_k60[0]["score"]

    def test_empty_lists_handled(self):
        from src.hybrid_retrieval.rrf import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([[], []])
        assert result == []

    def test_weighted_fusion(self):
        from src.hybrid_retrieval.rrf import reciprocal_rank_fusion
        r1 = [{"chunk_id": "A", "score": 1.0, "content": ""}]
        r2 = [{"chunk_id": "B", "score": 1.0, "content": ""}]
        # With equal weights both should appear
        fused = reciprocal_rank_fusion([r1, r2], weights=[0.5, 0.5])
        assert len(fused) == 2


# ── Retrieval Metrics Tests ────────────────────────────────────
class TestRetrievalMetrics:
    def test_precision_at_k(self):
        from src.evaluation.retrieval_metrics import precision_at_k
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5
        assert precision_at_k(retrieved, relevant, k=2) == 0.5

    def test_recall_at_k(self):
        from src.evaluation.retrieval_metrics import recall_at_k
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c", "e"}
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2/3)

    def test_mrr_perfect(self):
        from src.evaluation.retrieval_metrics import mean_reciprocal_rank
        retrieved = [["a", "b", "c"]]
        relevant = [{"a"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 1.0

    def test_mrr_miss(self):
        from src.evaluation.retrieval_metrics import mean_reciprocal_rank
        retrieved = [["x", "y", "z"]]
        relevant = [{"a"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.0

    def test_ndcg_perfect(self):
        from src.evaluation.retrieval_metrics import ndcg_at_k
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == 1.0

    def test_ndcg_imperfect(self):
        from src.evaluation.retrieval_metrics import ndcg_at_k
        retrieved = ["x", "a", "b"]  # First result is irrelevant
        relevant = {"a", "b"}
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert 0 < ndcg < 1.0
