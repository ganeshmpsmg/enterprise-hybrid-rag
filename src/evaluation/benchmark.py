"""
Benchmark - Full RAG system benchmarking with configurable test sets.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.ragas_evaluation import RAGASEvaluator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    retrieval_metrics: dict
    ragas_metrics: dict
    latency_stats: dict
    total_queries: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> dict:
        return {
            "retrieval_metrics": self.retrieval_metrics,
            "ragas_metrics": self.ragas_metrics,
            "latency_stats": self.latency_stats,
            "total_queries": self.total_queries,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        r = self.retrieval_metrics
        return (
            f"Benchmark Results ({self.total_queries} queries)\n"
            f"{'='*50}\n"
            f"Retrieval:\n"
            f"  Precision@5:  {r.get('precision@5', 0):.3f}\n"
            f"  Recall@5:     {r.get('recall@5', 0):.3f}\n"
            f"  MRR:          {r.get('mrr', 0):.3f}\n"
            f"  nDCG@10:      {r.get('ndcg@10', 0):.3f}\n"
            f"Generation:\n"
            f"  Faithfulness: {self.ragas_metrics.get('faithfulness', 0):.3f}\n"
            f"  Relevancy:    {self.ragas_metrics.get('answer_relevancy', 0):.3f}\n"
            f"Latency:\n"
            f"  P50: {self.latency_stats.get('p50_ms', 0):.0f}ms\n"
            f"  P95: {self.latency_stats.get('p95_ms', 0):.0f}ms\n"
        )


class RAGBenchmark:
    """
    Benchmark a complete RAG pipeline on a test dataset.

    Test dataset format:
    [
        {
            "question": "What is transformer attention?",
            "relevant_doc_ids": ["doc1_chunk3", "doc2_chunk7"],
            "ground_truth": "Transformer attention is..."
        }
    ]
    """

    def __init__(self, rag_pipeline, ragas_evaluator: Optional[RAGASEvaluator] = None):
        self.pipeline = rag_pipeline
        self.ragas = ragas_evaluator or RAGASEvaluator()

    def run(self, test_data: list[dict], max_samples: Optional[int] = None) -> BenchmarkResult:
        """Run full benchmark."""
        samples = test_data[:max_samples] if max_samples else test_data
        logger.info(f"Running benchmark on {len(samples)} samples...")

        questions, answers, contexts = [], [], []
        ground_truths, retrieved_ids, relevant_ids = [], [], []
        latencies = []

        for sample in samples:
            q = sample["question"]
            t0 = time.perf_counter()
            try:
                response = self.pipeline.run(query=q)
                latencies.append((time.perf_counter() - t0) * 1000)
                questions.append(q)
                answers.append(response.answer)
                contexts.append([c.get("content", "") for c in response.citations])
                ground_truths.append(sample.get("ground_truth", ""))
                retrieved_ids.append([c.get("chunk_id", "") for c in response.citations])
                relevant_ids.append(set(sample.get("relevant_doc_ids", [])))
            except Exception as e:
                logger.warning(f"Sample failed: {e}")

        import numpy as np
        retrieval = evaluate_retrieval(questions, retrieved_ids, relevant_ids)
        ragas_result = self.ragas.evaluate(questions, answers, contexts, ground_truths or None)

        latency_arr = np.array(latencies) if latencies else np.array([0])
        latency_stats = {
            "mean_ms": round(float(np.mean(latency_arr)), 2),
            "p50_ms": round(float(np.percentile(latency_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(latency_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(latency_arr, 99)), 2),
        }

        result = BenchmarkResult(
            retrieval_metrics=retrieval,
            ragas_metrics=ragas_result.to_dict(),
            latency_stats=latency_stats,
            total_queries=len(samples),
        )
        logger.info(f"\n{result.summary()}")
        return result

    def save_results(self, result: BenchmarkResult, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Benchmark results saved to {path}")
