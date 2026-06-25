"""
Prometheus Metrics - Exposes RAG system metrics for Prometheus scraping.
"""
import logging

logger = logging.getLogger(__name__)

# Try importing prometheus_client
try:
    # Removed unused 'Summary' import to fix F401 error
    from prometheus_client import (
        Counter, Gauge, Histogram,
        REGISTRY, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client not installed. Metrics disabled.")


class RAGMetrics:
    """
    Prometheus metrics for the RAG system.
    """

    def __init__(self, app_name: str = "enterprise_rag"):
        self.app_name = app_name
        self._initialized = False
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()

    def _init_metrics(self):
        prefix = self.app_name

        self.request_count = Counter(
            f"{prefix}_requests_total",
            "Total API requests",
            ["endpoint", "method", "status_code"],
        )
        self.request_duration = Histogram(
            f"{prefix}_request_duration_seconds",
            "API request duration in seconds",
            ["endpoint"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        )
        self.documents_indexed = Gauge(
            f"{prefix}_documents_total",
            "Total documents indexed",
        )
        self.chunks_indexed = Gauge(
            f"{prefix}_chunks_total",
            "Total chunks indexed",
        )
        self.query_stage_duration = Histogram(
            f"{prefix}_query_stage_duration_seconds",
            "Duration of each pipeline stage",
            ["stage"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        )
        self.retrieval_results = Histogram(
            f"{prefix}_retrieval_results_count",
            "Number of retrieval results per query",
            buckets=[0, 1, 3, 5, 10, 20, 50],
        )
        self.llm_tokens = Counter(
            f"{prefix}_llm_tokens_total",
            "Total LLM tokens",
            ["direction"],  # input | output
        )
        self.embedding_duration = Histogram(
            f"{prefix}_embedding_duration_seconds",
            "Embedding generation duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        )
        self.errors_total = Counter(
            f"{prefix}_errors_total",
            "Total errors by type",
            ["error_type"],
        )
        self._initialized = True

    def record_request(self, endpoint: str, method: str, status_code: int, duration: float):
        if self._initialized:
            self.request_count.labels(endpoint=endpoint, method=method, status_code=str(status_code)).inc()
            self.request_duration.labels(endpoint=endpoint).observe(duration)

    def record_retrieval(self, stage: str, duration: float, result_count: int = 0):
        if self._initialized:
            self.query_stage_duration.labels(stage=stage).observe(duration)
            if result_count > 0:
                self.retrieval_results.observe(result_count)

    def record_embedding(self, duration: float):
        if self._initialized:
            self.embedding_duration.observe(duration)

    def record_error(self, error_type: str):
        if self._initialized:
            self.errors_total.labels(error_type=error_type).inc()

    def set_indexed_counts(self, documents: int, chunks: int):
        if self._initialized:
            self.documents_indexed.set(documents)
            self.chunks_indexed.set(chunks)

    def get_metrics_output(self) -> tuple[bytes, str]:
        """Return Prometheus metrics in text format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
        return b"# Prometheus not available\n", "text/plain"


# Global metrics instance
metrics = RAGMetrics()
