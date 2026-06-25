"""
Grafana Dashboard - Auto-generates Grafana dashboard JSON for RAG monitoring.
"""

import json
from pathlib import Path


def generate_dashboard() -> dict:
    """Generate Grafana dashboard configuration for RAG system monitoring."""
    return {
        "id": None,
        "title": "Enterprise RAG System Dashboard",
        "tags": ["rag", "ml", "production"],
        "timezone": "browser",
        "refresh": "10s",
        "panels": [
            {
                "id": 1,
                "title": "Request Rate (req/s)",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": "rate(enterprise_rag_requests_total[1m])",
                        "legendFormat": "{{endpoint}}",
                    }
                ],
            },
            {
                "id": 2,
                "title": "P95 Latency (ms)",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(enterprise_rag_request_duration_seconds_bucket[5m])) * 1000",
                        "legendFormat": "P95 {{endpoint}}",
                    }
                ],
            },
            {
                "id": 3,
                "title": "Total Indexed Documents",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
                "targets": [{"expr": "enterprise_rag_documents_total"}],
            },
            {
                "id": 4,
                "title": "Total Indexed Chunks",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
                "targets": [{"expr": "enterprise_rag_chunks_total"}],
            },
            {
                "id": 5,
                "title": "Pipeline Stage Latency",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(enterprise_rag_query_stage_duration_seconds_bucket[5m])) * 1000",
                        "legendFormat": "{{stage}} P95",
                    }
                ],
            },
            {
                "id": 6,
                "title": "Error Rate",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                "targets": [
                    {
                        "expr": "rate(enterprise_rag_errors_total[5m])",
                        "legendFormat": "{{error_type}}",
                    }
                ],
            },
        ],
    }


def save_dashboard(output_path: str = "deployment/grafana/dashboard.json"):
    """Save Grafana dashboard JSON to file."""
    dashboard = generate_dashboard()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"dashboard": dashboard, "overwrite": True}, f, indent=2)
    print(f"Grafana dashboard saved to {output_path}")
