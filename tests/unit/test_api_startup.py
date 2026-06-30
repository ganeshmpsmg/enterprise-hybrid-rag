import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


@pytest.mark.unit
def test_health_returns_degraded_when_pipeline_initialization_fails(monkeypatch):
    async def broken_initialization():
        raise RuntimeError("simulated startup failure")

    monkeypatch.setattr(api_main, "_initialize_pipeline", broken_initialization)

    app = api_main.create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["vector_store_status"] == "uninitialized"
