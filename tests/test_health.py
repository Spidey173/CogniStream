"""Tests for /health, /ready, and /metrics endpoints."""


def test_health_returns_200(client):
    """GET /health should return 200 with status 'healthy'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_ready_endpoint(client):
    """GET /ready should return 200 with DB & Redis health status."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "database" in data
    assert "redis" in data


def test_metrics_endpoint(client):
    """GET /metrics should return Prometheus metrics output."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "frames_processed_total" in response.text
