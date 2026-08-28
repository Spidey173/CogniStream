"""Tests for WebSocket streaming, telemetry stats, and snapshot endpoints."""

import io
from PIL import Image


def test_websocket_connects_successfully(client):
    """WebSocket handshake with token authentication."""
    with client.websocket_connect("/api/v1/stream/cam1?api_key=dev-secret-api-key") as ws:
        assert ws is not None


def test_websocket_accepts_jpeg_frame(client, dummy_jpeg):
    """Sending a valid JPEG frame returns status 'ok' and face count."""
    with client.websocket_connect("/api/v1/stream/cam1?api_key=dev-secret-api-key") as ws:
        ws.send_bytes(dummy_jpeg)
        response = ws.receive_json()

        assert response["status"] == "ok"
        assert response["camera_id"] == "cam1"
        assert "faces" in response
        assert "ts" in response


def test_system_stats_endpoint(client):
    """GET /api/v1/stats returns system telemetry."""
    response = client.get("/api/v1/stats", headers={"X-API-Key": "dev-secret-api-key"})
    assert response.status_code == 200
    data = response.json()
    assert "active_cameras" in data
    assert "detector" in data


def test_cameras_endpoint(client):
    """GET /api/v1/cameras returns active cameras."""
    response = client.get("/api/v1/cameras", headers={"X-API-Key": "dev-secret-api-key"})
    assert response.status_code == 200
    assert "cameras" in response.json()


def test_detectors_endpoint(client):
    """GET /api/v1/detectors returns available plugin detector backends."""
    response = client.get("/api/v1/detectors", headers={"X-API-Key": "dev-secret-api-key"})
    assert response.status_code == 200
    assert "detectors" in response.json()
