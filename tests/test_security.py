"""Unit tests for security and API key authentication."""

from app.core.security import verify_api_key


def test_verify_api_key():
    assert verify_api_key("dev-secret-api-key") is True
    assert verify_api_key(None) is False
