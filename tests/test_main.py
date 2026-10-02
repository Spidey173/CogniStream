"""Tests for main.py application factory and exception handlers."""

import pytest
from fastapi import status


def test_main_app_creation(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK


def test_main_metrics_middleware(client):
    response = client.get("/health")
    assert response.status_code == 200
