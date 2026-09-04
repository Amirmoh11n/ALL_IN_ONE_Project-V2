"""Integration tests for FastAPI health and predict (model optional)."""

from io import BytesIO

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient
from PIL import Image

from webapplication.backend.main import app

client = TestClient(app)


def test_health_endpoint_shape():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert "model_ready" in body
    assert "model_version" in body
    assert "classes" in body
    assert len(body["classes"]) == 4


def test_live_health():
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_model_info_has_disclaimer():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "disclaimer" in body
    assert body["model_version"]


def test_predict_rejects_unsupported_type():
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_without_model_is_service_unavailable_or_ok():
    image = Image.new("RGB", (64, 64), color=(12, 12, 12))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    response = client.post("/api/predict", files={"file": ("slice.png", buffer, "image/png")})
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        body = response.json()
        assert body["predicted_class"] in {"glioma", "meningioma", "notumor", "pituitary"}
        assert "model_version" in body
        assert len(body["probabilities"]) == 4
