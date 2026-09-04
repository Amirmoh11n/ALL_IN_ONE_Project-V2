"""API schema contract tests (no live model)."""

from webapplication.backend.schemas import HealthResponse, PredictionResponse


def test_health_response_requires_version():
    payload = HealthResponse(
        status="model_not_ready",
        model_ready=False,
        model_format="ONNX",
        model_path="missing.onnx",
        model_version="2.0.0",
        classes=["glioma", "meningioma", "notumor", "pituitary"],
        provider="unavailable",
        error="not found",
    )
    dumped = payload.model_dump()
    assert dumped["model_version"] == "2.0.0"
    assert dumped["model_ready"] is False


def test_prediction_response_four_classes():
    payload = PredictionResponse(
        predicted_class="glioma",
        confidence=0.8,
        confidence_percentage=80.0,
        probabilities={"glioma": 0.8, "meningioma": 0.1, "notumor": 0.05, "pituitary": 0.05},
        ranked_probabilities=[
            {"class_name": "glioma", "value": 0.8, "percentage": 80.0},
            {"class_name": "meningioma", "value": 0.1, "percentage": 10.0},
            {"class_name": "notumor", "value": 0.05, "percentage": 5.0},
            {"class_name": "pituitary", "value": 0.05, "percentage": 5.0},
        ],
        model="brain_tumor_efficientnet_b4.onnx",
        model_version="2.0.0",
        device="CPUExecutionProvider",
        tumor_present=True,
        warning="Research/educational use only.",
    )
    assert payload.tumor_present is True
    assert len(payload.probabilities) == 4
