import pytest
from fastapi.testclient import TestClient

from credit_card_fraud_analysis.api import app


def test_predict_success():
    """
    Test a successful prediction with the correct feature dimension.
    The 'with' statement triggers the lifespan events (model loading).
    """
    # Use the context manager to trigger startup/shutdown lifespan events
    with TestClient(app) as client:
        # Your model expects 28 features based on previous errors
        # We send 28 dummy features
        payload = {"features": [0.1] * 28}

        response = client.post("/predict", json=payload)

        # Assertions
        if response.status_code == 503:
            pytest.skip("Model artifact not available in CI environment")

        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "reconstruction_error" in data
        assert isinstance(data["is_fraud"], bool)


def test_predict_optimized_skips_if_missing():
    """
    Test the optimized endpoint when the quantized/model is available.
    In CI, the artifact is typically, so the APIA should respond with 503,
    and we skip the tst rather than failing.
    """
    with TestClient(app) as client:
        payload = {"features": [0.1] * 28}
        response = client.post("/predict_optimized", json=payload)

        if response.status_code == 503:
            pytest.skip("Optimized/quantized model artifact not available in CI environmet")

        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "reconstruction_error" in data
        assert isinstance(data["is_fraud"], bool)


def test_predict_dimension_mismatch():
    """
    Test that the API correctly returns a 400 error when the
    feature dimension is incorrect.
    """
    with TestClient(app) as client:
        # Send only 1 feature when the model expects more (e.g., 28)
        payload = {"features": [0.5]}

        response = client.post("/predict", json=payload)

        if response.status_code == 503:
            pytest.skip("Model artifact not available in CI environment")

        assert response.status_code == 400
        assert "Dimension mismatch" in response.json()["detail"]


def test_predict_invalid_data():
    """
    Test that the API returns a 422 error for non-numeric data.
    """
    with TestClient(app) as client:
        payload = {"features": ["not", "a", "number"]}

        response = client.post("/predict", json=payload)

        if response.status_code == 503:
            pytest.skip("Model artifact not available in CI environment")

        assert response.status_code == 422
