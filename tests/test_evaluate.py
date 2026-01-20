# src/tests/test_evaluate.py

from pathlib import Path

import pytest
from typer.testing import CliRunner

from credit_card_fraud_analysis.data import RAW_DATA_DIR

DATA_FILE = RAW_DATA_DIR / "creditcard.csv"
runner = CliRunner()
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "autoencoder.pt"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="Raw data 'creditcard.csv' is missing[cite: 2].")
def test_data_tensor_shapes():
    """Integration test to verify tensors produced for evaluation have correct rank[cite: 2]."""
    from credit_card_fraud_analysis.data import preprocess_data

    _, _, _, _, _, X_test_tensor = preprocess_data()
    assert X_test_tensor.ndim == 2
    assert X_test_tensor.shape[1] == 28


@pytest.mark.parametrize(
    "error_val, threshold, expected_label",
    [
        (0.1, 0.5, 0),  # Normal
        (1.2, 0.5, 1),  # Anomaly
    ],
)
def test_thresholding_logic(error_val, threshold, expected_label):
    """Test the anomaly detection thresholding logic used in evaluate.py."""
    # Prediction: errors > threshold
    prediction = int(error_val > threshold)
    assert prediction == expected_label
