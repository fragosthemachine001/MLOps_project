from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from credit_card_fraud_analysis.data import preprocess_data
from credit_card_fraud_analysis.model import Autoencoder

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

app = typer.Typer()

@app.command()
def evaluate():
    X_train, X_test, _, y_test, _, X_test_tensor = preprocess_data()
    autoencoder = Autoencoder(X_train.shape[1])
    autoencoder.load_state_dict(
        torch.load(MODELS_DIR / "autoencoder.pt", map_location=torch.device("cpu"))
    )

    autoencoder.eval()
    with torch.no_grad():
        # Get reconstruction errors for test data
        test_reconstructions = autoencoder(X_test_tensor)
        reconstruction_errors = torch.mean((X_test_tensor - test_reconstructions) ** 2, dim=1)

    # Convert to numpy for further processing
    reconstruction_errors_np = reconstruction_errors.numpy()

    # Set threshold for anomaly detection (using percentile approach)
    threshold = np.percentile(reconstruction_errors_np, 95)  # Top 5% as anomalies

    # Make predictions based on threshold
    y_pred = (reconstruction_errors_np > threshold).astype(int)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nThreshold used: {threshold:.4f}")
    print(f"Number of anomalies detected: {np.sum(y_pred)}")
    print(f"Actual number of fraud cases: {np.sum(y_test)}")

if __name__ == "__main__":
    app()