from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, \
    precision_recall_curve, average_precision_score
from credit_card_fraud_analysis.data import transform_data, generate_train_data, preprocess_data
from credit_card_fraud_analysis.model import Autoencoder

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
from credit_card_fraud_analysis.hydra_config_loader import load_config

app = typer.Typer(add_completion=False)


@app.command()
def evaluate():
    config = load_config()
    X_train, X_test, _, y_test, _, X_test_tensor = preprocess_data()
    autoencoder = Autoencoder(X_train.shape[1],
                              hidden_dim=config.model.hidden_dim,
                              dropout=config.model.dropout)
    autoencoder.load_state_dict(
        torch.load(MODELS_DIR / "autoencoder.pt", map_location=torch.device("cpu"))
    )

    autoencoder.eval()
    with torch.no_grad():
        # Get reconstruction errors for test data
        reconstructions = autoencoder(X_test_tensor)
        errors = torch.mean((X_test_tensor - reconstructions) ** 2, dim=1).numpy()

    # Set threshold for anomaly detection (using percentile approach)
    threshold = np.percentile(errors, config.evaluation.threshold_percentile)

    # Make predictions based on threshold
    y_pred = (errors > threshold).astype(int)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nThreshold used: {threshold:.4f}")
    print(f"Number of anomalies detected: {np.sum(y_pred)}")
    print(f"Actual number of fraud cases: {np.sum(y_test)}")

    # Conditional metric reporting
    if "roc_auc" in config.evaluation.metrics:
        auc = roc_auc_score(y_test, errors)
        print(f"ROC AUC Score: {auc:.4f}")


if __name__ == "__main__":
    app()
