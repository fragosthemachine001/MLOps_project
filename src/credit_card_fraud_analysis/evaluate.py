from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
import typer
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.optim as optim

from credit_card_fraud_analysis.data import transform_data, generate_train_data, preprocess_data
from credit_card_fraud_analysis.model import Autoencoder

def evaluate():
    X_train, X_test, _, y_test, _, X_test_tensor = preprocess_data()
    autoencoder = Autoencoder(X_train.shape[1])

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