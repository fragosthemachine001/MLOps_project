from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Convert the DataFrame into two variable
    X: data columns (V1 - V28)
    y: lable column
    """
    X = df.iloc[:, 2:30].values
    y = df.Class.values
    return X, y


def compare_plot(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str):
    # Ensure the directory exists
    FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

    plt.figure(figsize=(12, 6))  # Create a new figure to avoid overlapping

    # Plot 1: Original
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c="r")
    plt.title("Original Set")

    # Plot 2: Resampled
    plt.subplot(1, 2, 2)
    plt.scatter(
        X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15
    )
    plt.scatter(
        X_resampled[y_resampled == 1, 0],
        X_resampled[y_resampled == 1, 1],
        label="Class #1",
        alpha=0.5,
        linewidth=0.15,
        c="r",
    )
    plt.title(f"Method: {method}")

    plt.legend()

    # Dynamic filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = FIGURES_DIR / f"comparison_{method}_{timestamp}.png"

    # Save and close
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def generate_train_data(df):
    # 1. Prepare X and y from the raw data
    X, y = prep_data(df)

    # 2. Split first: 70% train, 30% test
    # This keeps the test set "unseen" and realistically imbalanced.
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    # 3. Apply SMOTE ONLY to the training data
    method = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = method.fit_resample(X_train_raw, y_train_raw)

    # Return perfectly balanced training data and realistically imbalanced test data
    return X_train_resampled, X_test, y_train_resampled, y_test


def transform_data(X_train, X_test):
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    return X_train_tensor, X_test_tensor


def preprocess_data():
    df = pd.read_csv(RAW_DATA_DIR / "creditcard.csv")
    df.info()
    df.head()
    # Count the occurrences of fraud and no fraud and print them
    occ = df["Class"].value_counts()
    print(occ)
    ratio_cases = occ / len(df.index)
    print(f"Ratio of fraudulent cases: {ratio_cases[1]}\nRatio of non-fraudulent cases: {ratio_cases[0]}")
    X_train, X_test, y_train, y_test = generate_train_data(df)
    X_train_tensor, X_test_tensor = transform_data(X_train, X_test)
    return X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor


if __name__ == "__main__":
    typer.run(prep_data)
