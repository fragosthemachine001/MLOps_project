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

import torch.nn as nn
import torch.optim as optim


def prep_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Convert the DataFrame into two variable
    X: data columns (V1 - V28)
    y: lable column
    """
    X = df.iloc[:, 2:30].values
    y = df.Class.values
    return X, y

# Define a function to create a scatter plot of our data and labels
def plot_data(X: np.ndarray, y: np.ndarray):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()

def compare_plot(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray, method: str):
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title('Original Set')
    plt.subplot(1, 2, 2)
    plt.scatter(X_resampled[y_resampled == 0, 0], X_resampled[y_resampled == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X_resampled[y_resampled == 1, 0], X_resampled[y_resampled == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title(method)
    plt.legend()
    plt.show()

def generate_train_data(df):
    # Create X and y from the prep_data function
    X, y = prep_data(df)
    # Plot our data by running our plot data function on X and y
    plot_data(X, y)
    # Reproduced using the DataFrame
    plt.scatter(df.V2[df.Class == 0], df.V3[df.Class == 0], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(df.V2[df.Class == 1], df.V3[df.Class == 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    plt.show()
    # SMOTE
    print(f'X shape: {X.shape}\ny shape: {y.shape}')
    # Define the resampling method
    method = SMOTE()

    # Create the resampled feature set
    X_resampled, y_resampled = method.fit_resample(X, y)
    # Plot the resampled data
    plot_data(X_resampled, y_resampled)
    pd.value_counts(pd.Series(y))
    pd.value_counts(pd.Series(y_resampled))
    compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

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
    df = pd.read_csv("data/raw/creditcard.csv")
    df.info()
    df.head()
    # Count the occurrences of fraud and no fraud and print them
    occ = df['Class'].value_counts()
    print(occ)
    ratio_cases = occ / len(df.index)
    print(f'Ratio of fraudulent cases: {ratio_cases[1]}\nRatio of non-fraudulent cases: {ratio_cases[0]}')
    X_train, X_test, y_train, y_test = generate_train_data(df)
    X_train_tensor, X_test_tensor = transform_data(X_train, X_test)
    return X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor


if __name__ == "__main__":
    typer.run(prep_data)
