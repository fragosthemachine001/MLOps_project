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

from credit_card_fraud_analysis.data import transform_data, generate_train_data, preprocess_data
from credit_card_fraud_analysis.model import Autoencoder


def train():
    X_train, _, _, _, X_train_tensor, X_test_tensor = preprocess_data()

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model
    autoencoder = Autoencoder(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Train the autoencoder
    epochs = 50
    for epoch in range(epochs):
        for batch_data, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = autoencoder(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    train()
