from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import typer


from credit_card_fraud_analysis.data import preprocess_data
from credit_card_fraud_analysis.hydra_config_loader import load_config
from credit_card_fraud_analysis.model import Autoencoder

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
app = typer.Typer(add_completion=False)


@app.command()
def train():
    config = load_config()

    torch.manual_seed(config.seed)

    # 1. Load Data
    print("Preprocessing data...")
    X_train, _, _, _, X_train_tensor, _ = preprocess_data()

    # 2. Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4
    )

    # 3. Initialize model and move to device
    device = torch.device(config.device)
    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(
        input_dim=input_dim,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout
    ).to(device)

    # 4. Optimizer & Loss
    opt_class = getattr(torch.optim, config.training.optimizer)
    optimizer = opt_class(
        autoencoder.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    criterion = nn.MSELoss()

    print(f"Starting training on {device}...")

    # 5. Training Loop
    autoencoder.train()
    for epoch in range(config.training.epochs):
        print(f"Epoch {epoch + 1}/{config.training.epochs} ...")
        epoch_loss = 0.0
        for batch_data, _ in train_loader:
            # IMPORTANT: Move data to device to prevent freeze
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            reconstructed = autoencoder(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{config.training.epochs}], Loss: {avg_loss:.4f}')

    # 6. Save Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / config.evaluation.model_filename
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    app()
