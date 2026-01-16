# src/credit_card_fraud_analysis/train_lightning.py
from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
from hydra import compose, initialize
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from credit_card_fraud_analysis.data import preprocess_data
from credit_card_fraud_analysis.lightning_module import LitAutoEncoder

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
app = typer.Typer()


@app.command()
def train():
    """
    Train the autoencoder using PyTorch Lightning + W&B logging.
    """
    # Load Hydra config
    with initialize(version_base="1.2", config_path="../../configs"):
        config = compose(config_name="config")

    torch.manual_seed(config.seed)

    # 1) Load / preprocess data
    print("Preprocessing data...")
    X_train, _, _, _, X_train_tensor, _ = preprocess_data()

    # 2) Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 3) Lightning Module
    model = LitAutoEncoder(
        input_dim=X_train_tensor.shape[1],
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # 4) W&B Logger (logs metrics + can store model checkpoints as artifacts)
    wandb_logger = WandbLogger(
        project=getattr(config, "wandb", {}).get("project", "mlops-credit-card-fraud"),
        name=getattr(config, "wandb", {}).get("name", "lit-autoencoder"),
        log_model=True,
    )

    # 5) Callbacks (checkpoint + LR monitor)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(MODELS_DIR),
            filename="lit-autoencoder-{epoch:02d}-{train_loss:.4f}",
            save_top_k=1,
            monitor="train_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # 6) Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="cpu" if config.device == "cpu" else "auto",
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        profiler="simple",
    )

    # 7) Train
    trainer.fit(model, train_loader)

    print(f"Done. Best checkpoint saved under: {MODELS_DIR}")


if __name__ == "__main__":
    app()
