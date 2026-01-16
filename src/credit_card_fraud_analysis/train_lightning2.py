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
from credit_card_fraud_analysis.utils.my_logger import logger

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
app = typer.Typer()


@app.command()
def train():
    """
    Train the autoencoder using PyTorch Lightning + W&B logging.
    """
    try:
        # Load Hydra config
        logger.info("Loading configuration...")
        with initialize(version_base="1.2", config_path="../../configs"):
            config = compose(config_name="config")
        logger.debug(f"Configuration loaded successfully")

        torch.manual_seed(config.seed)
        logger.debug(f"Random seed set to: {config.seed}")

        # 1) Load / preprocess data
        logger.info("Starting data preprocessing...")
        X_train, _, _, _, X_train_tensor, _ = preprocess_data()
        logger.info(f"Data preprocessing complete. Training data shape: {X_train_tensor.shape}")

        # 2) Create DataLoader
        logger.debug(f"Creating DataLoader with batch size: {config.training.batch_size}")
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.info(f"DataLoader created with {len(train_dataset)} samples, {len(train_loader)} batches")

        # 3) Lightning Module
        logger.debug("Initializing autoencoder model...")
        input_dim = X_train_tensor.shape[1]
        model = LitAutoEncoder(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        logger.info(f"Model initialized - Input dim: {input_dim}, Hidden dim: {config.model.hidden_dim}, LR: {config.training.lr}")
        
        # Check for potential issues
        if config.training.lr < 0.001:
            logger.warning(f"Learning rate is very low: {config.training.lr}")

        # 4) W&B Logger (logs metrics + can store model checkpoints as artifacts)
        wandb_project = getattr(config, "wandb", {}).get("project", "mlops-credit-card-fraud")
        wandb_name = getattr(config, "wandb", {}).get("name", "lit-autoencoder")
        logger.info(f"Setting up W&B logger - Project: {wandb_project}, Name: {wandb_name}")
        
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            log_model=True,
        )
        logger.debug("W&B logger configured successfully")

        # 5) Callbacks (checkpoint + LR monitor)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Models directory created/verified: {MODELS_DIR}")

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
        logger.info(f"Callbacks configured - ModelCheckpoint saving to {MODELS_DIR}")

        # 6) Trainer
        device_type = "cpu" if config.device == "cpu" else "auto"
        logger.info(f"Starting training on device: {device_type}")
        logger.debug(f"Training configuration - Epochs: {config.training.epochs}, Batch size: {config.training.batch_size}")
        
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            accelerator=device_type,
            logger=wandb_logger,
            callbacks=callbacks,
            log_every_n_steps=10,
            profiler="simple",
        )
        logger.debug("Trainer configured successfully")

        # 7) Train
        logger.info("Starting training loop")
        trainer.fit(model, train_loader)
        logger.info("Training completed successfully")

        logger.info(f"Training complete. Best checkpoint saved to: {MODELS_DIR}")
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        logger.error("Something went wrong in training process")
        raise


if __name__ == "__main__":
    app()
