from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import typer
import wandb

# logger
from utils.my_logger import logger

# for wadb
from dotenv import load_dotenv 
load_dotenv() 
import os


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
    logger.info("Starting data preprocessing...")
    X_train, _, _, _, X_train_tensor, _ = preprocess_data()

    # 2. Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0
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

    logger.info(f"Starting training on device: {device}")
    logger.debug(f"Batch size: {config.training.batch_size}")
    if config.training.lr < 0.001:
        logger.warning(f"Learning rate is very low: {config.training.lr}")

    # Initialize wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config={
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.lr,
            "model": "Autoencoder",
            "hidden_dim": config.model.hidden_dim,
            "dropout": config.model.dropout,
            "optimizer": config.training.optimizer,
            "weight_decay": config.training.weight_decay,
            "input_dim": input_dim,
        }
    )

    # 5. Training Loop
    logger.info("Starting training loop")
    autoencoder.train()
    for epoch in range(config.training.epochs):
        epoch_loss = 0.0
        logger.debug(f"Starting epoch {epoch + 1}/{config.training.epochs}")
        
        for batch_idx, (batch_data, _) in enumerate(train_loader):
            logger.debug(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
            try:
                # IMPORTANT: Move data to device to prevent freeze
                batch_data = batch_data.to(device)
                logger.debug(f"Batch shape: {batch_data.shape}")

                optimizer.zero_grad()
                reconstructed = autoencoder(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                logger.debug(f"Loss: {loss.item():.6f}")
                
                # Log batch metrics to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1
                })
                
            except Exception as e:
                logger.exception(f"Error in epoch {epoch + 1}, batch {batch_idx + 1}: {e}")
                logger.error("Something went wrong in training step")
                raise

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f'Epoch [{epoch + 1}/{config.training.epochs}], Average Loss: {avg_loss:.4f}')
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch_avg_loss": avg_loss,
                "epoch": epoch + 1
            })

    # 6. Save Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / config.evaluation.model_filename
    torch.save(autoencoder.state_dict(), model_path)
    logger.info(f"Training complete. Model saved to: {model_path}")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    app()