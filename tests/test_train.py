# src/tests/test_train.py

import pytest
import torch
import torch.nn as nn

from credit_card_fraud_analysis.data import RAW_DATA_DIR
from credit_card_fraud_analysis.hydra_config_loader import load_config
from credit_card_fraud_analysis.model import Autoencoder

DATA_MISSING = not (RAW_DATA_DIR / "creditcard.csv").exists()


@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_model_forward_pass_shapes(batch_size):
    """Ensure the autoencoder output shape matches the input shape for various batches."""
    input_dim = 30
    model = Autoencoder(input_dim=input_dim, hidden_dim=16, dropout=0.1)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size, input_dim)


def test_training_step_weight_update():
    """Verify that a single optimizer step actually changes model weights[cite: 1]."""
    config = load_config()
    input_dim = 30
    model = Autoencoder(input_dim, config.model.hidden_dim, config.model.dropout)

    # Store initial weights
    initial_weights = [p.clone() for p in model.parameters()]

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    criterion = nn.MSELoss()

    # One training iteration
    dummy_input = torch.randn(8, input_dim)
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_input)
    loss.backward()
    optimizer.step()

    # Check if weights updated
    for p_init, p_after in zip(initial_weights, model.parameters()):
        assert not torch.equal(p_init, p_after), "Weights did not change after backpropagation."
