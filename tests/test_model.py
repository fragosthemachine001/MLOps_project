import pytest
import torch
from credit_card_fraud_analysis.model import Autoencoder

@pytest.mark.parametrize("input_dim, hidden_dim, dropout", [
    (30, 16, 0.1),
    (28, 10, 0.2),
    (100, 50, 0.0)
])
def test_autoencoder_dimensions(input_dim, hidden_dim, dropout):
    """Test if the model output shape matches the input shape."""
    model = Autoencoder(input_dim, hidden_dim, dropout)
    x = torch.randn(1, input_dim)
    output = model(x)
    assert output.shape == (1, input_dim), f"Expected shape (1, {input_dim}), got {output.shape}"

def test_model_layers():
    """Verify the bottleneck dimension is half of the hidden dimension."""
    hidden_dim = 16
    model = Autoencoder(input_dim=30, hidden_dim=hidden_dim, dropout=0.1)
    layer = model.decoder[0]
    assert isinstance(layer, torch.nn.Linear)
    assert layer.in_features == hidden_dim // 2