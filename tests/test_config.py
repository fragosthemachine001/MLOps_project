import pytest
from credit_card_fraud_analysis.hydra_config_loader import load_config


def test_load_config():
    """Check if the config object loads and has required sections."""
    config = load_config()
    assert config is not None
    assert "model" in config
    assert "training" in config
    assert "evaluation" in config


@pytest.mark.parametrize("param_path, expected_type", [
    ("model.hidden_dim", int),
    ("training.lr", float),
    ("training.epochs", int),
    ("device", str),
])
def test_config_parameter_types(param_path, expected_type):
    """Verify specific config values have the correct types."""
    config = load_config()

    # Navigate the nested Omegaconf object
    value = config
    for part in param_path.split("."):
        value = getattr(value, part)

    assert isinstance(value, expected_type), f"{param_path} should be {expected_type}"