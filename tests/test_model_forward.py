import torch


def test_autoencoder_forward_shape():
    from credit_card_fraud_analysis.lightning_module import LitAutoEncoder

    input_dim = 10  # Example input dimension
    model = LitAutoEncoder(
        input_dim=input_dim,
        hidden_dim=8,
        dropout=0.1,
        lr=0.001,
        weight_decay=0.0,
    )

    x = torch.randn((4, input_dim))  # Batch size of 4
    y = model(x)

    assert y.shape == x.shape, "Output shape should match input shape"
