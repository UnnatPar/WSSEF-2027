import torch

from models.polarbert import MAEHead, PolarBERTEncoder


def test_encoder_output_shape_no_padding():
    encoder = PolarBERTEncoder(d=16, n_layers=2, n_heads=4)
    x = torch.randn(4, 10, 6)
    out = encoder(x)
    assert out.shape == (4, 16)


def test_encoder_output_shape_with_padding_mask():
    encoder = PolarBERTEncoder(d=16, n_layers=2, n_heads=4)
    x = torch.randn(4, 10, 6)
    padding_mask = torch.zeros(4, 10, dtype=torch.bool)
    padding_mask[:, 7:] = True  # last 3 positions of every event are padding
    out = encoder(x, padding_mask=padding_mask)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_encoder_gradients_flow():
    encoder = PolarBERTEncoder(d=16, n_layers=2, n_heads=4)
    x = torch.randn(4, 10, 6, requires_grad=True)
    out = encoder(x)
    out.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in encoder.parameters())


def test_mae_head_output_shape():
    head = MAEHead(d=16)
    node_embeddings = torch.randn(30, 16)
    out = head(node_embeddings)
    assert out.shape == (30, 2)
