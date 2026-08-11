import math

import torch

from models.heads import ClassificationHead, DirectionHead


def test_direction_head_output_ranges():
    head = DirectionHead(d=16)
    x = torch.randn(32, 16)
    az, zen = head(x)
    assert az.shape == (32,)
    assert zen.shape == (32,)
    assert (az >= 0).all() and (az <= 2 * math.pi).all()
    assert (zen >= 0).all() and (zen <= math.pi).all()


def test_direction_head_gradients_flow():
    head = DirectionHead(d=16)
    x = torch.randn(8, 16, requires_grad=True)
    az, zen = head(x)
    (az.sum() + zen.sum()).backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in head.parameters())


def test_classification_head_output_shape():
    head = ClassificationHead(d=16)
    x = torch.randn(32, 16)
    logits = head(x)
    assert logits.shape == (32, 1)


def test_classification_head_gradients_flow():
    head = ClassificationHead(d=16)
    x = torch.randn(8, 16, requires_grad=True)
    logits = head(x)
    logits.sum().backward()
    assert x.grad is not None


def test_direction_head_output_diversity_survives_wildly_different_input_scales():
    """Root cause fixed by LayerNorm inside DirectionHead.mlp, confirmed
    directly on a real run (scratch_v3, step 30,922): az/zen collapsed to
    an exact constant even with a healthy, non-collapsed encoder underneath
    (node_emb std=0.57) -- traced to DirectionHead's own internal layers,
    which have no normalization and let diversity erode through the stack
    (measured std 0.065 -> 0.027 -> 0.056 -> 0.032 across real layers),
    ending in a deeply saturated final Linear (pre-sigmoid range
    [-16.3, -4.5], sigmoid collapses to ~0 regardless of input). Simulates
    the failure condition directly: inputs whose scale varies a lot
    (matching how g's raw magnitude varies before normalization elsewhere
    in the pipeline) must not collapse az/zen output diversity."""
    torch.manual_seed(0)
    head = DirectionHead(d=16)
    x = torch.randn(64, 16) * torch.linspace(0.1, 50.0, 64).unsqueeze(-1)
    az, zen = head(x)
    assert az.std() > 0.01
    assert zen.std() > 0.01
