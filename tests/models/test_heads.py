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
