from types import SimpleNamespace

import torch

from models.finetune import SupervisedFineTune
from train.probe import build_probe_model


def make_flat_cfg():
    return SimpleNamespace(
        d=16, L=2, k=4, freeze_encoder=True,
        lr_encoder=0.0, lr_heads=1e-3, weight_decay=0.01,
        lambda_direction=1.0, lambda_classification=0.5,
    )


def test_build_probe_model_returns_frozen_supervised_finetune(tmp_path):
    cfg = make_flat_cfg()
    base_model = SupervisedFineTune(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": base_model.state_dict()}, ckpt_path)

    probe_model = build_probe_model(cfg, str(ckpt_path))

    assert isinstance(probe_model, SupervisedFineTune)
    assert all(not p.requires_grad for p in probe_model.encoder.parameters())
    assert all(p.requires_grad for p in probe_model.direction_head.parameters())
    assert all(p.requires_grad for p in probe_model.classification_head.parameters())
