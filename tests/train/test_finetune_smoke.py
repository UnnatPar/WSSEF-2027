from types import SimpleNamespace

import torch

from models.finetune import SupervisedFineTune
from train.finetune import build_finetune_model


def make_flat_cfg(**overrides):
    base = dict(
        d=16, L=2, k=4, freeze_encoder=False,
        lr_encoder=1e-4, lr_heads=1e-3, weight_decay=0.01,
        lambda_direction=1.0, lambda_classification=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_finetune_from_checkpoint_unfreezes_encoder(tmp_path):
    cfg = make_flat_cfg()
    base_model = SupervisedFineTune(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": base_model.state_dict()}, ckpt_path)

    model = build_finetune_model(cfg, str(ckpt_path))

    assert all(p.requires_grad for p in model.encoder.parameters())


def test_finetune_from_scratch_when_checkpoint_is_none():
    cfg = make_flat_cfg()
    model = build_finetune_model(cfg, None)
    assert all(p.requires_grad for p in model.encoder.parameters())
    # Random init: two independent calls should not produce identical weights.
    other_model = build_finetune_model(cfg, None)
    first_param = next(model.encoder.parameters())
    other_first_param = next(other_model.encoder.parameters())
    assert not torch.equal(first_param, other_first_param)
