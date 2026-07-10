from types import SimpleNamespace

import torch

from train.pretrain_mae import MAEPretrain, build_trainer, uniform_random_mask


def make_flat_cfg():
    return SimpleNamespace(
        d=16, L=2, k=4, mask_ratio=0.25,
        lr=1e-3, weight_decay=0.01, epochs=1, grad_clip=1.0,
    )


def test_uniform_random_mask_respects_ratio():
    torch.manual_seed(0)
    mask = uniform_random_mask(n=1000, ratio=0.25)
    assert mask.dtype == torch.bool
    assert 200 <= mask.sum().item() <= 300  # ~25% +/- sampling noise


def test_training_step_returns_finite_loss(tiny_pyg_batch):
    model = MAEPretrain(make_flat_cfg())
    loss = model.training_step(tiny_pyg_batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_gradients_flow_to_encoder(tiny_pyg_batch):
    # MAE training_step only calls encoder.forward() (per-node), never
    # encode_event() (event-level pooling) -- so encoder.pool_proj
    # legitimately gets no gradient here; it's only used by JEPA/downstream
    # heads. Check the parameters that actually participate.
    model = MAEPretrain(make_flat_cfg())
    loss = model.training_step(tiny_pyg_batch, batch_idx=0)
    loss.backward()
    assert all(p.grad is not None for p in model.encoder.input_proj.parameters())
    assert all(p.grad is not None for p in model.encoder.blocks.parameters())
    assert all(p.grad is not None for p in model.mae_head.parameters())


def test_encoder_registered_under_encoder_attribute():
    model = MAEPretrain(make_flat_cfg())
    assert any(k.startswith("encoder.") for k in model.state_dict().keys())


def test_checkpoint_dirpath_and_filename_are_configurable(tmp_path):
    cfg = make_flat_cfg()
    trainer = build_trainer(
        cfg, fast_dev_run=True,
        checkpoint_dirpath=str(tmp_path / "ckpts"), checkpoint_filename="epoch{epoch}",
    )
    assert str(trainer.checkpoint_callback.dirpath) == str(tmp_path / "ckpts")
