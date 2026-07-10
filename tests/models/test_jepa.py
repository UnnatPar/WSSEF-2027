from types import SimpleNamespace

import torch

from models.jepa import NeutrinoJEPA


def make_cfg(**overrides):
    base = dict(
        d=16, L=2, k=4, ema_decay=0.99,
        ratio_min=0.4, ratio_max=0.6, n_clusters=2,
        lr=1e-3, weight_decay=0.01, epochs=10,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_training_step_returns_finite_scalar_loss(tiny_pyg_batch):
    model = NeutrinoJEPA(make_cfg())
    loss = model.training_step(tiny_pyg_batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 4.0  # bounded range of 2 - 2*cos_sim


def test_training_step_backward_populates_gradients(tiny_pyg_batch):
    model = NeutrinoJEPA(make_cfg())
    loss = model.training_step(tiny_pyg_batch, batch_idx=0)
    loss.backward()
    encoder_grads = [p.grad for p in model.encoder.parameters()]
    predictor_grads = [p.grad for p in model.predictor.parameters()]
    assert all(g is not None for g in encoder_grads)
    assert all(g is not None for g in predictor_grads)


def test_no_target_encoder_attribute():
    model = NeutrinoJEPA(make_cfg())
    assert not hasattr(model, "target_encoder")


def test_ema_update_changes_shadow_params(tiny_pyg_batch):
    model = NeutrinoJEPA(make_cfg())
    before = [p.clone() for p in model.ema.shadow_params]
    optimizer = model.configure_optimizers()[0][0]
    loss = model.training_step(tiny_pyg_batch, batch_idx=0)
    loss.backward()
    optimizer.step()
    model.on_before_zero_grad(optimizer)
    after = model.ema.shadow_params
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_checkpoint_roundtrip_preserves_ema_state(tiny_pyg_batch):
    model = NeutrinoJEPA(make_cfg())
    optimizer = model.configure_optimizers()[0][0]
    loss = model.training_step(tiny_pyg_batch, batch_idx=0)
    loss.backward()
    optimizer.step()
    model.on_before_zero_grad(optimizer)

    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    assert "ema_state" in checkpoint

    fresh_model = NeutrinoJEPA(make_cfg())
    fresh_model.on_load_checkpoint(checkpoint)
    for original, loaded in zip(model.ema.shadow_params, fresh_model.ema.shadow_params):
        assert torch.equal(original, loaded)


def test_configure_optimizers_returns_adamw_and_cosine_scheduler():
    model = NeutrinoJEPA(make_cfg())
    optimizers, schedulers = model.configure_optimizers()
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert isinstance(schedulers[0], torch.optim.lr_scheduler.CosineAnnealingLR)


def test_encoder_registered_under_encoder_attribute_for_checkpoint_loading():
    model = NeutrinoJEPA(make_cfg())
    state_dict_keys = model.state_dict().keys()
    assert any(k.startswith("encoder.") for k in state_dict_keys)
