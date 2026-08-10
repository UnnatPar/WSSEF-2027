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


class _FakeTrainer:
    """Minimal stand-in exposing exactly what configure_optimizers reads --
    see train/pretrain_mae.py's test_pretrain_mae_smoke.py for why this is
    necessary: configure_optimizers now reads self.trainer.global_step and
    self.trainer.estimated_stepping_batches directly, not a scheduler's own
    internal step counter."""

    def __init__(self, global_step=0, estimated_stepping_batches=1000):
        self.global_step = global_step
        self.estimated_stepping_batches = estimated_stepping_batches


def make_model_with_fake_trainer(**cfg_overrides):
    model = NeutrinoJEPA(make_cfg(**cfg_overrides))
    model.trainer = _FakeTrainer()
    return model


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
    model = make_model_with_fake_trainer()
    # Past warmup_cosine_lr_lambda's warmup window, or the very first
    # optimizer.step() below is a no-op (lr=0.0 by design at global_step=0)
    # and neither the encoder nor its EMA shadow would move at all.
    model.trainer.global_step = 200
    before = [p.clone() for p in model.ema.shadow_params]
    optimizer = model.configure_optimizers()["optimizer"]
    # configure_optimizers needs self.trainer; training_step's self.log()
    # call needs it to be either fully Lightning-real or unset (our fake
    # stub only satisfies configure_optimizers' two attributes) -- unset it
    # now that the optimizer is built.
    model.trainer = None
    loss = model.training_step(tiny_pyg_batch, batch_idx=0)
    loss.backward()
    optimizer.step()
    model.on_before_zero_grad(optimizer)
    after = model.ema.shadow_params
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_checkpoint_roundtrip_preserves_ema_state(tiny_pyg_batch):
    model = make_model_with_fake_trainer()
    optimizer = model.configure_optimizers()["optimizer"]
    model.trainer = None  # see test_ema_update_changes_shadow_params
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


def test_configure_optimizers_returns_adamw_and_decaying_schedule():
    model = make_model_with_fake_trainer()
    result = model.configure_optimizers()
    assert isinstance(result["optimizer"], torch.optim.AdamW)
    assert isinstance(result["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    assert result["lr_scheduler"]["interval"] == "step"


def test_lr_reflects_trainer_global_step_not_the_schedulers_own_counter():
    """Same mechanism/test pattern as train/pretrain_mae.py's
    MAEPretrain -- see that file's configure_optimizers for the full
    reasoning. Proves the lambda tracks global_step directly: with the
    scheduler's own counter frozen at 0 (zero .step() calls made), just
    mutating global_step externally (as a real resume effectively does)
    must still move the LR.

    Compares against cfg.lr (the schedule's peak), not lr at global_step=0:
    warmup_cosine_lr_lambda deliberately makes lr=0.0 exactly at step 0 (see
    train/optim.py), so lr_at_step_0 is no longer a valid "near-full-LR"
    reference point now that warmup exists."""
    model = make_model_with_fake_trainer(estimated_stepping_batches=1000)
    result = model.configure_optimizers()
    optimizer, scheduler = result["optimizer"], result["lr_scheduler"]["scheduler"]

    model.trainer.global_step = 900
    scheduler.step()
    lr_after_resume = optimizer.param_groups[0]["lr"]

    assert lr_after_resume < model.cfg.lr * 0.1


def test_encoder_registered_under_encoder_attribute_for_checkpoint_loading():
    model = NeutrinoJEPA(make_cfg())
    state_dict_keys = model.state_dict().keys()
    assert any(k.startswith("encoder.") for k in state_dict_keys)
