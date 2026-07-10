from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from models.finetune import SupervisedFineTune, build_supervised_model
from models.jepa import NeutrinoJEPA


def make_cfg(**overrides):
    base = dict(
        d=16, L=2, k=4, freeze_encoder=False,
        lr_encoder=1e-4, lr_heads=1e-3, weight_decay=0.01,
        lambda_direction=1.0, lambda_classification=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_labeled_batch(with_pid=False):
    g = torch.Generator().manual_seed(0)
    events = [Data(x=torch.rand(n, 6, generator=g)) for n in [12, 18, 9]]
    batch = Batch.from_data_list(events)
    batch.azimuth = torch.rand(3, generator=g) * 6.28
    batch.zenith = torch.rand(3, generator=g) * 3.14
    if with_pid:
        batch.pid = torch.randint(0, 2, (3,), generator=g)
    return batch


def test_training_step_returns_finite_scalar_loss_without_pid():
    model = SupervisedFineTune(make_cfg())
    batch = make_labeled_batch(with_pid=False)
    loss = model.training_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_training_step_returns_finite_scalar_loss_with_pid():
    model = SupervisedFineTune(make_cfg())
    batch = make_labeled_batch(with_pid=True)
    loss = model.training_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_training_step_backward_populates_gradients_when_unfrozen():
    model = SupervisedFineTune(make_cfg(freeze_encoder=False))
    batch = make_labeled_batch(with_pid=True)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    assert all(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.direction_head.parameters())
    assert all(p.grad is not None for p in model.classification_head.parameters())
    assert all(p.grad is not None for p in model.kappa_head.parameters())


def test_no_pid_batch_leaves_classification_head_ungraded():
    model = SupervisedFineTune(make_cfg(freeze_encoder=False))
    batch = make_labeled_batch(with_pid=False)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    assert all(p.grad is None for p in model.classification_head.parameters())
    assert all(p.grad is not None for p in model.direction_head.parameters())


def test_frozen_encoder_has_no_grad_and_optimizer_excludes_it():
    model = SupervisedFineTune(make_cfg(freeze_encoder=True))
    assert all(not p.requires_grad for p in model.encoder.parameters())
    optimizer = model.configure_optimizers()
    optimized_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    assert optimized_param_ids.isdisjoint(encoder_param_ids)


def test_unfrozen_optimizer_has_two_param_groups_with_correct_lrs():
    cfg = make_cfg(freeze_encoder=False)
    model = SupervisedFineTune(cfg)
    optimizer = model.configure_optimizers()
    lrs = sorted(g["lr"] for g in optimizer.param_groups)
    assert lrs == sorted([cfg.lr_encoder, cfg.lr_heads])


def test_build_supervised_model_loads_encoder_weights_from_jepa_checkpoint(tmp_path):
    jepa_cfg = SimpleNamespace(
        d=16, L=2, k=4, ema_decay=0.99,
        ratio_min=0.4, ratio_max=0.6, n_clusters=2,
        lr=1e-3, weight_decay=0.01, epochs=1,
    )
    jepa_model = NeutrinoJEPA(jepa_cfg)
    ckpt_path = tmp_path / "jepa_ckpt.pt"
    torch.save({"state_dict": jepa_model.state_dict()}, ckpt_path)

    supervised_model = build_supervised_model(make_cfg(freeze_encoder=True), str(ckpt_path))

    for jepa_param, loaded_param in zip(
        jepa_model.encoder.parameters(), supervised_model.encoder.parameters()
    ):
        assert torch.equal(jepa_param, loaded_param)
    assert all(not p.requires_grad for p in supervised_model.encoder.parameters())


def test_validation_step_logs_val_angular_error():
    model = SupervisedFineTune(make_cfg())
    batch = make_labeled_batch(with_pid=False)
    loss = model.validation_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_build_supervised_model_random_init_when_checkpoint_is_none():
    model_a = build_supervised_model(make_cfg(freeze_encoder=False), None)
    model_b = build_supervised_model(make_cfg(freeze_encoder=False), None)
    first_param_a = next(model_a.encoder.parameters())
    first_param_b = next(model_b.encoder.parameters())
    assert not torch.equal(first_param_a, first_param_b)
    assert all(p.requires_grad for p in model_a.encoder.parameters())
