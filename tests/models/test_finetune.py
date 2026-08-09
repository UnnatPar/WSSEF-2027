from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from models.finetune import SupervisedFineTune, build_supervised_model, load_full_checkpoint
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


def test_load_full_checkpoint_restores_heads_not_just_encoder(tmp_path):
    trained = SupervisedFineTune(make_cfg())
    with torch.no_grad():
        for p in trained.direction_head.parameters():
            p.add_(1.0)  # simulate training having moved the head weights
    ckpt_path = tmp_path / "finetune_ckpt.pt"
    torch.save({"state_dict": trained.state_dict()}, ckpt_path)

    loaded = load_full_checkpoint(make_cfg(), str(ckpt_path))

    for trained_p, loaded_p in zip(trained.direction_head.parameters(), loaded.direction_head.parameters()):
        assert torch.equal(trained_p, loaded_p)
    for trained_p, loaded_p in zip(trained.encoder.parameters(), loaded.encoder.parameters()):
        assert torch.equal(trained_p, loaded_p)
    assert not loaded.training


def test_training_step_stays_finite_when_kappa_head_saturates_to_zero():
    """Real production run: `train/loss` was NaN on ~80-100% of logged steps.
    Root cause -- reproduced directly against graphnet's VonMisesFisher3DLoss,
    not assumed: kappa_head's Softplus output underflows to exactly 0.0 for a
    sufficiently negative pre-activation (this is much easier to hit than it
    sounds under `precision="16-mixed"`, since fp16's subnormal floor is
    ~6e-8 vs fp32's ~1e-38). At kappa=0.0 exactly, graphnet's LogCMK.forward
    computes `log(kappa) - log(iv(0.5, kappa))` = -inf - (-inf) = NaN in
    floating point, even though the mathematical limit is finite."""
    model = SupervisedFineTune(make_cfg())
    with torch.no_grad():
        model.kappa_head.weight.zero_()
        model.kappa_head.bias.fill_(-1000.0)  # forces Softplus(x) == 0.0 exactly
    batch = make_labeled_batch(with_pid=False)
    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_direction_head_still_gets_meaningful_gradient_when_kappa_head_collapses():
    """A second, worse failure the finite-loss test above doesn't catch:
    measured directly against graphnet's real loss, d(loss)/d(kappa) is
    POSITIVE whenever direction predictions are still wrong (the normal state
    at the start of training) -- so gradient descent drives kappa_head's raw
    output toward zero from step one. Once kappa is near zero, the gradient
    reaching direction_head collapses ~10,000x (measured: 0.13 at kappa=1.0
    vs 0.000013 at kappa=1e-4), starving it of signal and freezing bad
    direction predictions permanently -- which in turn keeps kappa pinned at
    zero. This is a self-reinforcing collapse, not a numerical edge case: it
    pinned real production runs at a constant loss for 1.7k+ steps with zero
    learning. The fix (flooring kappa at 1.0, not just epsilon-above-zero)
    must guarantee direction_head keeps receiving a non-negligible gradient
    even when kappa_head's own output has collapsed to ~0."""
    model = SupervisedFineTune(make_cfg())
    with torch.no_grad():
        model.kappa_head.weight.zero_()
        model.kappa_head.bias.fill_(-1000.0)  # kappa_head's raw output collapses to 0
    batch = make_labeled_batch(with_pid=False)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    direction_grad_norm = torch.cat(
        [p.grad.flatten() for p in model.direction_head.parameters()]
    ).norm()
    assert direction_grad_norm > 1e-3


def test_kappa_head_gradient_survives_even_when_its_own_output_has_saturated():
    """A third failure, discovered after fixing the first two: flooring
    kappa's *value* at 1.0 stops the collapse, but plain Softplus's own
    gradient (sigmoid of the pre-activation) still underflows to exactly 0.0
    once the optimizer -- which still wants to push kappa down whenever
    direction predictions are imperfect -- drives kappa_head's pre-activation
    far enough negative. Verified by direct step-by-step tracing on a real
    batch: kappa_grad_norm hit exactly 0.0 by step 20 of a real overfit run
    and never recovered, permanently dead-neuroning kappa_head regardless of
    how training proceeded afterward. The leak term (LeakyReLU-style) must
    keep a nonzero gradient flowing into kappa_head no matter how negative
    its pre-activation gets."""
    model = SupervisedFineTune(make_cfg())
    with torch.no_grad():
        model.kappa_head.weight.zero_()
        model.kappa_head.bias.fill_(-1000.0)
    batch = make_labeled_batch(with_pid=False)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    kappa_grad_norm = torch.cat(
        [p.grad.flatten() for p in model.kappa_head.parameters()]
    ).norm()
    assert kappa_grad_norm > 0


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
