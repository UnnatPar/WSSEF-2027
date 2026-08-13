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


def test_direction_head_split_defaults_off_but_respects_cfg():
    """split_direction_head defaults to False (joint head) when a config
    doesn't set it -- real regression found when applying split=True
    unconditionally to the from-scratch (fully cold-start) regime, see
    models/heads.py's DirectionHead docstring. mae_finetune/jepa_finetune
    configs opt in explicitly via split_direction_head: true."""
    model_default = SupervisedFineTune(make_cfg())
    assert not model_default.direction_head.split

    model_split = SupervisedFineTune(make_cfg(split_direction_head=True))
    assert model_split.direction_head.split


class _FakeTrainer:
    def __init__(self, global_step=0, estimated_stepping_batches=1000):
        self.global_step = global_step
        self.estimated_stepping_batches = estimated_stepping_batches


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
    model.trainer = _FakeTrainer()
    optimizer = model.configure_optimizers()["optimizer"]
    optimized_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    assert optimized_param_ids.isdisjoint(encoder_param_ids)


def test_unfrozen_optimizer_uses_both_lrs_with_decay_split_per_lr():
    # 4 groups, not 2: each of encoder/heads is further split into
    # decay/no_decay (see train/optim.py -- LayerNorm's gain must never be
    # decayed, which is why this split exists at all).
    cfg = make_cfg(freeze_encoder=False)
    model = SupervisedFineTune(cfg)
    model.trainer = _FakeTrainer()
    optimizer = model.configure_optimizers()["optimizer"]
    # "initial_lr" (the pre-warmup base LR each group was constructed with),
    # not "lr" -- warmup_cosine_lr_lambda deliberately makes lr=0.0 exactly
    # at global_step=0 (see train/optim.py), so runtime lr is no longer a
    # valid "which base LR was this group given" check now that warmup
    # exists.
    lrs_present = {g["initial_lr"] for g in optimizer.param_groups}
    assert lrs_present == {cfg.lr_encoder, cfg.lr_heads}
    for group in optimizer.param_groups:
        assert group["weight_decay"] in (0.0, cfg.weight_decay)
    # every group with weight_decay=0.0 must be non-empty and 1D-only
    no_decay_groups = [g for g in optimizer.param_groups if g["weight_decay"] == 0.0]
    assert all(len(g["params"]) > 0 for g in no_decay_groups)
    assert all(p.dim() < 2 for g in no_decay_groups for p in g["params"])


def test_configure_optimizers_returns_adamw_and_warmup_cosine_schedule():
    model = SupervisedFineTune(make_cfg(freeze_encoder=False))
    model.trainer = _FakeTrainer()
    result = model.configure_optimizers()
    assert isinstance(result["optimizer"], torch.optim.AdamW)
    assert isinstance(result["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    assert result["lr_scheduler"]["interval"] == "step"


def test_lr_is_zero_at_step_0_and_ramps_up_during_warmup():
    """Root cause fixed here, confirmed on a real run (mae_finetune_v3, step
    11,561): configure_optimizers used a flat, unscheduled AdamW -- zero LR
    warmup for the now-unfrozen encoder -- and node_emb (the encoder's own
    per-pulse output) degraded from a healthy 0.61 to 0.19 after 11,561
    steps, the same over-smoothing collapse fixed for MAE pretraining in
    799354b but never carried over here."""
    model = SupervisedFineTune(make_cfg(freeze_encoder=False))
    model.trainer = _FakeTrainer(global_step=0, estimated_stepping_batches=1000)
    result = model.configure_optimizers()
    optimizer = result["optimizer"]
    assert optimizer.param_groups[0]["lr"] == 0.0
    model.trainer.global_step = 25
    result["lr_scheduler"]["scheduler"].step()
    assert optimizer.param_groups[0]["lr"] > 0.0


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

    # pool_proj/pool_norm are always excluded from loading (see
    # build_supervised_model's comment -- they're untrained by both MAE and
    # JEPA pretraining, and pool_proj's shape can legitimately differ across
    # checkpoint eras), so they're compared separately as "must NOT match"
    # rather than included in the blanket equality loop below.
    for jepa_name, jepa_param in jepa_model.encoder.named_parameters():
        if jepa_name.startswith("pool_proj.") or jepa_name.startswith("pool_norm."):
            continue
        loaded_param = dict(supervised_model.encoder.named_parameters())[jepa_name]
        assert torch.equal(jepa_param, loaded_param)
    assert all(not p.requires_grad for p in supervised_model.encoder.parameters())


def test_cos_distance_loss_used_during_warmup_kappa_head_gets_no_gradient():
    """Root cause fixed by direction_warmup_steps, confirmed on a real run
    (mae_finetune_v1, step 74,301): raw vMF loss from a cold direction_head/
    kappa_head collapsed az/zen to an exact constant. During warmup, the
    loss must be pure cos-distance (kappa never enters the loss graph at
    all), so kappa_head gets literally zero gradient -- same mechanism as
    pool_proj legitimately getting no gradient during MAE pretraining.

    Calls _compute_loss directly, not training_step: self.log() (called
    inside training_step) needs a fully-real Trainer or none at all -- our
    minimal _FakeTrainer stub only satisfies the global_step read this
    warmup logic needs, and breaks self.log()'s own internals, same
    constraint noted in tests/models/test_jepa.py."""
    model = SupervisedFineTune(make_cfg(direction_warmup_steps=1000))
    model.trainer = _FakeTrainer(global_step=0)
    batch = make_labeled_batch(with_pid=False)
    n_events = int(batch.batch.max().item()) + 1
    g, az, zen, kappa = model._forward(batch, batch_size=n_events)
    loss = model._compute_loss(batch, g, az, zen, kappa)
    loss.backward()
    assert all(p.grad is None for p in model.kappa_head.parameters())
    assert all(p.grad is not None for p in model.direction_head.parameters())


def test_vmf_loss_used_after_warmup_kappa_head_gets_gradient():
    model = SupervisedFineTune(make_cfg(direction_warmup_steps=1000))
    model.trainer = _FakeTrainer(global_step=1000)
    batch = make_labeled_batch(with_pid=False)
    n_events = int(batch.batch.max().item()) + 1
    g, az, zen, kappa = model._forward(batch, batch_size=n_events)
    loss = model._compute_loss(batch, g, az, zen, kappa)
    loss.backward()
    assert all(p.grad is not None for p in model.kappa_head.parameters())


def test_direction_warmup_defaults_to_zero_when_cfg_lacks_the_field():
    """Backward compatibility: a cfg without direction_warmup_steps at all
    (e.g. an old config not yet updated) must behave exactly as before --
    immediate vMF loss, kappa_head gets gradient from step 0."""
    model = SupervisedFineTune(make_cfg())  # no direction_warmup_steps key
    assert not hasattr(model.cfg, "direction_warmup_steps")
    model.trainer = _FakeTrainer(global_step=0)
    batch = make_labeled_batch(with_pid=False)
    n_events = int(batch.batch.max().item()) + 1
    g, az, zen, kappa = model._forward(batch, batch_size=n_events)
    loss = model._compute_loss(batch, g, az, zen, kappa)
    loss.backward()
    assert all(p.grad is not None for p in model.kappa_head.parameters())


def test_compute_loss_without_trainer_attached_still_uses_vmf_loss():
    """Preserves every pre-existing test in this file that calls
    training_step directly without ever setting model.trainer -- accessing
    self.trainer with no real Trainer attached raises in Lightning, which
    must be caught and treated as global_step=None, i.e. past warmup."""
    model = SupervisedFineTune(make_cfg(direction_warmup_steps=1000))
    batch = make_labeled_batch(with_pid=False)
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    assert all(p.grad is not None for p in model.kappa_head.parameters())


def test_build_supervised_model_tolerates_a_checkpoint_missing_pool_norm(tmp_path):
    """Real production bug: encoder.pool_norm (models/pet.py) didn't exist
    when earlier MAE pretrain checkpoints (e.g. pretrain_mae_v6) were saved
    -- MAE pretraining calls encoder.forward() directly and never touches
    encode_event()/pool_proj/pool_norm at all. build_supervised_model's
    encoder load used to be strict, which made loading any such checkpoint
    crash with "Missing key(s): pool_norm.weight, pool_norm.bias" --
    confirmed directly, this broke both probe.py and finetune.py against a
    real v6 checkpoint. Simulates that exact old-format checkpoint here
    (PETEncoder's own state_dict with pool_norm keys stripped) and checks
    loading succeeds, leaving pool_norm at its fresh init. pool_proj is now
    excluded from loading unconditionally (see build_supervised_model), so
    this also confirms that a checkpoint's pool_proj -- even when present --
    is left alone rather than loaded."""
    from models.pet import PETEncoder

    encoder = PETEncoder(d=16, L=2, k=4)
    stale_state = {k: v for k, v in encoder.state_dict().items() if not k.startswith("pool_norm")}
    ckpt_path = tmp_path / "old_mae_ckpt.pt"
    torch.save({"state_dict": {f"encoder.{k}": v for k, v in stale_state.items()}}, ckpt_path)

    model = build_supervised_model(make_cfg(freeze_encoder=True), str(ckpt_path))

    for name, param in stale_state.items():
        if name.startswith("pool_proj."):
            continue
        assert torch.equal(param, model.encoder.state_dict()[name])
    # pool_norm wasn't in the checkpoint -- must still exist at its fresh,
    # untouched init (LayerNorm defaults: weight=1, bias=0), not crash.
    assert torch.equal(model.encoder.pool_norm.weight, torch.ones(16))
    assert torch.equal(model.encoder.pool_norm.bias, torch.zeros(16))
    # pool_proj WAS in the checkpoint (only pool_norm was stripped above),
    # but must NOT have been loaded -- always fresh-init, per
    # build_supervised_model's comment.
    assert not torch.equal(model.encoder.pool_proj.weight, stale_state["pool_proj.weight"])


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
