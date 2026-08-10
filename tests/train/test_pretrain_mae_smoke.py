import math
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


def test_configure_optimizers_returns_a_decaying_cosine_schedule(tiny_pyg_batch, tmp_path):
    """Real production run: train/mae_loss dropped fast in the first ~15k of
    39,063 steps/epoch, then stayed flat for 135k+ more steps across
    multiple epochs -- because configure_optimizers previously returned a
    bare AdamW with no LR decay at all, so it reached a basin fast and then
    just oscillated in it indefinitely. Verified directly (not assumed) this
    wasn't a precision bug or broken gradients: fp32/fp16/bf16 all converge
    to the identical floor on an overfit-one-batch test, and that same test
    shows a clean, fast loss drop -- the optimizer mechanics work fine, they
    just never decayed. This test checks the schedule is wired up AND
    actually produces a lower LR after real steps, not just that a
    scheduler object exists."""
    from train.pretrain_mae import build_trainer

    model = MAEPretrain(make_flat_cfg())
    trainer = build_trainer(make_flat_cfg(), fast_dev_run=False, checkpoint_dirpath=str(tmp_path))
    trainer.fit_loop.epoch_loop.max_steps = 5  # enough real steps to see decay, not a full epoch
    trainer.limit_val_batches = 0

    loader = torch.utils.data.DataLoader(
        [tiny_pyg_batch] * 5, batch_size=None, collate_fn=lambda x: x,
    )

    trainer.fit(model, loader)
    optimizer = trainer.optimizers[0]
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr < model.cfg.lr


def test_lr_schedule_total_steps_survives_a_resume_with_a_smaller_epoch_budget():
    """Real, confirmed production bug: switching CosineAnnealingLR's T_max
    to reflect a corrected epoch budget (100 -> 8) had ZERO effect on a
    resumed run. Checked directly against a real checkpoint:
    load_state_dict() on resume restores the scheduler's *entire* saved
    state, including T_max -- so the freshly-computed, smaller T_max got
    silently overwritten by the stale value (3,906,300, from the old
    epochs=100 config) the instant the checkpoint was loaded. A config
    change to fix the schedule had no effect because the checkpoint always
    won.

    LambdaLR fixes this: PyTorch's LambdaLR.state_dict() explicitly excludes
    the lambda itself (only the step counter is saved/restored), so a
    closure-captured `total_steps` computed fresh in the current process can
    never be clobbered by an old checkpoint -- only the step position
    resumes, not the schedule shape. This test simulates exactly the bug
    scenario: a scheduler "trained" under a large total_steps (standing in
    for the old epochs=100), then its saved state loaded into a fresh
    scheduler built with a much smaller total_steps (standing in for the
    epochs=8 fix) -- the smaller total_steps must survive."""
    opt_a = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    large_total_steps = 1000

    def lambda_large(step):
        return 0.5 * (1 + math.cos(math.pi * min(step / large_total_steps, 1.0)))

    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lambda_large)
    for _ in range(10):
        sched_a.step()
    saved_state = sched_a.state_dict()

    opt_b = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    small_total_steps = 20  # stands in for the epochs=8 fix's much smaller horizon

    def lambda_small(step):
        return 0.5 * (1 + math.cos(math.pi * min(step / small_total_steps, 1.0)))

    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lambda_small)
    sched_b.load_state_dict(saved_state)  # what Lightning does on resume

    for _ in range(9):  # steps 10-18 of small_total_steps=20 -- deep into decay
        sched_b.step()
    lr_with_small_schedule = opt_b.param_groups[0]["lr"]

    # If total_steps had been clobbered back to 1000 (the bug), step ~19/1000
    # would be almost undecayed (~99.9% of initial lr). With the fix, step
    # ~19/20 is nearly fully decayed.
    assert lr_with_small_schedule < 1e-3 * 0.1


class _FakeTrainer:
    """Minimal stand-in exposing exactly what configure_optimizers reads."""

    def __init__(self, global_step, estimated_stepping_batches):
        self.global_step = global_step
        self.estimated_stepping_batches = estimated_stepping_batches


def test_lr_reflects_trainer_global_step_not_the_schedulers_own_counter():
    """A second, deeper problem the LambdaLR fix above doesn't cover on its
    own: checked directly against the same real checkpoint, the scheduler's
    own step counter did NOT track true cumulative progress across
    resumes -- global_step was 191,885 but the scheduler's own last_epoch
    was only 12,501 (steps since that particular process's scheduler object
    was constructed, not since training began). Relying on the `step`
    argument LambdaLR passes its lambda -- or on .step() call counts at
    all -- inherits this same disconnect. trainer.global_step is Lightning's
    single authoritative counter, saved as its own top-level checkpoint key
    and correctly continued across every resume regardless of any
    scheduler-internal bookkeeping. This test proves the lambda tracks
    global_step directly: with the scheduler's own counter frozen at 0
    (zero .step() calls made), just mutating global_step externally (as a
    real resume effectively does) must still move the LR.

    Compares against cfg.lr (the schedule's peak), not lr at global_step=0:
    warmup_cosine_lr_lambda deliberately makes lr=0.0 exactly at step 0 (see
    train/optim.py), so lr_at_step_0 is no longer a valid "near-full-LR"
    reference point now that warmup exists."""
    model = MAEPretrain(make_flat_cfg())
    model.trainer = _FakeTrainer(global_step=0, estimated_stepping_batches=1000)
    result = model.configure_optimizers()
    optimizer, scheduler = result["optimizer"], result["lr_scheduler"]["scheduler"]

    # Simulate a resume landing deep into the schedule -- mutate global_step
    # directly, WITHOUT ever calling scheduler.step(), exactly like a real
    # resume where the scheduler object is freshly constructed but
    # trainer.global_step already reflects real prior progress.
    model.trainer.global_step = 900
    scheduler.step()  # triggers one recomputation using the lambda
    lr_after_resume = optimizer.param_groups[0]["lr"]

    assert lr_after_resume < model.cfg.lr * 0.1  # deep into decay, not still ~full LR


def test_checkpoint_dirpath_and_filename_are_configurable(tmp_path):
    cfg = make_flat_cfg()
    trainer = build_trainer(
        cfg, fast_dev_run=True,
        checkpoint_dirpath=str(tmp_path / "ckpts"), checkpoint_filename="epoch{epoch}",
    )
    assert str(trainer.checkpoint_callback.dirpath) == str(tmp_path / "ckpts")
