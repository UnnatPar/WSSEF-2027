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


def test_checkpoint_dirpath_and_filename_are_configurable(tmp_path):
    cfg = make_flat_cfg()
    trainer = build_trainer(
        cfg, fast_dev_run=True,
        checkpoint_dirpath=str(tmp_path / "ckpts"), checkpoint_filename="epoch{epoch}",
    )
    assert str(trainer.checkpoint_callback.dirpath) == str(tmp_path / "ckpts")
