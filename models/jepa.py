import random

import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_ema import ExponentialMovingAverage

from data.masking import spatial_cluster_mask
from models.pet import PETEncoder
from train.optim import make_param_groups, warmup_cosine_lr_lambda


class NeutrinoJEPA(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = PETEncoder(cfg.d, cfg.L, cfg.k)
        self.ema = ExponentialMovingAverage(
            self.encoder.parameters(), decay=cfg.ema_decay
        )
        # predictor: global context summary + masked DOM xyz -> predicted target embedding
        self.predictor = nn.Sequential(
            nn.Linear(cfg.d + 3, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d)
        )

    def _step(self, batch):
        """Shared forward pass for training_step and validation_step."""
        x, batch_vec = batch.x, batch.batch

        # PyG batching lays out each event's nodes contiguously (sorted by
        # event), so per-event slices are exact offset ranges -- no need for
        # a boolean event_mask + nonzero() per event. Offsets are pulled to
        # CPU with a single .tolist() (one sync total) instead of one
        # implicit sync per event via slice-bound .item() conversion; the
        # mask ratio is drawn with plain Python `random` instead of a GPU
        # tensor + .item(). Previously this loop (up to 256 events/step) did
        # 2+ synchronizing GPU round trips per event -- profiling on a real
        # L4 showed this class of sync stall dominating step time (see
        # ledger 2026-07-28).
        counts = torch.bincount(batch_vec)
        offsets = torch.zeros(counts.numel() + 1, dtype=torch.long, device=x.device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        offsets = offsets.tolist()
        n_events = len(offsets) - 1  # free -- offsets is already synced above

        # Build one whole-batch boolean target mask (slice-assigning each
        # event's bool tensor -- a plain copy, not an index op) instead of
        # converting each event's mask to indices via boolean-mask indexing
        # (`idx[tgt]`), which calls a synchronizing nonzero() once per
        # event. That collapses up to 256 nonzero() calls/step into 1.
        # (`context` was computed here too but never used below -- dropped.)
        n_total = x.shape[0]
        target_mask = torch.zeros(n_total, dtype=torch.bool, device=x.device)
        for ev in range(len(offsets) - 1):
            start, end = offsets[ev], offsets[ev + 1]
            xyz = x[start:end, :3]
            ratio = random.uniform(self.cfg.ratio_min, self.cfg.ratio_max)
            _, tgt = spatial_cluster_mask(xyz, ratio, self.cfg.n_clusters)
            target_mask[start:end] = tgt

        target_flat = target_mask.nonzero(as_tuple=True)[0]

        x_ctx = x.clone()
        x_ctx[target_mask] = 0.0

        # Target encoding uses self.encoder with EMA-swapped weights, not a
        # separate target_encoder module (the spec's literal snippet keeps
        # both, which is contradictory -- a deepcopy target_encoder never
        # gets updated by ema.average_parameters(), which only swaps
        # self.encoder's own parameters).
        with self.ema.average_parameters():
            with torch.no_grad():
                z_tgt = self.encoder(x, batch_vec, batch_size=n_events)

        g_ctx = self.encoder.encode_event(x_ctx, batch_vec, batch_size=n_events)
        ev_of_tgt = batch_vec[target_flat]
        pred_input = torch.cat([g_ctx[ev_of_tgt], x[target_flat, :3]], dim=-1)
        pred = self.predictor(pred_input)
        target = z_tgt[target_flat].detach()

        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        loss = 2 - 2 * (pred_n * target_n).sum(-1).mean()
        return loss, n_events

    def training_step(self, batch, batch_idx):
        loss, n_events = self._step(batch)
        self.log("train/loss", loss, batch_size=n_events)
        return loss

    def validation_step(self, batch, batch_idx):
        # Lightning disables grad + calls self.eval() around this automatically;
        # self.ema.average_parameters() below still works the same under
        # no_grad(). See train/pretrain_mae.py's validation_step for why this
        # reuses the same [595, 627] range other stages already treat as val.
        loss, n_events = self._step(batch)
        self.log("val/loss", loss, batch_size=n_events, on_epoch=True, on_step=False)
        return loss

    def on_before_zero_grad(self, optimizer):
        # Must run after optimizer.step(), not on_before_optimizer_step
        # (which fires before the step and would leave the EMA shadow one
        # step stale).
        self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        # torch_ema's state isn't an nn.Module, so it's invisible to
        # Lightning's automatic state_dict. Deliberately excludes
        # collected_params: torch_ema's store() (used internally by
        # average_parameters(), which training_step calls every step) clones
        # live model parameters without torch.no_grad()/.detach(), leaving
        # non-leaf tensors permanently in self.ema.collected_params -- which
        # breaks state_dict()'s deepcopy on any later checkpoint save.
        # collected_params is just store()/restore() scratch space, not real
        # EMA state, so it's safe and necessary to drop it here.
        checkpoint["ema_state"] = {
            "decay": self.ema.decay,
            "num_updates": self.ema.num_updates,
            "shadow_params": self.ema.shadow_params,
            "collected_params": None,
        }

    def on_load_checkpoint(self, checkpoint):
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])

    def configure_optimizers(self):
        # LambdaLR reading self.trainer.global_step directly, not
        # CosineAnnealingLR(T_max=cfg.epochs) with Lightning's default
        # per-epoch stepping -- same fix as train/pretrain_mae.py's
        # MAEPretrain.configure_optimizers, applied here proactively before
        # this experiment's first real run (not yet exercised in production,
        # unlike MAE pretrain where this exact pattern was confirmed to
        # silently neuter the LR schedule: T_max=cfg.epochs paired with
        # multi-hour epochs meant the schedule barely moved across any
        # realistically observable training window, and a resumed run's
        # checkpoint-restored scheduler state further clobbered any
        # config fix). See that file for the full reasoning and empirical
        # verification of both failure modes.
        # make_param_groups, NOT plain self.parameters(): PETEncoder-based
        # models were confirmed to suffer a severe representation collapse
        # from weight_decay being applied to LayerNorm's gain with no
        # opposing gradient -- see train/pretrain_mae.py's
        # configure_optimizers and train/optim.py for the full, precisely
        # confirmed story (gain decay matched exp(-lr*weight_decay*steps)
        # almost exactly in a real checkpoint). Applied here proactively
        # since NeutrinoJEPA shares the same PETEncoder and the same
        # weight_decay-on-everything pattern, even though this experiment
        # hasn't run in production yet this session.
        # warmup_cosine_lr_lambda, NOT a bare cosine schedule: applied here
        # proactively for the same reason make_param_groups was -- confirmed
        # in train/pretrain_mae.py's sibling MAEPretrain (same PETEncoder)
        # that a bare cosine schedule with no warmup lets AdamW's first few
        # steps (bias-corrected second moment ~1000x amplified with no
        # warmup) collapse node embedding diversity from healthy (~0.68) to
        # near-zero (~0.004) within 20 steps, independent of and prior to
        # the separate LayerNorm-gain-decay collapse make_param_groups
        # fixes. See train/optim.py for the full mechanism and the
        # controlled A/B that confirmed it.
        opt = torch.optim.AdamW(
            make_param_groups(self, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay),
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(1000, total_steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, warmup_cosine_lr_lambda(self.trainer, total_steps, warmup_steps)
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
