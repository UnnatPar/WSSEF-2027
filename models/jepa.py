import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_ema import ExponentialMovingAverage

from data.masking import spatial_cluster_mask
from models.pet import PETEncoder


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

    def training_step(self, batch, batch_idx):
        x, batch_vec = batch.x, batch.batch

        context_idx, target_idx = [], []
        for ev in batch_vec.unique():
            event_mask = batch_vec == ev
            xyz = x[event_mask, :3]
            ratio = torch.empty(1, device=x.device).uniform_(
                self.cfg.ratio_min, self.cfg.ratio_max
            ).item()
            ctx, tgt = spatial_cluster_mask(xyz, ratio, self.cfg.n_clusters)
            idx = event_mask.nonzero(as_tuple=True)[0]
            context_idx.append(idx[ctx])
            target_idx.append(idx[tgt])

        target_flat = torch.cat(target_idx)

        x_ctx = x.clone()
        x_ctx[target_flat] = 0.0

        # Target encoding uses self.encoder with EMA-swapped weights, not a
        # separate target_encoder module (the spec's literal snippet keeps
        # both, which is contradictory -- a deepcopy target_encoder never
        # gets updated by ema.average_parameters(), which only swaps
        # self.encoder's own parameters).
        with self.ema.average_parameters():
            with torch.no_grad():
                z_tgt = self.encoder(x, batch_vec)

        g_ctx = self.encoder.encode_event(x_ctx, batch_vec)
        ev_of_tgt = batch_vec[target_flat]
        pred_input = torch.cat([g_ctx[ev_of_tgt], x[target_flat, :3]], dim=-1)
        pred = self.predictor(pred_input)
        target = z_tgt[target_flat].detach()

        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        loss = 2 - 2 * (pred_n * target_n).sum(-1).mean()

        n_events = int(batch_vec.max().item()) + 1
        self.log("train/loss", loss, batch_size=n_events)
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
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        return [opt], [sched]
