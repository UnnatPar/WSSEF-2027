import argparse
import math
import os
import sys

# Running this file directly (`python train/pretrain_mae.py`, exactly as
# documented in README.md / neutrinojepa.md / the generated Colab notebook)
# only puts this file's own directory (train/) on sys.path -- Python never
# adds the repo root for a directly-invoked script. Without this, the
# `from models...` / `from data...` imports below fail with
# `ModuleNotFoundError` in a fresh environment with no PYTHONPATH set (only
# masked previously by pytest's own rootdir insertion, or by callers that
# imported main() in-process instead of running this file as a script).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as pl
import torch
import torch.nn.functional as F

from models.heads import MAEHead
from models.pet import PETEncoder
from train.checkpoints import latest_checkpoint
from train.config import flatten_sections, load_config
from train.dataset import build_dataloader, build_dataset
from train.pretrain import build_trainer


def uniform_random_mask(n: int, ratio: float, device=None) -> torch.Tensor:
    """Per PolarBERT paper convention: 25% of pulses masked uniformly at
    random (distinct from data.masking.spatial_cluster_mask, which is
    JEPA-only).

    Accepts `device` so the training loop can build this directly on GPU --
    without it, the mask is built on CPU and every subsequent boolean-index
    use (`x[mask]`) implicitly transfers it and calls a synchronizing
    `nonzero()`, which showed up as ~50% of step time in profiling on a real
    L4 (see ledger 2026-07-28)."""
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    n_masked = int(ratio * n)
    if n_masked > 0:
        perm = torch.randperm(n, device=device)[:n_masked]
        mask[perm] = True
    return mask


class MAEPretrain(pl.LightningModule):
    """PET+MAE pre-training (experiment 1, spec's Model C). No PolarBERT
    branch -- that ablation is deferred."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = PETEncoder(cfg.d, cfg.L, cfg.k)
        self.mae_head = MAEHead(cfg.d)

    def _step(self, batch):
        """Shared forward pass for training_step and validation_step -- same
        masking + reconstruction loss, no dropout/BN in this model so there's
        nothing else that needs train/eval-mode branching."""
        x, batch_vec = batch.x, batch.batch
        n = x.shape[0]
        mask = uniform_random_mask(n, self.cfg.mask_ratio, device=x.device)
        # Boolean-index gather/scatter (`x[mask]`) calls a synchronizing
        # nonzero() internally every time it's used. Compute it once here
        # and reuse the integer indices instead of indexing by `mask` twice.
        mask_idx = mask.nonzero(as_tuple=True)[0]

        x_masked = x.clone()
        x_masked[mask_idx, 3:5] = 0.0  # zero out (t, q) at masked positions

        n_events = int(batch_vec.max().item()) + 1
        node_embeddings = self.encoder(x_masked, batch_vec, batch_size=n_events)
        pred_tq = self.mae_head(node_embeddings[mask_idx])
        true_tq = x[mask_idx, 3:5]

        loss = F.mse_loss(pred_tq, true_tq)
        return loss, n_events

    def training_step(self, batch, batch_idx):
        loss, n_events = self._step(batch)
        self.log("train/mae_loss", loss, batch_size=n_events)
        return loss

    def validation_step(self, batch, batch_idx):
        # Held out on the same [595, 627] batch range probe/finetune use as
        # their own val split (see configs/*.yaml) -- keeps [628, 660]
        # (eval/run_report.py's test set) untouched throughout the whole
        # pipeline, not just at the supervised stages. Random masking still
        # applies here (this model has no other stochasticity to disable),
        # so val/mae_loss carries the same masking-ratio noise as train/mae_loss
        # -- useful for spotting divergence between the two curves, not for a
        # noise-free absolute number.
        loss, n_events = self._step(batch)
        self.log("val/mae_loss", loss, batch_size=n_events, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        # Real production run: train/mae_loss dropped fast in the first
        # ~15k of 39,063 steps/epoch, then stayed flat for the next 135k+
        # steps across multiple epochs -- verified directly (not assumed) to
        # NOT be a precision artifact (fp32/fp16/bf16 all converge to the
        # same floor on an overfit-one-batch test) or a broken-gradient bug
        # (the same test shows a clean, fast 80% loss drop). With no LR
        # decay at all, AdamW at a constant step size reaches a basin fast
        # and then just oscillates in it indefinitely -- exactly this
        # symptom. `estimated_stepping_batches` is Lightning's own count of
        # the real total steps this run will actually take (accounts for
        # max_epochs, dataset size, and resume state), so T_max tracks the
        # actual training horizon instead of a step count that would need
        # to be hand-computed and kept in sync with the dataset/config.
        #
        # LambdaLR, NOT CosineAnnealingLR: verified directly against a real
        # checkpoint that resuming restores the scheduler's full saved
        # state_dict, including T_max, clobbering a freshly-computed T_max
        # right after construction -- an epochs config change (100 -> 8) had
        # zero effect on a resumed run because the checkpoint's stale
        # T_max=3,906,300 silently won every time.
        #
        # The lambda ignores the `step` argument LambdaLR passes it and
        # reads self.trainer.global_step instead -- verified directly
        # against the same checkpoint that the scheduler's own internal
        # counter does NOT reliably track true cumulative progress across
        # resumes either: global_step was 191,885 but the scheduler's own
        # last_epoch was only 12,501 (steps since that particular process's
        # scheduler was constructed, not since training began).
        # trainer.global_step is Lightning's single authoritative counter,
        # saved as its own top-level checkpoint key and correctly continued
        # across every resume regardless of any scheduler-internal
        # bookkeeping -- so computing progress from it directly is immune to
        # both failure modes at once (stale T_max AND a disconnected step
        # counter), not just the first one.
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(_):
            progress = min(self.trainer.global_step / total_steps, 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_model(flat_cfg) -> MAEPretrain:
    return MAEPretrain(flat_cfg)


def main(config_path: str, fast_dev_run: bool = False):
    cfg = load_config(config_path)
    flat_cfg = flatten_sections(cfg, "model", "training")

    model = build_model(flat_cfg)
    dataset = build_dataset(
        cfg.data.batch_dir, cfg.data.geometry_path, cfg.data.meta_path,
        cfg.data.train_batches, cfg.data.max_pulses, shuffle=True,
    )
    loader = build_dataloader(
        dataset, batch_size=flat_cfg.batch_size, num_workers=flat_cfg.num_workers,
    )
    # val_batches is optional (older configs / fast_dev_run's synthetic-data
    # tests may not define it) -- skip validation entirely rather than fail,
    # so this stays backward compatible with any config that predates this.
    #
    # num_workers=0 deliberately, NOT flat_cfg.num_workers: a persistent-worker
    # DataLoader (see build_dataloader) holds its workers alive for the whole
    # run, not just while validation is active. At production scale
    # (check_val_every_n_epoch=1, ~39k train steps/epoch) validation runs once
    # every couple hours, so there is no throughput reason to parallelize it --
    # but reusing num_workers=2 here doubled the process's persistent worker
    # count (2 train + 2 val, ~4.8GB RSS each observed on a real A100) and
    # crashed the very first resumed run of this feature within minutes, the
    # same OOM-killer signature (silent SIGKILL, `resource_tracker: leaked
    # semaphore objects`) as the original num_workers=8 bug. in-process
    # (num_workers=0) validation avoids the extra persistent workers entirely.
    val_loader = None
    val_batches = getattr(cfg.data, "val_batches", None)
    if val_batches is not None:
        val_dataset = build_dataset(
            cfg.data.batch_dir, cfg.data.geometry_path, cfg.data.meta_path,
            val_batches, cfg.data.max_pulses,
        )
        val_loader = build_dataloader(
            val_dataset, batch_size=flat_cfg.batch_size, num_workers=0,
        )
    trainer = build_trainer(
        flat_cfg, fast_dev_run=fast_dev_run, run_name=cfg.logging.run_name,
        project=cfg.logging.project,
        checkpoint_dirpath=cfg.checkpoint.dirpath, checkpoint_filename=cfg.checkpoint.filename,
    )
    # Auto-resume from a prior session's last checkpoint -- see train/pretrain.py's
    # build_trainer for why every_n_epochs=1 alone isn't enough for hours-long epochs.
    # Uses latest_checkpoint() (mtime-based), not a hardcoded "last.ckpt"
    # check -- see train/checkpoints.py for why a literal "last.ckpt" can go
    # stale (verified in this exact checkpoint dir: Lightning wrote
    # "last-v1.ckpt" instead of overwriting a pre-existing seeded "last.ckpt").
    try:
        ckpt_path = latest_checkpoint(cfg.checkpoint.dirpath)
    except FileNotFoundError:
        ckpt_path = None
    trainer.fit(model, loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_dev_run=args.fast_dev_run)
