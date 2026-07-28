import argparse

import lightning as pl
import torch
import torch.nn.functional as F

from models.heads import MAEHead
from models.pet import PETEncoder
from train.config import flatten_sections, load_config
from train.data import build_dataloader, build_dataset
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

    def training_step(self, batch, batch_idx):
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
        self.log("train/mae_loss", loss, batch_size=n_events)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )


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
    trainer = build_trainer(
        flat_cfg, fast_dev_run=fast_dev_run, run_name=cfg.logging.run_name,
        project=cfg.logging.project,
        checkpoint_dirpath=cfg.checkpoint.dirpath, checkpoint_filename=cfg.checkpoint.filename,
    )
    trainer.fit(model, loader)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_dev_run=args.fast_dev_run)
