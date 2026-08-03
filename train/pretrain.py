import argparse
import os
import sys
from datetime import timedelta

# See train/pretrain_mae.py for why this is needed: a directly-invoked
# script only gets its own directory on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from models.jepa import NeutrinoJEPA
from train.config import flatten_sections, load_config
from train.dataset import build_dataloader, build_dataset

# Tensor Cores on Ampere+ (A100, L4) go unused at the default "highest"
# setting -- this trades a rounding-error's worth of matmul precision
# (irrelevant next to bf16/fp16 AMP, which is already coarser) for real
# throughput. Set once per process, before any CUDA tensor is created.
torch.set_float32_matmul_precision("high")


def build_model(flat_cfg) -> NeutrinoJEPA:
    return NeutrinoJEPA(flat_cfg)


def build_trainer(
    flat_cfg,
    fast_dev_run: bool = False,
    run_name: str = "pretrain",
    project: str = "neutrinojepa",
    checkpoint_dirpath: str | None = None,
    checkpoint_filename: str | None = None,
) -> pl.Trainer:
    kwargs = dict(
        max_epochs=flat_cfg.epochs,
        precision="16-mixed",
        gradient_clip_val=flat_cfg.grad_clip,
        # Pretrain epochs (10M events / batch_size) run for hours -- far longer
        # than a Colab session tends to survive. Checkpointing only at epoch
        # boundaries (the old every_n_epochs=1) would mean a session death
        # loses the entire epoch's progress. train_time_interval saves on wall
        # clock instead, so a killed session never loses more than ~15 min of
        # work; save_last=True always keeps a stable "last.ckpt" pointer for
        # resume regardless of which of the last save_top_k files that is.
        callbacks=[ModelCheckpoint(
            dirpath=checkpoint_dirpath, filename=checkpoint_filename,
            every_n_epochs=0, train_time_interval=timedelta(minutes=15),
            save_top_k=1, save_last=True,
        )],
    )
    if fast_dev_run:
        kwargs["fast_dev_run"] = True
        kwargs["accelerator"] = "cpu"
    else:
        kwargs["logger"] = WandbLogger(project=project, name=run_name)
    return pl.Trainer(**kwargs)


def main(config_path: str, fast_dev_run: bool = False):
    cfg = load_config(config_path)
    flat_cfg = flatten_sections(cfg, "model", "masking", "training")

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
    # Auto-resume: a fresh Colab session re-running this exact command should
    # continue from wherever the last session's ModelCheckpoint left off, not
    # restart from step 0 -- last.ckpt restores model, optimizer, and the
    # global step/epoch counters.
    last_ckpt = os.path.join(cfg.checkpoint.dirpath, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
    trainer.fit(model, loader, ckpt_path=ckpt_path)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_dev_run=args.fast_dev_run)
