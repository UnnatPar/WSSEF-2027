import os
from datetime import timedelta

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from models.finetune import build_supervised_model
from train.checkpoints import latest_checkpoint
from train.config import flatten_sections, load_config
from train.dataset import build_dataloader, build_dataset

torch.set_float32_matmul_precision("high")

build_probe_model = build_supervised_model
build_finetune_model = build_supervised_model


def run_supervised(
    config_path: str, fast_dev_run: bool = False, early_stopping: bool = False,
) -> pl.Trainer:
    """Shared training loop for probe/finetune/scratch -- they differ only in
    whether a pretrain checkpoint is required and whether EarlyStopping runs.

    cfg.model.checkpoint (when present) is a directory, resolved to whatever
    epoch actually finished -- not a hardcoded filename, which would break if
    pretraining was interrupted early. Absent entirely for the from-scratch
    experiment (configs/scratch.yaml).
    """
    cfg = load_config(config_path)
    flat_cfg = flatten_sections(cfg, "model", "training")

    checkpoint_dir = getattr(cfg.model, "checkpoint", None)
    checkpoint_path = latest_checkpoint(checkpoint_dir) if checkpoint_dir is not None else None
    model = build_supervised_model(flat_cfg, checkpoint_path)

    num_workers = getattr(flat_cfg, "num_workers", 0)
    train_dataset = build_dataset(
        cfg.data.batch_dir, cfg.data.geometry_path, cfg.data.meta_path,
        cfg.data.train_batches, cfg.data.max_pulses, shuffle=True,
    )
    val_dataset = build_dataset(
        cfg.data.batch_dir, cfg.data.geometry_path, cfg.data.meta_path,
        cfg.data.val_batches, cfg.data.max_pulses,
    )
    train_loader = build_dataloader(train_dataset, batch_size=flat_cfg.batch_size, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=flat_cfg.batch_size, num_workers=num_workers)

    # See train/pretrain.py's build_trainer for why time-based (not epoch-based)
    # checkpointing matters: a real-scale epoch here can run for hours, well
    # past a Colab session's lifetime, so every_n_epochs=1 alone can lose an
    # entire epoch's progress to one session death.
    callbacks = [ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath, filename=cfg.checkpoint.filename,
        every_n_epochs=0, train_time_interval=timedelta(minutes=15),
        save_top_k=1, save_last=True,
    )]
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor="val/angular_error",
            patience=getattr(flat_cfg, "early_stopping_patience", 5),
            mode="min",
        ))

    kwargs = dict(
        max_epochs=flat_cfg.epochs,
        # See train/pretrain.py's build_trainer for the full story: fp16
        # ("16-mixed") verified by direct A/B overfit test to corrupt
        # gradients badly enough to turn clean monotonic loss descent into
        # noisy near-flat fluctuation, for this model specifically. bf16 fixed
        # it completely at no throughput cost on A100.
        precision="bf16-mixed",
        gradient_clip_val=flat_cfg.grad_clip, callbacks=callbacks,
    )
    if fast_dev_run:
        kwargs["fast_dev_run"] = True
        kwargs["accelerator"] = "cpu"
    else:
        # CSVLogger writes metrics.csv directly into checkpoint.dirpath (no
        # network access needed) -- eval/run_report.py reads it for the
        # training-curves figure. WandbLogger is kept alongside for the
        # live dashboard.
        kwargs["logger"] = [
            WandbLogger(project=cfg.logging.project, name=cfg.logging.run_name),
            CSVLogger(save_dir=cfg.checkpoint.dirpath, name="", version=""),
        ]
    trainer = pl.Trainer(**kwargs)
    # Auto-resume from a prior session's last checkpoint in this same run's dir.
    last_ckpt = os.path.join(cfg.checkpoint.dirpath, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    return trainer
