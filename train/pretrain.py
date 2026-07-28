import argparse

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from models.jepa import NeutrinoJEPA
from train.config import flatten_sections, load_config
from train.data import build_dataloader, build_dataset

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
        callbacks=[ModelCheckpoint(
            dirpath=checkpoint_dirpath, filename=checkpoint_filename,
            every_n_epochs=1, save_top_k=-1,
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
    trainer.fit(model, loader)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_dev_run=args.fast_dev_run)
