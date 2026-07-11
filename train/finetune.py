import argparse

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from models.finetune import SupervisedFineTune, build_supervised_model
from train.config import flatten_sections, load_config
from train.data import build_dataloader, build_dataset


def build_finetune_model(flat_cfg, checkpoint_path: str | None) -> SupervisedFineTune:
    return build_supervised_model(flat_cfg, checkpoint_path)


def main(config_path: str, fast_dev_run: bool = False):
    cfg = load_config(config_path)
    flat_cfg = flatten_sections(cfg, "model", "training")

    checkpoint_path = getattr(cfg.model, "checkpoint", None)
    model = build_finetune_model(flat_cfg, checkpoint_path)

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

    early_stop = EarlyStopping(
        monitor="val/angular_error",
        patience=getattr(flat_cfg, "early_stopping_patience", 5),
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath, filename=cfg.checkpoint.filename,
        every_n_epochs=1, save_top_k=-1,
    )
    kwargs = dict(
        max_epochs=flat_cfg.epochs, precision="16-mixed",
        gradient_clip_val=flat_cfg.grad_clip, callbacks=[early_stop, checkpoint_callback],
    )
    if fast_dev_run:
        kwargs["fast_dev_run"] = True
        kwargs["accelerator"] = "cpu"
    else:
        # CSVLogger writes metrics.csv directly into checkpoint.dirpath (no
        # network access needed) -- eval/run_report.py reads it for the
        # training-curves figure. WandbLogger is kept alongside it for the
        # live dashboard.
        kwargs["logger"] = [
            WandbLogger(project="neutrinojepa", name=cfg.logging.run_name),
            CSVLogger(save_dir=cfg.checkpoint.dirpath, name="", version=""),
        ]
    trainer = pl.Trainer(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_dev_run=args.fast_dev_run)
