import argparse

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from models.finetune import SupervisedFineTune, build_supervised_model
from train.checkpoints import latest_checkpoint
from train.config import flatten_sections, load_config
from train.data import build_dataloader, build_dataset


def build_probe_model(flat_cfg, checkpoint_path: str) -> SupervisedFineTune:
    return build_supervised_model(flat_cfg, checkpoint_path)


def main(config_path: str, fast_dev_run: bool = False):
    cfg = load_config(config_path)
    flat_cfg = flatten_sections(cfg, "model", "training")

    # cfg.model.checkpoint is a directory (e.g. checkpoints/pretrain_jepa_v1/),
    # resolved to whatever epoch actually finished -- not a hardcoded
    # filename, which would break if pretraining was interrupted early.
    checkpoint_path = latest_checkpoint(cfg.model.checkpoint)
    model = build_probe_model(flat_cfg, checkpoint_path)

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

    kwargs = dict(
        max_epochs=flat_cfg.epochs, precision="16-mixed", gradient_clip_val=flat_cfg.grad_clip,
        callbacks=[ModelCheckpoint(
            dirpath=cfg.checkpoint.dirpath, filename=cfg.checkpoint.filename,
            every_n_epochs=1, save_top_k=-1,
        )],
    )
    if fast_dev_run:
        kwargs["fast_dev_run"] = True
        kwargs["accelerator"] = "cpu"
    else:
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
