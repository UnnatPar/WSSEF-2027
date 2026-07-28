"""Throughput benchmark for MAEPretrain and NeutrinoJEPA training_step on
real data, using the actual model classes and the real dataloader config
(num_workers=8, matching configs/*.yaml). Not part of the training
pipeline -- a standing tool for re-checking GPU throughput after future
changes to the encoder, masking, or dataloader.

Requires a local (or VM) copy of data/train/batch_1.parquet,
data/sensor_geometry.csv, data/train_meta.parquet (see
scripts/download_data.sh). Run with PYTHONPATH set to the repo root:

    PYTHONPATH=. python scripts/profile_step.py
"""
import time
from types import SimpleNamespace

import torch

from models.jepa import NeutrinoJEPA
from train.dataset import build_dataloader, build_dataset
from train.pretrain_mae import MAEPretrain

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")


def make_loader():
    dataset = build_dataset(
        "data/train", "data/sensor_geometry.csv", "data/train_meta.parquet",
        [1, 1], max_pulses=256, shuffle=True,
    )
    return build_dataloader(dataset, batch_size=256, num_workers=8)


def bench(name, model, opt, step_fn, n_warmup=5, n_steps=40):
    loader = make_loader()
    it = iter(loader)
    for _ in range(n_warmup):
        batch = next(it).to(device)
        loss = step_fn(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        batch = next(it).to(device)
        loss = step_fn(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"RESULT {name}: {n_steps} steps in {t1-t0:.2f}s -> {n_steps/(t1-t0):.3f} it/s")
    del it, loader


if __name__ == "__main__":
    mae_cfg = SimpleNamespace(d=256, L=6, k=8, mask_ratio=0.25, lr=1e-3, weight_decay=0.01)
    mae_model = MAEPretrain(mae_cfg).to(device)
    mae_opt = mae_model.configure_optimizers()
    bench("MAE pretrain", mae_model, mae_opt, lambda m, b: m.training_step(b, 0))

    jepa_cfg = SimpleNamespace(
        d=256, L=6, k=8, ema_decay=0.99, ratio_min=0.4, ratio_max=0.6,
        n_clusters=4, lr=1e-3, weight_decay=0.01, epochs=100,
    )
    jepa_model = NeutrinoJEPA(jepa_cfg).to(device)
    jepa_opt = jepa_model.configure_optimizers()[0][0]
    bench("JEPA pretrain", jepa_model, jepa_opt, lambda m, b: m.training_step(b, 0))
