"""Detailed torch.profiler breakdown for MAEPretrain.training_step on real
data/hyperparams, to find what dominates wall-clock at the current ~1.8 it/s
(post GPU-sync-stall fix). Not part of the training pipeline.

    PYTHONPATH=. python scripts/profile_step_detailed.py
"""
import torch
from torch.profiler import ProfilerActivity, profile
from types import SimpleNamespace

from train.dataset import build_dataloader, build_dataset
from train.pretrain_mae import MAEPretrain

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

dataset = build_dataset(
    "data/train", "data/sensor_geometry.csv", "data/train_meta.parquet",
    [1, 1], max_pulses=256, shuffle=True,
)
loader = build_dataloader(dataset, batch_size=256, num_workers=0)
it = iter(loader)

cfg = SimpleNamespace(d=256, L=6, k=8, mask_ratio=0.25, lr=1e-3, weight_decay=0.01)
model = MAEPretrain(cfg).to(device)
opt = model.configure_optimizers()

# warmup
for _ in range(5):
    batch = next(it).to(device)
    loss = model.training_step(batch, 0)
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()

batch = next(it).to(device)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(10):
        b = next(it).to(device)
        loss = model.training_step(b, 0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
print("\n=== self_cpu_time_total sort ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
print("\nPROFILE DETAIL DONE")
