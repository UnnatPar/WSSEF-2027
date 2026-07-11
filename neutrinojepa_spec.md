# NeutrinoJEPA — Codebase Specification

## Overview

Self-supervised pre-training of a Point-Edge Transformer (PET) encoder on IceCube neutrino
events using a JEPA (Joint-Embedding Predictive Architecture) objective, followed by multi-task
fine-tuning on direction reconstruction and track/cascade classification.

**Stack:** Python 3.11, PyTorch 2.3, PyTorch Geometric 2.5, PyTorch Lightning 2.3,
GraphNeT (data loading + DynEdge baseline), torch-ema (EMA update).

---

## What You Write vs. What You Import

The entire novel codebase is ~250 lines. Everything else is imported.

### Imported — zero custom code

| Component | Import |
|-----------|--------|
| Data loading, parquet parsing, geometry join, normalization | `graphnet.data.dataset.ParquetDataset` |
| PyG `Data` object construction and batching | `graphnet.data.dataset` internals |
| DynEdge baseline (Model A) | `graphnet.models.gnn.DynEdge` |
| Dynamic edge convolution primitive | `torch_geometric.nn.DynamicEdgeConv` |
| Global mean/max/sum pooling | `torch_geometric.nn.global_{mean,max,add}_pool` |
| Angular (Von Mises) loss | `graphnet.training.loss_functions.VonMisesFisher3DLoss` |
| EMA update for target encoder | `torch_ema.ExponentialMovingAverage` |
| Training loop, checkpointing, mixed precision | `lightning.LightningModule` + `lightning.Trainer` |
| W&B logging | `lightning.loggers.WandbLogger` |
| AdamW + cosine LR schedule | `torch.optim.AdamW` + `torch.optim.lr_scheduler.CosineAnnealingLR` |

### You write (~250 lines total)

| File | What | Lines |
|------|------|-------|
| `models/pet.py` | `PETBlock` + `PETEncoder` | ~70 |
| `models/jepa.py` | `NeutrinoJEPA` LightningModule | ~50 |
| `models/polarbert.py` | `PolarBERTEncoder` + MAE head (Model B ablation) | ~60 |
| `data/masking.py` | `spatial_cluster_mask` | ~30 |
| `models/heads.py` | `DirectionHead` + `ClassificationHead` | ~40 |

Everything else — data loading, losses, training loops, evaluation, logging — is imported
or is trivial glue (<5 lines per call site).

---

## Repository Layout

```
neutrinojepa/
├── data/
│   └── masking.py          # YOU WRITE: spatial cluster masking
├── models/
│   ├── pet.py              # YOU WRITE: PETBlock + PETEncoder
│   ├── jepa.py             # YOU WRITE: NeutrinoJEPA LightningModule
│   ├── polarbert.py        # YOU WRITE: PolarBERT encoder + MAE head (ablation B)
│   └── heads.py            # YOU WRITE: DirectionHead + ClassificationHead
├── train/
│   ├── pretrain.py         # ~20 lines: instantiate + trainer.fit()
│   ├── probe.py            # ~20 lines: freeze encoder + trainer.fit()
│   └── finetune.py         # ~20 lines: two param groups + trainer.fit()
├── eval/
│   └── metrics.py          # ~30 lines: angular error binned by energy
├── configs/
│   ├── pretrain.yaml
│   ├── probe.yaml
│   └── finetune.yaml
└── scripts/
    ├── download_data.sh
    └── run_ablations.sh
```

---

## Dependencies

```bash
# PyTorch + CUDA
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# PyG + graph ops
pip install torch-geometric torch-cluster torch-scatter \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# GraphNeT (data loading + DynEdge baseline)
pip install graphnet

# Training infra
pip install lightning torch-ema wandb

# Data
pip install pandas pyarrow numpy pyyaml tqdm
```

**Pinned versions:**
```
torch==2.3.0
torch-geometric==2.5.0
torch-cluster==1.6.3
torch-scatter==2.1.2
lightning==2.3.0
torch-ema==0.3.0
graphnet>=1.0
pandas==2.2.0
pyarrow==15.0.0
numpy==1.26.0
```

---

## Data

### Source

```bash
kaggle competitions download -c icecube-neutrinos-in-deep-ice
```

Structure after unzip:
```
train/                    # 660 parquet batch files, ~200k events each, 130M total
  batch_1.parquet ... batch_660.parquet
train_meta.parquet        # event_id, batch_id, first/last_pulse_index, azimuth, zenith
sensor_geometry.csv       # sensor_id → x, y, z (positions of all 5160 DOMs)
```

Each batch parquet columns: `event_id, sensor_id, time, charge, auxiliary`.

### Loading via GraphNeT

GraphNeT's `ParquetDataset` handles everything: parquet reading, geometry joining,
normalization, and PyG `Data` object construction. Use it directly.

```python
from graphnet.data.dataset import ParquetDataset

# GraphNeT produces PyG Data objects with:
#   data.x:       (N, 6)  [x, y, z, t, q, auxiliary] — already normalized
#   data.batch:   (N,)    batch index (handled by DataLoader)
# Normalization GraphNeT applies:
#   x,y,z: divided by 500.0
#   t:     min-subtracted per event, divided by 3e4
#   q:     log1p then divided by 3.0
#   aux:   float 0/1
```

**Splits:** Batches 1–594 = train, 595–627 = val, 628–660 = test. Never shuffle across.
**Pre-training subset:** Batches 1–50 (~10M events). Sufficient at d=256 scale.
**Fine-tuning subset:** Batches 1–250 (~50M events available; use 5M = ~25 batches).

### `data/masking.py` — Spatial Cluster Masking (YOU WRITE, ~30 lines)

**Why not random masking:** PolarBERT masks 25% of pulses uniformly at random. Adjacent
unmasked pulses trivially interpolate the masked ones. Spatial cluster masking forces the
model to predict entire unobserved detector sub-volumes — physically, an unseen section of
the Cherenkov cone — which requires understanding interaction geometry rather than local
interpolation.

```python
def spatial_cluster_mask(
    xyz: Tensor,        # (N, 3) DOM positions of real pulses only
    mask_ratio: float,  # fraction to mask, sampled per-batch from U[0.4, 0.6]
    n_clusters: int,    # number of spatial anchor points = 4
) -> tuple[BoolTensor, BoolTensor]:
    """
    Returns (context_mask, target_mask), each shape (N,), bool.
    context_mask & target_mask == False everywhere (no overlap).
    context_mask | target_mask == True everywhere (covers all real pulses).

    Algorithm:
        target = empty set
        for _ in range(n_clusters):
            anchor = random pulse not already in target
            sorted_neighbors = argsort(||xyz - xyz[anchor]||₂)
            add neighbors greedily until |target| >= mask_ratio * N
            (stop early if N already covered)
        context = all pulses not in target
    """
```

Called once per event inside the pre-training `training_step`. Operates on CPU before
moving to GPU — cheap at N≤256.

---

## Models

### `models/pet.py` — Point-Edge Transformer (YOU WRITE, ~70 lines)

#### Key imports

```python
from torch_geometric.nn import DynamicEdgeConv          # edge conv primitive
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
```

#### Design

`DynamicEdgeConv` from PyG already handles:
- Dynamic k-NN graph construction in current feature space (recomputed each layer)
- Edge feature computation: `MLP([x_i || x_i - x_j])` → edge embedding
- Aggregation over k neighbors

What you add on top: **attention weighting** over those k neighbors instead of plain
sum/max aggregation. This is the architectural novelty — edge-conditioned local attention.

#### `PETBlock` (~40 lines)

```python
class PETBlock(nn.Module):
    def __init__(self, d: int, k: int = 8):
        # edge_conv: DynamicEdgeConv with MLP(2d → d), produces edge embeddings
        self.edge_conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d)
            ),
            k=k,
            aggr='mean',    # temporary aggregation; overridden by attention below
        )
        # attention projections over neighbors
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        # FFN
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
```

**Implementation note on attention:** `DynamicEdgeConv` with `aggr='mean'` gives you
the edge-aggregated node features, but you need per-neighbor access for attention weights.
The cleanest approach: subclass `torch_geometric.nn.MessagePassing` directly for the
attention step, using the same knn graph that `DynamicEdgeConv` would build.
Concretely:

```python
# In PETBlock.forward(x, batch):
#   1. Build knn edge_index using torch_cluster.knn_graph(x, k, batch)
#   2. Run DynamicEdgeConv to get edge embeddings e_ij for all edges
#   3. For each node i, compute Q_i from x_i; K_ij, V_ij from (x_j + e_ij)
#   4. Softmax over j ∈ N(i) → weighted sum → x_i'
#   5. Residual + LN + FFN
```

Steps 3–5 are ~15 lines using scatter_softmax + scatter_sum from torch_scatter:
```python
from torch_scatter import scatter_softmax, scatter_sum

alpha = scatter_softmax((Q[edge_index[1]] * K).sum(-1) / d**0.5, edge_index[1])
x_prime = scatter_sum(alpha.unsqueeze(-1) * V, edge_index[1], dim=0, dim_size=N)
```

#### `PETEncoder` (~30 lines)

```python
class PETEncoder(nn.Module):
    def __init__(self, d: int = 256, L: int = 6, k: int = 8):
        self.input_proj = nn.Sequential(nn.Linear(6, d), nn.LayerNorm(d), nn.ReLU())
        self.blocks = nn.ModuleList([PETBlock(d, k) for _ in range(L)])
        self.pool_proj = nn.Linear(3 * d, d)   # 3-way: mean + max + sum

    def forward(self, x, batch):
        # x: (total_nodes, 6), batch: (total_nodes,)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, batch)
        return x   # (total_nodes, d) node-level embeddings

    def encode_event(self, x, batch):
        # returns (B, d) per-event global embedding
        x = self.forward(x, batch)
        g = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch),
        ], dim=-1)                     # (B, 3d)
        return self.pool_proj(g)       # (B, d)
```

Note: spec previously said 4-way aggregation (mean/max/min/sum). Dropped min because
`global_min_pool` doesn't exist in PyG — use 3-way (mean/max/sum) instead.
`pool_proj` input changes from 4d → 3d accordingly.

---

### `models/jepa.py` — NeutrinoJEPA LightningModule (YOU WRITE, ~50 lines)

#### Key imports

```python
import copy
import lightning as pl
from torch_ema import ExponentialMovingAverage
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from data.masking import spatial_cluster_mask
```

#### Structure

```python
class NeutrinoJEPA(pl.LightningModule):
    def __init__(self, cfg):
        self.encoder = PETEncoder(cfg.d, cfg.L, cfg.k)
        self.ema = ExponentialMovingAverage(
            self.encoder.parameters(), decay=cfg.ema_decay
        )
        # target encoder: shadow copy maintained by EMA, never directly optimized
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # predictor: global context summary + masked DOM xyz → target embedding
        self.predictor = nn.Sequential(
            nn.Linear(cfg.d + 3, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d)
        )

    def training_step(self, batch, batch_idx):
        x, batch_idx_vec = batch.x, batch.batch

        # 1. spatial cluster mask — per event
        context_idx, target_idx = [], []
        for ev in batch_idx_vec.unique():
            mask = batch_idx_vec == ev
            xyz = x[mask, :3]
            ratio = torch.empty(1).uniform_(self.cfg.ratio_min, self.cfg.ratio_max).item()
            ctx, tgt = spatial_cluster_mask(xyz, ratio, self.cfg.n_clusters)
            context_idx.append(mask.nonzero()[ctx])
            target_idx.append(mask.nonzero()[tgt])

        # 2. context encoding (gradients flow here)
        x_ctx = x.clone()
        x_ctx[torch.cat(target_idx)] = 0.0       # zero out target pulses
        z_ctx = self.encoder(x_ctx, batch_idx_vec)

        # 3. target encoding (no gradients, use EMA weights)
        with self.ema.average_parameters():       # temporarily swap to EMA weights
            with torch.no_grad():
                z_tgt = self.target_encoder(x, batch_idx_vec)

        # 4. predict: global context summary + target DOM xyz → predicted embedding
        g_ctx = self.encoder.encode_event(x_ctx, batch_idx_vec)  # (B, d)
        tgt_flat = torch.cat(target_idx)
        ev_of_tgt = batch_idx_vec[tgt_flat]
        pred_input = torch.cat([g_ctx[ev_of_tgt], x[tgt_flat, :3]], dim=-1)  # (M, d+3)
        pred = self.predictor(pred_input)         # (M, d)
        target = z_tgt[tgt_flat].detach()         # (M, d) stop-gradient

        # 5. normalized L2 loss (prevents collapse)
        pred_n   = F.normalize(pred,   dim=-1)
        target_n = F.normalize(target, dim=-1)
        loss = 2 - 2 * (pred_n * target_n).sum(-1).mean()

        self.log('train/loss', loss)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # EMA update happens after each optimizer step
        self.ema.update()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95)
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        return [opt], [sched]
```

**EMA note:** `torch_ema.ExponentialMovingAverage` maintains shadow parameters that track
the exponential moving average of the context encoder weights. `ema.average_parameters()`
is a context manager that temporarily swaps in the EMA weights — use this for target
encoding rather than maintaining a separate `target_encoder` module. This is simpler than
the manual EMA approach in the previous spec and avoids the deepcopy.

---

### `models/polarbert.py` — PolarBERT Encoder + MAE Head (YOU WRITE, ~60 lines)

Model B ablation. Reproduces PolarBERT (NeurIPS ML4PS 2024) using PyTorch's native
transformer — no custom code needed beyond the wrapper.

```python
class PolarBERTEncoder(nn.Module):
    """
    Standard transformer encoder, pre-layer norm, no positional encoding.
    d_model=256, n_layers=6, n_heads=4, ffn_dim=1024 (per paper).
    """
    def __init__(self, d: int = 256, n_layers: int = 6, n_heads: int = 4):
        self.input_proj = nn.Linear(6, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4*d,
            dropout=0.0, batch_first=True, norm_first=True,  # pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x, padding_mask=None):
        # x: (B, N, 6), padding_mask: (B, N) True=padding
        B = x.size(0)
        x = self.input_proj(x)                              # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                     # (B, N+1, d)
        if padding_mask is not None:
            # prepend False for CLS token (never masked)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x[:, 0]   # (B, d) CLS token = global embedding

class MAEHead(nn.Module):
    """Reconstructs (t, q) of masked DOMs from their node embeddings."""
    def __init__(self, d: int = 256):
        self.proj = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 2))

    def forward(self, node_embeddings):
        return self.proj(node_embeddings)   # (M, 2) → predicted (t, q)
```

PolarBERT pre-training loss: MSE between predicted and actual normalized (t, q) at
masked positions. Masking is uniform random at 25% (per paper), not spatial clusters.

---

### `models/heads.py` — Fine-tuning Heads (YOU WRITE, ~40 lines)

```python
class DirectionHead(nn.Module):
    """(B, d) → (B, 2): predicted (azimuth, zenith) in radians."""
    def __init__(self, d: int = 256):
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d//2), nn.GELU(),
            nn.Linear(d//2, 2)
        )

    def forward(self, x):
        out = self.mlp(x)                          # (B, 2)
        az  = 2 * torch.pi * torch.sigmoid(out[:, 0])   # [0, 2π]
        zen = torch.pi  * torch.sigmoid(out[:, 1])       # [0, π]
        return az, zen

class ClassificationHead(nn.Module):
    """(B, d) → (B, 1): logit for track (1) vs cascade (0)."""
    def __init__(self, d: int = 256):
        self.mlp = nn.Sequential(
            nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 1)
        )

    def forward(self, x):
        return self.mlp(x)   # (B, 1) raw logit, use BCEWithLogitsLoss
```

---

## Losses

### Angular direction loss — imported

```python
from graphnet.training.loss_functions import VonMisesFisher3DLoss
loss_fn = VonMisesFisher3DLoss()
# Expects (B, 3) Cartesian unit vectors, not (az, zen)
# Convert: v = [sin(zen)*cos(az), sin(zen)*sin(az), cos(zen)]
```

### JEPA latent loss — inline (3 lines, no separate file needed)

```python
pred_n   = F.normalize(pred,   dim=-1)
target_n = F.normalize(target, dim=-1)
loss     = 2 - 2 * (pred_n * target_n).sum(-1).mean()
```

Normalized cosine distance. Equivalent to normalized L2. Prevents collapse without
needing negative samples or separate regularization terms.

### Fine-tuning combined loss — inline

```python
loss = (
    cfg.lambda_direction     * direction_loss_fn(pred_vec, true_vec) +
    cfg.lambda_classification * F.binary_cross_entropy_with_logits(pred_cls, labels)
)
```

---

## Training

### Pre-training (`train/pretrain.py`, ~20 lines)

```python
import lightning as pl
from lightning.loggers import WandbLogger

model   = NeutrinoJEPA(cfg)
dataset = ParquetDataset(...)   # graphnet
loader  = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

trainer = pl.Trainer(
    max_epochs=cfg.epochs,
    precision='16-mixed',
    gradient_clip_val=cfg.grad_clip,
    logger=WandbLogger(project='neutrinojepa', name=cfg.run_name),
    callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=1, save_top_k=-1)],
)
trainer.fit(model, loader)
```

### Linear probing (`train/probe.py`, ~20 lines)

Load checkpoint. Freeze encoder. Attach heads. Train heads only.

```python
model = NeutrinoJEPA.load_from_checkpoint(cfg.checkpoint)
for p in model.encoder.parameters():
    p.requires_grad_(False)
model.direction_head     = DirectionHead(cfg.d)
model.classification_head = ClassificationHead(cfg.d)
trainer.fit(model, train_loader, val_loader)
```

### Fine-tuning (`train/finetune.py`, ~20 lines)

Two parameter groups: encoder at lower lr, heads at higher lr.

```python
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(),       'lr': cfg.lr_encoder},
    {'params': model.direction_head.parameters(), 'lr': cfg.lr_heads},
    {'params': model.classification_head.parameters(), 'lr': cfg.lr_heads},
], weight_decay=cfg.weight_decay)
```

`EarlyStopping` callback on `val/angular_error`, patience=5.

---

## Evaluation (`eval/metrics.py`, ~30 lines)

Angular error by energy is the only non-trivial metric. Everything else is sklearn.

```python
from sklearn.metrics import roc_auc_score

def mean_angular_error(pred_az, pred_zen, true_az, true_zen) -> float:
    """Great-circle distance in degrees. Identical to Kaggle metric."""
    def to_cart(az, zen):
        return torch.stack([
            torch.sin(zen)*torch.cos(az),
            torch.sin(zen)*torch.sin(az),
            torch.cos(zen)
        ], dim=-1)
    dot = (to_cart(pred_az, pred_zen) * to_cart(true_az, true_zen)).sum(-1).clamp(-1+1e-7, 1-1e-7)
    return torch.acos(dot).mean().item() * 180 / torch.pi

def angular_error_by_energy(pred_az, pred_zen, true_az, true_zen, log_energies) -> dict:
    """
    Bin events into 10 log10-energy bins across [2, 7].
    Returns {bin_center: mean_angular_error}.
    Match binning to PolarBERT paper Fig 2 for direct comparison.
    """
    bins = torch.linspace(2, 7, 11)
    result = {}
    for i in range(10):
        mask = (log_energies >= bins[i]) & (log_energies < bins[i+1])
        if mask.sum() > 0:
            result[float((bins[i] + bins[i+1]) / 2)] = mean_angular_error(
                pred_az[mask], pred_zen[mask], true_az[mask], true_zen[mask]
            )
    return result
```

---

## Ablation Models

All 4 share identical data splits, loss functions, and eval code. Only encoder + pre-training
objective differ.

| Model | Encoder | Pre-training | Custom code |
|-------|---------|-------------|-------------|
| A: DynEdge | `graphnet.models.gnn.DynEdge` | None (supervised) | Zero — fully imported |
| B: PolarBERT | `PolarBERTEncoder` (you write) | MAE input-space | ~60 lines |
| C: PET + MAE | `PETEncoder` (you write) | MAE input-space | Pet.py + MAEHead |
| D: NeutrinoJEPA | `PETEncoder` (you write) | JEPA latent-space | Full system |

Model C reuses `PETEncoder` from `pet.py` and `MAEHead` from `polarbert.py`.
No additional files needed.

---

## Configs

### `configs/pretrain.yaml`
```yaml
model:
  d: 256
  L: 6
  k: 8
  ema_decay: 0.996

masking:
  ratio_min: 0.4
  ratio_max: 0.6
  n_clusters: 4

training:
  batch_size: 256
  lr: 1.0e-3
  weight_decay: 0.05
  epochs: 100
  grad_clip: 1.0
  n_events: 10_000_000
  num_workers: 8

data:
  max_pulses: 256
  batch_dir: data/train/
  meta_path: data/train_meta.parquet
  geometry_path: data/sensor_geometry.csv
  train_batches: [1, 50]

logging:
  project: neutrinojepa
  run_name: pretrain_v1
```

### `configs/finetune.yaml`
```yaml
model:
  checkpoint: checkpoints/pretrain_v1_epoch100.pt
  freeze_encoder: false

training:
  batch_size: 256
  lr_encoder: 1.0e-4
  lr_heads: 1.0e-3
  weight_decay: 0.05
  epochs: 50
  warmup_epochs: 5
  grad_clip: 1.0
  n_events: 5_000_000
  lambda_direction: 1.0
  lambda_classification: 0.5
  early_stopping_patience: 5

data:
  max_pulses: 256
  train_batches: [1, 594]
  val_batches: [595, 627]
  test_batches: [628, 660]
```

---

## Key Implementation Notes

**PyG vs. padded tensors:** GraphNeT produces PyG `Data` objects with flattened node
tensors and a `batch` index vector — not padded (B, N, d) tensors. `PETBlock` and
`PETEncoder` operate in this format natively. Do not convert to padded tensors;
it breaks `knn_graph` and `scatter_*` operations.

**EMA via torch_ema:** `ExponentialMovingAverage.average_parameters()` is a context
manager that temporarily replaces the model's parameters with their EMA values. Use it
only for target encoding (forward pass). The context encoder always uses raw (non-EMA)
weights during forward and backward.

**scatter_softmax + scatter_sum for attention:**
```python
from torch_scatter import scatter_softmax, scatter_sum
# edge_index: (2, E), row=source j, col=target i (flow='source_to_target')
scores = (Q[edge_index[1]] * K).sum(-1) / d**0.5   # (E,)
alpha  = scatter_softmax(scores, edge_index[1], dim=0, dim_size=N)  # (E,)
x_attn = scatter_sum(alpha.unsqueeze(-1) * V, edge_index[1], dim=0, dim_size=N)  # (N, d)
```

**Mixed precision:** Handled entirely by `pl.Trainer(precision='16-mixed')`.
No manual `autocast` or `GradScaler` needed.

**Checkpointing:** `ModelCheckpoint(every_n_epochs=1, save_top_k=-1)` saves all epochs.
Lightning saves optimizer, scheduler, and epoch state automatically. EMA state is saved
as part of the LightningModule's `state_dict` if you register it as a buffer or handle
it in `on_save_checkpoint` / `on_load_checkpoint`.

**knn_graph flow convention:** Use `flow='source_to_target'` consistently.
`edge_index[0]` = source (neighbor j), `edge_index[1]` = target (node i being updated).
This matches PyG's `MessagePassing` convention.
