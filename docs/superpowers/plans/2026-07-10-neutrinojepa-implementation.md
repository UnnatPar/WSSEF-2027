# NeutrinoJEPA Implementation Steps

Run these steps in order in a Claude Code session, one at a time, with tests before/after each. TDD where the component is a pure function/module; smoke tests (shape/gradient/finite-loss checks on tiny synthetic tensors) where it's a LightningModule or training loop. Commit after each step.

## Goal

Three experiments isolating pre-training objective, same PET encoder architecture throughout:
1. **NeutrinoPET** — PET encoder pre-trained via MAE → probe (frozen) → fine-tune (unfrozen)
2. **NeutrinoJEPAPET** — PET encoder pre-trained via JEPA (the spec's actual contribution) → probe → fine-tune
3. **PET+heads from scratch** — encoder + heads trained together, random init, no freezing — the baseline

DynEdge and PolarBERT+MAE (the spec's other two ablations) are out of scope — different encoder architecture, not part of these 3 experiments. Defer to a follow-up plan.

## Repo layout

`data/`, `models/`, `train/`, `eval/`, `configs/`, `scripts/`, `tests/` (mirroring) as top-level importable packages at repo root — no `src/` layout, matches the spec's own import style (`from data.masking import ...`).

## Non-obvious gotchas — read before implementing, don't rediscover these mid-build

- **EMA target encoding**: spec's `jepa.py` snippet keeps a `target_encoder` deepcopy AND uses `ema.average_parameters()` — contradictory, and the deepcopy never gets updated. Drop `target_encoder` entirely; target encoding = call `self.encoder(...)` inside `ema.average_parameters()`.
- **`DynamicEdgeConv` doesn't expose per-edge embeddings** — PyG's version only returns aggregated node output. Build the knn graph via `torch_cluster.knn_graph` directly and compute the edge MLP per-edge yourself; don't instantiate `DynamicEdgeConv`.
- **EMA update timing**: `ema.update()` must run in `on_before_zero_grad`, not `on_before_optimizer_step` (which fires *before* the optimizer step — one-step-stale otherwise).
- **EMA state isn't in the default checkpoint** (`torch_ema` isn't an `nn.Module`) — persist it explicitly via `on_save_checkpoint`/`on_load_checkpoint`.
- **Pooling is 3-way** (mean/max/sum) — `global_min_pool` doesn't exist in PyG 2.5.0.
- **probe.py/finetune.py in the spec never actually train the heads** — they attach heads to a `NeutrinoJEPA` instance but `trainer.fit` would still run JEPA's self-supervised `training_step`. Needs a genuinely new `SupervisedFineTune` LightningModule with its own supervised `training_step` (encode → heads → `VonMisesFisher3DLoss` after az/zen→Cartesian conversion + BCE). Use a **bare `PETEncoder`**, not a full `NeutrinoJEPA` — avoids needing `ema_decay`/predictor fields in probe/finetune configs.
- **Checkpoint compatibility across pretrain paths**: both `NeutrinoJEPA` and the MAE pretrain module must register their encoder as `self.encoder` — the fine-tune loader finds pretrained weights by filtering `state_dict` keys with an `"encoder."` prefix, and this must work identically regardless of which pretraining produced the checkpoint.
- **GraphNeT real API** (verified against source, not guessed): `ParquetDataset(path, pulsemaps, features, truth, *, graph_definition=None, truth_table="truth", ...)`; use `graphnet.models.detector.icecube.IceCubeKaggle` (purpose-built for this exact competition's schema — don't hand-parse `sensor_geometry.csv`); `graphnet.data.constants.FEATURES.KAGGLE` / `TRUTH.KAGGLE` for column lists; import `GraphDefinition`/`KNNGraph` from `graphnet.models.data_representation` (the `graphnet.models.graphs` path is deprecated); use `graphnet.data.dataloader.DataLoader` (GraphNeT's own, filters <2-pulse events), not raw PyG DataLoader, for real data. **Still unverified**: exact `pulsemaps` string value and whether a `list[str]` `path` behaves as simple concatenation vs. GraphNeT's "ensemble" mode — write a test against synthetic Kaggle-schema fixtures that fails loudly and specifically if this is wrong, don't silently assume it works.
- **`max_pulses` must be enforced yourself** — nothing else caps event size, and the spec's "cheap at N≤256" assumption for `PETBlock`'s `knn_graph` depends on it. Do this as a dependency-free post-load truncation wrapper (random subsample), not by guessing at a GraphNeT truncation parameter.
- **Checkpoint paths must be explicit** — Lightning's default `ModelCheckpoint` location won't match what configs reference; always pass explicit `dirpath`/`filename`, and give every stage (pretrain, pretrain_mae, probe, finetune, scratch) its own non-colliding checkpoint dir.
- **Fine-tuning loss**: use the spec-mandated `graphnet.training.loss_functions.VonMisesFisher3DLoss` with the az/zen→Cartesian conversion (`[sin(zen)cos(az), sin(zen)sin(az), cos(zen)]`), not a hand-rolled cosine loss.

---

## Steps

- **STEP-01**: Scaffold repo — `pyproject.toml`, `requirements.txt` (pinned: torch 2.3.0, torch-geometric 2.5.0, torch-cluster 1.6.3, torch-scatter 2.1.2, lightning 2.3.0, torch-ema 0.3.0, graphnet>=1.0, pandas 2.2.0, pyarrow 15.0.0, numpy 1.26.0, plus pyyaml/tqdm/scikit-learn/wandb/pytest/nbformat), `.gitignore`, `README.md`, package dirs with `__init__.py`, `git init`.

- **STEP-02**: `tests/test_environment.py` — assert pinned versions installed; assert the specific GraphNeT symbols this plan depends on import cleanly (`IceCubeKaggle`, `KNNGraph`/`GraphDefinition` from `data_representation`, `FEATURES.KAGGLE`/`TRUTH.KAGGLE`, `graphnet.data.dataloader.DataLoader`); assert `knn_graph`/`scatter_softmax` work.

- **STEP-03**: `tests/conftest.py` — synthetic fixtures: sensor geometry CSV, a handful of real-schema `batch_N.parquet` files (`event_id, sensor_id, time, charge, auxiliary`), matching `train_meta.parquet`, and a small flattened PyG `Batch` shaped like GraphNeT's normalized output (`x`: `[x,y,z,t,q,aux]`).

- **STEP-04**: `train/config.py` — `load_config(path)` (YAML → nested `SimpleNamespace`), `flatten_sections(cfg, *names)` (merge named sections into one flat namespace, later overrides earlier).

- **STEP-05**: `data/masking.py` — `spatial_cluster_mask(xyz, mask_ratio, n_clusters) -> (context_mask, target_mask)`: greedily grow clusters from random anchors by nearest-neighbor distance until `mask_ratio * N` pulses covered; disjoint, covering, bool masks.

- **STEP-06**: `models/heads.py` — `DirectionHead` (→ az∈[0,2π], zen∈[0,π] via sigmoid scaling), `ClassificationHead` (→ raw logit).

- **STEP-07/08**: `models/pet.py` — `PETBlock`: knn graph via `torch_cluster.knn_graph`, per-edge MLP (`Linear(2d→d)→ReLU→Linear(d→d)`), attention via `scatter_softmax`/`scatter_sum` over K/V built from neighbor features + edge embedding, residual+LN+FFN. `PETEncoder`: stack of `PETBlock`s, input projection, `encode_event` = 3-way pool (mean/max/sum) → `pool_proj`.

- **STEP-09**: `models/polarbert.py` — `PolarBERTEncoder` (padded-tensor pre-LN transformer, CLS token, unused elsewhere in this plan) + `MAEHead` (→ predicted normalized `(t,q)`, used by STEP-16).

- **STEP-10**: `models/jepa.py` — `NeutrinoJEPA(pl.LightningModule)`: `PETEncoder` + `torch_ema.ExponentialMovingAverage` + predictor MLP (context summary + target xyz → predicted embedding). `training_step`: spatial-cluster-mask per event, zero masked pulses for context encoding, target via EMA-swapped `self.encoder` (see gotchas), normalized-L2 loss (`2 - 2*cos_sim`). EMA update in `on_before_zero_grad`. Checkpoint EMA state explicitly. Cosine LR schedule + AdamW.

- **STEP-11**: `eval/metrics.py` — `mean_angular_error` (great-circle distance in degrees via Cartesian dot product), `angular_error_by_energy` (10 log10-energy bins across [2,7]).

- **STEP-12**: `train/data.py` — `batch_file_paths(batch_dir, batch_range)`; `truncate_to_max_pulses`/`MaxPulsesDataset` (pure-Python random-subsample wrapper, no GraphNeT dependency); `build_graph_definition()` = `KNNGraph(detector=IceCubeKaggle())`; `build_dataset(batch_dir, meta_path, batch_range, max_pulses)` = `ParquetDataset` over the explicit per-split file list (bypass `GraphNeTDataModule`'s ratio-based splitting — it has no notion of batch-file ranges), wrapped in `MaxPulsesDataset`; `build_dataloader` using GraphNeT's own `DataLoader`. Test against the real fixtures from STEP-03, fail loud and specific (not silent) if the unverified `pulsemaps`/`path` assumptions are wrong.

- **STEP-13**: Configs — `pretrain.yaml` (JEPA), `pretrain_mae.yaml` (MAE), `probe.yaml`, `finetune.yaml`, `scratch.yaml` (no `checkpoint` field — random init). Each has its own `checkpoint: {dirpath, filename}` section, all distinct dirs. `probe`/`finetune`/`scratch` model sections only need `d,L,k` (bare `PETEncoder`), not `ema_decay`.

- **STEP-14**: `models/finetune.py` — `SupervisedFineTune(pl.LightningModule)`: bare `PETEncoder` + both heads, works frozen/unfrozen/random-init identically. `training_step` = `encode_event` → heads → `VonMisesFisher3DLoss` + BCE, weighted combine. `configure_optimizers`: single group (heads only) if frozen, else two groups (`lr_encoder`, `lr_heads`). `build_supervised_model(cfg, checkpoint_path)`: loads only `"encoder."`-prefixed `state_dict` keys if a checkpoint is given (works against both JEPA and MAE checkpoints); `None` → random init.

- **STEP-15**: `train/pretrain.py` — JEPA pretraining entrypoint (experiment 2). Real `build_dataset`/`build_dataloader` from STEP-12. `ModelCheckpoint` with explicit `dirpath`/`filename` from config. `--config`/`--fast-dev-run` CLI.

- **STEP-16**: `train/pretrain_mae.py` — `MAEPretrain(pl.LightningModule)`: bare `PETEncoder` + `MAEHead` (no PolarBERT), uniform-random 25% masking (distinct helper from STEP-05's spatial masking), MSE loss on masked `(t,q)`. Registers `self.encoder`. Own entrypoint, own checkpoint dir. Experiment 1's pretraining stage.

- **STEP-17**: `train/probe.py` — `build_supervised_model` with checkpoint required, `freeze_encoder=True`. Own `ModelCheckpoint`. Train/val loaders via STEP-12.

- **STEP-18**: `train/finetune.py` — same shape, `freeze_encoder=False`, checkpoint optional (`None` → random init, doubles as experiment 3 via `scratch.yaml`, no separate script needed). `EarlyStopping` on `val/angular_error`. Own `ModelCheckpoint`.

- **STEP-19**: `scripts/download_data.sh` — `kaggle competitions download -c icecube-neutrinos-in-deep-ice` + unzip into the layout `configs/*.yaml`'s `data:` sections expect.

- **STEP-20**: `tests/test_integration_pipeline.py` — end-to-end smoke test, all 3 experiments, `fast_dev_run=True` on synthetic data: (1) JEPA pretrain→checkpoint→probe→finetune, (2) MAE pretrain→checkpoint→probe→finetune, (3) from-scratch finetune with no checkpoint. No GPU/real data required.

- **STEP-21**: `scripts/make_colab_notebook.py` — generates `colab_train_neutrinojepa.ipynb` (via `nbformat`) modeled on the user's reference `colab_train.ipynb` (LeWM project — same `subprocess.run`-driven pattern, different domain): install torch/PyG cu121 wheels → clone repo (`REPO_URL` as an editable config-cell variable, not yet known) → `pip install -r requirements.txt` → mount Drive, create a Drive checkpoint dir → Kaggle creds from Colab secrets → `scripts/download_data.sh` with streaming stdout + post-download sanity check (raise if incomplete, mirroring the reference notebook's dataset-size check) → shared `run_stage(script, config, watch_dir)` helper: `subprocess.Popen` streaming stdout + a daemon thread polling `watch_dir` every 30s, debouncing on stable file size (avoid copying mid-write), copying new `*.ckpt` files to Drive, `raise SystemExit` on nonzero return code → `set_checkpoint(config_path, new_checkpoint)` helper (regex-patches a config's `checkpoint:` line in place, mirroring the reference notebook's own config-patch cell) → three cells running all 3 experiments end to end (JEPA: pretrain→probe→finetune; MAE: pretrain_mae→patch probe/finetune configs→probe→finetune; scratch: finetune.py directly on `scratch.yaml`). Test: generate, load with `nbformat`, assert cell count and that each stage's script/config names actually appear in the generated source.

## Out of scope

DynEdge/PolarBERT ablations (different architecture), real Kaggle download + real training runs (needs credentials/GPU you have, not this session), W&B login, hyperparameter tuning, reconciling Colab's preinstalled CUDA version against the pinned wheels (do that when actually running STEP-21's notebook, not during STEP-01/02).
