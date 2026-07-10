from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from eval.metrics import mean_angular_error
from models.finetune import SupervisedFineTune, build_supervised_model
from train.pretrain import build_model as build_jepa_model
from train.pretrain import build_trainer as build_jepa_trainer
from train.pretrain_mae import build_model as build_mae_model
from train.pretrain_mae import build_trainer as build_mae_trainer


class TinyEventDataset(Dataset):
    def __init__(self, n_events=8, max_nodes=15, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.events = [
            Data(x=torch.rand(torch.randint(5, max_nodes, (1,), generator=g).item(), 6, generator=g))
            for _ in range(n_events)
        ]

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.events[idx]


def make_jepa_cfg():
    return SimpleNamespace(
        d=16, L=2, k=4, ema_decay=0.99,
        ratio_min=0.4, ratio_max=0.6, n_clusters=2,
        lr=1e-3, weight_decay=0.01, epochs=1,
        batch_size=4, grad_clip=1.0,
    )


def make_mae_cfg():
    return SimpleNamespace(
        d=16, L=2, k=4, mask_ratio=0.25,
        lr=1e-3, weight_decay=0.01, epochs=1,
        batch_size=4, grad_clip=1.0,
    )


def make_supervised_cfg(freeze_encoder):
    return SimpleNamespace(
        d=16, L=2, k=4, freeze_encoder=freeze_encoder,
        lr_encoder=1e-4, lr_heads=1e-3, weight_decay=0.01,
        lambda_direction=1.0, lambda_classification=0.5,
    )


def _probe_forward_and_eval(model, loader):
    batch = next(iter(loader))
    with torch.no_grad():
        g = model.encoder.encode_event(batch.x, batch.batch)
    az, zen = model.direction_head(g)
    cls_logits = model.classification_head(g)
    assert az.shape == zen.shape == (g.shape[0],)
    assert cls_logits.shape == (g.shape[0], 1)
    true_az = torch.rand(g.shape[0]) * 6.28
    true_zen = torch.rand(g.shape[0]) * 3.14
    err = mean_angular_error(az, zen, true_az, true_zen)
    assert 0.0 <= err <= 180.0


def test_experiment_2_jepa_pretrain_to_probe_and_finetune(tmp_path):
    jepa_cfg = make_jepa_cfg()
    model = build_jepa_model(jepa_cfg)
    loader = DataLoader(TinyEventDataset(), batch_size=jepa_cfg.batch_size)
    trainer = build_jepa_trainer(jepa_cfg, fast_dev_run=True)
    trainer.fit(model, loader)

    ckpt_path = tmp_path / "jepa_ckpt.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    probe_model = build_supervised_model(make_supervised_cfg(freeze_encoder=True), str(ckpt_path))
    assert all(not p.requires_grad for p in probe_model.encoder.parameters())
    _probe_forward_and_eval(probe_model, loader)

    finetune_model = build_supervised_model(make_supervised_cfg(freeze_encoder=False), str(ckpt_path))
    assert all(p.requires_grad for p in finetune_model.encoder.parameters())


def test_experiment_1_mae_pretrain_to_probe_and_finetune(tmp_path):
    mae_cfg = make_mae_cfg()
    model = build_mae_model(mae_cfg)
    loader = DataLoader(TinyEventDataset(), batch_size=mae_cfg.batch_size)
    trainer = build_mae_trainer(mae_cfg, fast_dev_run=True)
    trainer.fit(model, loader)

    ckpt_path = tmp_path / "mae_ckpt.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    probe_model = build_supervised_model(make_supervised_cfg(freeze_encoder=True), str(ckpt_path))
    assert all(not p.requires_grad for p in probe_model.encoder.parameters())
    _probe_forward_and_eval(probe_model, loader)


def test_experiment_3_from_scratch_needs_no_checkpoint():
    model = build_supervised_model(make_supervised_cfg(freeze_encoder=False), None)
    assert isinstance(model, SupervisedFineTune)
    assert all(p.requires_grad for p in model.encoder.parameters())
    loader = DataLoader(TinyEventDataset(), batch_size=4)
    _probe_forward_and_eval(model, loader)


def test_real_kaggle_schema_pipeline_end_to_end(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    # Exercises train/data.py's real loader (not the hand-built TinyEventDataset
    # above) feeding directly into the JEPA pretraining step, closing the loop
    # between the data-loading layer and the model layer.
    from train.data import build_dataloader, build_dataset

    dataset = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet),
        [1, 3], max_pulses=256,
    )
    loader = build_dataloader(dataset, batch_size=4, shuffle=False)

    jepa_cfg = make_jepa_cfg()
    model = build_jepa_model(jepa_cfg)
    trainer = build_jepa_trainer(jepa_cfg, fast_dev_run=True)
    trainer.fit(model, loader)
    assert trainer.state.finished

    # And the same real batches carry azimuth/zenith the supervised path needs.
    batch = next(iter(loader))
    supervised_model = SupervisedFineTune(make_supervised_cfg(freeze_encoder=False))
    loss = supervised_model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
