from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train.pretrain import build_model, build_trainer


class TinyEventDataset(Dataset):
    def __init__(self, n_events=8, max_nodes=15):
        self.events = []
        g = torch.Generator().manual_seed(0)
        for _ in range(n_events):
            n = torch.randint(5, max_nodes, (1,), generator=g).item()
            self.events.append(Data(x=torch.rand(n, 6, generator=g)))

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.events[idx]


def make_flat_cfg():
    return SimpleNamespace(
        d=16, L=2, k=4, ema_decay=0.99,
        ratio_min=0.4, ratio_max=0.6, n_clusters=2,
        lr=1e-3, weight_decay=0.01, epochs=1,
        batch_size=4, grad_clip=1.0,
    )


def test_build_model_returns_neutrinojepa():
    from models.jepa import NeutrinoJEPA
    model = build_model(make_flat_cfg())
    assert isinstance(model, NeutrinoJEPA)


def test_fast_dev_run_completes_one_batch():
    cfg = make_flat_cfg()
    model = build_model(cfg)
    loader = DataLoader(TinyEventDataset(), batch_size=cfg.batch_size)
    trainer = build_trainer(cfg, fast_dev_run=True)
    trainer.fit(model, loader)
    assert trainer.state.finished


def test_checkpoint_dirpath_and_filename_are_configurable(tmp_path):
    cfg = make_flat_cfg()
    trainer = build_trainer(
        cfg, fast_dev_run=True,
        checkpoint_dirpath=str(tmp_path / "ckpts"), checkpoint_filename="epoch{epoch}",
    )
    checkpoint_callback = trainer.checkpoint_callback
    assert str(checkpoint_callback.dirpath) == str(tmp_path / "ckpts")
    assert checkpoint_callback.filename == "epoch{epoch}"
