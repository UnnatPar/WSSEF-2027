import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

N_SENSORS = 60
N_EVENTS_PER_BATCH = 24
MAX_PULSES_PER_EVENT = 40
N_SYNTHETIC_BATCH_FILES = 3


@pytest.fixture
def tiny_sensor_geometry_csv(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "sensor_geometry.csv"
    df = pd.DataFrame({
        "sensor_id": np.arange(N_SENSORS),
        "x": rng.uniform(-500, 500, N_SENSORS),
        "y": rng.uniform(-500, 500, N_SENSORS),
        "z": rng.uniform(-500, 500, N_SENSORS),
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_batch_dir(tmp_path):
    """N_SYNTHETIC_BATCH_FILES batch_N.parquet files, schema matching the real
    Kaggle competition exactly: event_id, sensor_id, time, charge, auxiliary.
    """
    batch_dir = tmp_path / "train"
    batch_dir.mkdir()
    rng = np.random.default_rng(1)
    for batch_num in range(1, N_SYNTHETIC_BATCH_FILES + 1):
        rows = []
        for event_id in range(N_EVENTS_PER_BATCH):
            n_pulses = rng.integers(5, MAX_PULSES_PER_EVENT)
            for _ in range(n_pulses):
                rows.append({
                    "event_id": batch_num * 1000 + event_id,
                    "sensor_id": int(rng.integers(0, N_SENSORS)),
                    "time": float(rng.uniform(0, 3e4)),
                    "charge": float(rng.exponential(2.0)),
                    "auxiliary": bool(rng.integers(0, 2)),
                })
        pd.DataFrame(rows).to_parquet(batch_dir / f"batch_{batch_num}.parquet")
    return batch_dir


@pytest.fixture
def tiny_meta_parquet(tmp_path):
    rng = np.random.default_rng(2)
    n_total = N_SYNTHETIC_BATCH_FILES * N_EVENTS_PER_BATCH
    event_ids = [
        b * 1000 + e
        for b in range(1, N_SYNTHETIC_BATCH_FILES + 1)
        for e in range(N_EVENTS_PER_BATCH)
    ]
    df = pd.DataFrame({
        "event_id": event_ids,
        "batch_id": [eid // 1000 for eid in event_ids],
        "azimuth": rng.uniform(0, 2 * np.pi, n_total),
        "zenith": rng.uniform(0, np.pi, n_total),
    })
    path = tmp_path / "train_meta.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def tiny_pyg_batch():
    """A small flattened PyG Batch shaped like GraphNeT's ParquetDataset output:
    data.x is (N, 6) = [x, y, z, t, q, auxiliary], already normalized-range.
    """
    rng = torch.Generator().manual_seed(42)
    events = []
    for n_nodes in [12, 18, 9, 15]:
        xyz = torch.rand(n_nodes, 3, generator=rng) * 2 - 1
        t = torch.rand(n_nodes, 1, generator=rng)
        q = torch.rand(n_nodes, 1, generator=rng)
        aux = torch.randint(0, 2, (n_nodes, 1), generator=rng).float()
        x = torch.cat([xyz, t, q, aux], dim=-1)
        events.append(Data(x=x))
    return Batch.from_data_list(events)
