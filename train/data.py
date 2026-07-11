from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graphnet.models.detector.icecube import IceCubeKaggle
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def batch_file_paths(batch_dir: str, batch_range: list[int]) -> list[str]:
    start, end = batch_range
    return [str(Path(batch_dir) / f"batch_{i}.parquet") for i in range(start, end + 1)]


def truncate_to_max_pulses(data: Data, max_pulses: int) -> Data:
    """Randomly subsample an event's pulses down to max_pulses. Nothing else
    caps event size, and the spec's "cheap at N<=256" assumption for
    spatial_cluster_mask / PETBlock's knn_graph depends on it."""
    n = data.x.shape[0]
    if n <= max_pulses:
        return data
    perm = torch.randperm(n)[:max_pulses]
    data.x = data.x[perm]
    return data


class MaxPulsesDataset(Dataset):
    """Wraps any Dataset yielding PyG Data objects and truncates each event
    to at most max_pulses pulses on access."""

    def __init__(self, dataset, max_pulses: int):
        self.dataset = dataset
        self.max_pulses = max_pulses

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return truncate_to_max_pulses(self.dataset[idx], self.max_pulses)


class KaggleParquetDataset(Dataset):
    """Loads IceCube Kaggle-competition-schema parquet files directly.

    GraphNeT's own graphnet.data.dataset.ParquetDataset CANNOT read these
    raw files -- verified by direct testing, not assumed. It expects its own
    converted, chunked directory layout
    (path/<table_name>/<table_name>_<chunk_id>.parquet, indexed by
    "event_no"), produced by GraphNeT's DataConverter/ParquetWriter -- not
    the competition's flat batch_N.parquet files (event_id, sensor_id, time,
    charge, auxiliary) plus a separate sensor_geometry.csv and
    train_meta.parquet. So this dataset does the geometry join and
    normalization itself, reusing GraphNeT's real IceCubeKaggle().feature_map()
    functions rather than re-deriving them (also verified by direct testing:
    the real normalization is x,y,z/500, time=(t-1e4)/3e4, charge=log10(q)/3,
    auxiliary unchanged -- the spec's prose description of "min-subtracted
    per event" time and "log1p" charge does not match the actual shipped
    GraphNeT behavior).

    Track/cascade classification labels ("pid") are NOT present in this
    Kaggle competition's data at all -- only azimuth/zenith -- so no `pid`
    attribute is attached to the returned Data objects. Models consuming
    this dataset must treat classification loss as optional.
    """

    FEATURE_COLUMNS = ["x", "y", "z", "time", "charge"]

    def __init__(
        self,
        batch_dir: str,
        geometry_path: str,
        meta_path: str,
        batch_range: list[int],
        cache_size: int = 4,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.batch_dir = Path(batch_dir)
        self.geometry = pd.read_csv(geometry_path).set_index("sensor_id")
        self.feature_map = IceCubeKaggle().feature_map()

        meta = pd.read_parquet(meta_path)
        start, end = batch_range
        meta = meta[(meta["batch_id"] >= start) & (meta["batch_id"] <= end)]
        self.meta = meta.set_index("event_id")
        self.event_ids = self._ordered_event_ids(shuffle, seed)

        self._cache_size = cache_size
        self._cache: dict = {}
        self._cache_order: list = []

    def _ordered_event_ids(self, shuffle: bool, seed: int) -> list:
        # Deliberately NOT a global event-level shuffle: with hundreds of real
        # batch files and only a handful cached, requesting events in fully
        # random order means almost every __getitem__ evicts the cache and
        # re-reads + re-groups an entire ~200k-row parquet file from disk.
        # Order by batch_id groups instead (shuffling the group order, and
        # optionally within each group) so a shuffled epoch still mostly hits
        # whatever batch file is already cached. Verified this is necessary
        # by reasoning through real Kaggle-competition scale (660 files,
        # ~200k events each) -- the tiny synthetic test fixtures (3 files)
        # trivially fit in cache regardless of ordering and never catch this.
        rng = np.random.default_rng(seed)
        groups = self.meta.groupby("batch_id").groups
        batch_ids = list(groups.keys())
        if shuffle:
            rng.shuffle(batch_ids)
        event_ids = []
        for batch_id in batch_ids:
            ids = list(groups[batch_id])
            if shuffle:
                rng.shuffle(ids)
            event_ids.extend(ids)
        return event_ids

    def __len__(self) -> int:
        return len(self.event_ids)

    def _load_batch(self, batch_num: int) -> dict:
        if batch_num in self._cache:
            return self._cache[batch_num]
        df = pd.read_parquet(self.batch_dir / f"batch_{batch_num}.parquet")
        df = df.join(self.geometry, on="sensor_id")
        grouped = {event_id: g for event_id, g in df.groupby("event_id")}
        self._cache[batch_num] = grouped
        self._cache_order.append(batch_num)
        if len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]
        return grouped

    def __getitem__(self, idx: int) -> Data:
        event_id = self.event_ids[idx]
        batch_num = int(self.meta.loc[event_id, "batch_id"])
        pulses = self._load_batch(batch_num)[event_id]

        columns = []
        for col in self.FEATURE_COLUMNS:
            t = torch.tensor(pulses[col].to_numpy(), dtype=torch.float32)
            columns.append(self.feature_map[col](t))
        columns.append(torch.tensor(pulses["auxiliary"].to_numpy(), dtype=torch.float32))
        x = torch.stack(columns, dim=-1)

        data = Data(x=x)
        data.azimuth = torch.tensor(self.meta.loc[event_id, "azimuth"], dtype=torch.float32)
        data.zenith = torch.tensor(self.meta.loc[event_id, "zenith"], dtype=torch.float32)
        return data


def build_dataset(
    batch_dir: str,
    geometry_path: str,
    meta_path: str,
    batch_range: list[int],
    max_pulses: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
):
    dataset = KaggleParquetDataset(
        batch_dir, geometry_path, meta_path, batch_range, shuffle=shuffle, seed=seed,
    )
    if max_pulses is not None:
        return MaxPulsesDataset(dataset, max_pulses)
    return dataset


def build_dataloader(dataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    # No `shuffle` parameter here on purpose: PyTorch's own row-level shuffle
    # would silently undo build_dataset(..., shuffle=True)'s cache-friendly
    # batch-locality ordering (see KaggleParquetDataset._ordered_event_ids).
    # Any epoch-level shuffling must happen via that flag, at construction
    # time, not here.
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, shuffle=False)
    if num_workers > 0:
        # pandas/pyarrow's parquet reader is not fork-safe -- a worker forked
        # while pyarrow holds an internal thread-pool mutex can hang forever.
        # GraphNeT's own ParquetDataset docstring warns about exactly this
        # for the same reason (also pandas/pyarrow-based).
        kwargs["multiprocessing_context"] = "spawn"
        # Without this, PyTorch tears down and respawns every worker at the
        # end of each epoch (the default). Verified by direct measurement:
        # a single spawned worker takes ~13s just to re-import
        # torch/lightning/graphnet from scratch (spawn doesn't inherit
        # already-loaded modules the way fork does) -- across num_workers=8
        # and 100 pretrain epochs that's potentially hours of pure respawn
        # overhead for zero benefit.
        kwargs["persistent_workers"] = True
        kwargs["pin_memory"] = True
    return DataLoader(dataset, **kwargs)
