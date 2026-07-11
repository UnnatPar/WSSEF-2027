from pathlib import Path

import torch

from train.data import (
    KaggleParquetDataset,
    MaxPulsesDataset,
    batch_file_paths,
    build_dataloader,
    build_dataset,
    truncate_to_max_pulses,
)


def test_batch_file_paths_constructs_correct_range():
    paths = batch_file_paths("data/train", [1, 3])
    assert paths == [
        str(Path("data/train/batch_1.parquet")),
        str(Path("data/train/batch_2.parquet")),
        str(Path("data/train/batch_3.parquet")),
    ]


def test_batch_file_paths_single_batch():
    paths = batch_file_paths("data/train", [50, 50])
    assert paths == [str(Path("data/train/batch_50.parquet"))]


def test_truncate_to_max_pulses_leaves_small_events_unchanged():
    from torch_geometric.data import Data
    data = Data(x=torch.randn(10, 6))
    truncated = truncate_to_max_pulses(data, max_pulses=256)
    assert truncated.x.shape == (10, 6)


def test_truncate_to_max_pulses_subsamples_large_events():
    from torch_geometric.data import Data
    torch.manual_seed(0)
    data = Data(x=torch.randn(500, 6))
    truncated = truncate_to_max_pulses(data, max_pulses=256)
    assert truncated.x.shape == (256, 6)


def test_max_pulses_dataset_wraps_and_truncates():
    from torch_geometric.data import Data

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, events):
            self.events = events

        def __len__(self):
            return len(self.events)

        def __getitem__(self, idx):
            return self.events[idx]

    inner = ListDataset([Data(x=torch.randn(10, 6)), Data(x=torch.randn(500, 6))])
    wrapped = MaxPulsesDataset(inner, max_pulses=256)
    assert len(wrapped) == 2
    assert wrapped[0].x.shape == (10, 6)
    assert wrapped[1].x.shape == (256, 6)


def test_build_dataset_against_real_kaggle_schema_fixtures(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    dataset = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet),
        [1, 3], max_pulses=256,
    )
    assert len(dataset) > 0

    event = dataset[0]
    assert event.x.shape[1] == 6
    assert event.x.shape[0] <= 256
    assert hasattr(event, "azimuth")
    assert hasattr(event, "zenith")
    assert 0.0 <= event.azimuth.item() <= 2 * torch.pi
    assert 0.0 <= event.zenith.item() <= torch.pi
    # x, y, z should be roughly in [-1, 1] after /500 normalization for our
    # synthetic geometry (drawn from [-500, 500])
    assert event.x[:, :3].abs().max() <= 1.5


def test_build_dataset_respects_batch_range(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    full = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 3],
    )
    partial = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 1],
    )
    assert len(partial) < len(full)


def test_build_dataloader_batches_events(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    dataset = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet),
        [1, 3], max_pulses=256,
    )
    loader = build_dataloader(dataset, batch_size=4)
    batch = next(iter(loader))
    assert batch.x.shape[1] == 6
    assert batch.batch.max().item() <= 3  # up to 4 events (indices 0-3) in the batch


def test_unshuffled_event_order_is_grouped_by_batch_id(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    dataset = KaggleParquetDataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 3],
    )
    batch_ids_in_order = [int(dataset.meta.loc[eid, "batch_id"]) for eid in dataset.event_ids]
    # Consecutive events must come from the same batch file except exactly
    # at the n_batches-1 boundaries between groups -- this is what keeps the
    # LRU file cache from thrashing at real (hundreds-of-files) scale.
    same_batch_transitions = sum(
        1 for a, b in zip(batch_ids_in_order, batch_ids_in_order[1:]) if a == b
    )
    n_batches = len(set(batch_ids_in_order))
    assert same_batch_transitions == len(batch_ids_in_order) - n_batches


def test_shuffled_event_order_still_groups_by_batch_id(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    dataset = KaggleParquetDataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 3],
        shuffle=True, seed=1,
    )
    batch_ids_in_order = [int(dataset.meta.loc[eid, "batch_id"]) for eid in dataset.event_ids]
    # A truly global shuffle across all events would make same-batch
    # transitions rare (~1/n_batches chance per transition). Grouped-shuffle
    # keeps whole batches contiguous, so almost every transition should
    # still stay within the same batch.
    same_batch_transitions = sum(
        1 for a, b in zip(batch_ids_in_order, batch_ids_in_order[1:]) if a == b
    )
    n_batches = len(set(batch_ids_in_order))
    assert same_batch_transitions >= len(batch_ids_in_order) - n_batches


def test_shuffled_and_unshuffled_orderings_contain_the_same_events(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    unshuffled = KaggleParquetDataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 3],
    )
    shuffled = KaggleParquetDataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 3],
        shuffle=True, seed=2,
    )
    assert sorted(unshuffled.event_ids) == sorted(shuffled.event_ids)


def test_build_dataset_shuffle_flag_reaches_the_underlying_dataset(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    dataset = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet),
        [1, 3], max_pulses=256, shuffle=True, seed=3,
    )
    # build_dataset wraps in MaxPulsesDataset when max_pulses is set
    assert isinstance(dataset, MaxPulsesDataset)
    assert isinstance(dataset.dataset, KaggleParquetDataset)


def test_build_dataloader_uses_spawn_context_when_multiprocessing(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    # pandas/pyarrow's parquet reader is not fork-safe; a worker forked
    # while pyarrow holds an internal thread-pool mutex can hang forever.
    dataset = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 3],
    )
    loader = build_dataloader(dataset, batch_size=4, num_workers=2)
    assert loader.multiprocessing_context is not None
    assert type(loader.multiprocessing_context).__name__ == "SpawnContext"


def test_build_dataloader_no_multiprocessing_context_when_single_process(
    tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    dataset = build_dataset(
        str(tiny_batch_dir), str(tiny_sensor_geometry_csv), str(tiny_meta_parquet), [1, 3],
    )
    loader = build_dataloader(dataset, batch_size=4, num_workers=0)
    assert loader.multiprocessing_context is None
