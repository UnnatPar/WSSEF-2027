import importlib

import pytest

PINNED = {
    "torch": "2.3.0",
    "torch_geometric": "2.5.0",
    "torch_cluster": "1.6.3",
    "torch_scatter": "2.1.2",
    "lightning": "2.3.0",
    "torch_ema": "0.3.0",
    "pandas": "2.2.0",
    "pyarrow": "15.0.0",
    "numpy": "1.26.0",
}


@pytest.mark.parametrize("module_name,expected_version", PINNED.items())
def test_pinned_version(module_name, expected_version):
    mod = importlib.import_module(module_name)
    assert mod.__version__ == expected_version, (
        f"{module_name} is {mod.__version__}, expected {expected_version}"
    )


def test_graphnet_importable():
    import graphnet  # noqa: F401


def test_graphnet_kaggle_symbols_exist():
    # Verifies the exact GraphNeT symbols train/data.py depends on. Fail
    # loudly here rather than deep in a data-loading stack trace if a
    # graphnet version bump ever moves/renames them.
    from graphnet.data.constants import FEATURES, TRUTH
    from graphnet.data.dataloader import DataLoader  # noqa: F401
    from graphnet.data.dataset import ParquetDataset  # noqa: F401
    from graphnet.models.data_representation import GraphDefinition, KNNGraph  # noqa: F401
    from graphnet.models.detector.icecube import IceCubeKaggle  # noqa: F401

    assert FEATURES.KAGGLE == ["x", "y", "z", "time", "charge", "auxiliary"]
    assert set(TRUTH.KAGGLE) == {"zenith", "azimuth"}


def test_pyg_extension_ops_work():
    import torch
    from torch_cluster import knn_graph
    from torch_scatter import scatter_softmax

    x = torch.randn(20, 8)
    batch = torch.zeros(20, dtype=torch.long)
    edge_index = knn_graph(x, k=4, batch=batch, flow="source_to_target")
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 20 * 4

    scores = torch.randn(edge_index.shape[1])
    alpha = scatter_softmax(scores, edge_index[1], dim=0, dim_size=20)
    assert alpha.shape == scores.shape
