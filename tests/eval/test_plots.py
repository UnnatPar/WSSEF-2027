import numpy as np
import torch

from data.masking import spatial_cluster_mask
from eval.plots import (
    plot_angular_error_vs_energy,
    plot_embedding_projection,
    plot_masking_comparison,
    plot_method_comparison_bars,
    plot_sample_efficiency,
    plot_training_curves,
)
from train.pretrain_mae import uniform_random_mask


def test_plot_angular_error_vs_energy_writes_file(tmp_path):
    binned = {
        "jepa": {2.5: 10.0, 3.5: 8.0, 4.5: 5.0},
        "mae": {2.5: 15.0, 3.5: 12.0, 4.5: 9.0},
        "scratch": {2.5: 25.0, 3.5: 22.0, 4.5: 20.0},
    }
    save_path = tmp_path / "fig1a.png"
    plot_angular_error_vs_energy(binned, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_method_comparison_bars_writes_file(tmp_path):
    results = {
        "jepa": {"probe": 12.0, "finetune": 8.0},
        "mae": {"probe": 18.0, "finetune": 11.0},
        "scratch": {"probe": 30.0, "finetune": 20.0},
    }
    save_path = tmp_path / "fig1b.png"
    plot_method_comparison_bars(results, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_method_comparison_bars_handles_missing_probe_stage(tmp_path):
    # "scratch" has no pretrained encoder to freeze, so it legitimately has
    # no probe stage -- must not crash.
    results = {
        "jepa": {"probe": 12.0, "finetune": 8.0},
        "scratch": {"finetune": 20.0},
    }
    save_path = tmp_path / "fig1b_missing_probe.png"
    plot_method_comparison_bars(results, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_sample_efficiency_writes_file(tmp_path):
    results = {
        "jepa": {1000: 20.0, 10000: 12.0, 100000: 8.0},
        "scratch": {1000: 40.0, 10000: 30.0, 100000: 22.0},
    }
    save_path = tmp_path / "fig2a.png"
    plot_sample_efficiency(results, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_training_curves_writes_file(tmp_path):
    histories = {
        "jepa": [30.0, 20.0, 15.0, 10.0],
        "mae": [35.0, 25.0, 20.0, 15.0],
        "scratch": [50.0, 40.0, 35.0, 30.0],
    }
    save_path = tmp_path / "fig2b.png"
    plot_training_curves(histories, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_embedding_projection_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    embeddings = {
        "scratch": rng.normal(size=(40, 16)),
        "mae": rng.normal(size=(40, 16)),
        "jepa": rng.normal(size=(40, 16)),
    }
    color_values = {name: rng.uniform(0, 3.14, size=40) for name in embeddings}
    save_path = tmp_path / "fig3.png"
    plot_embedding_projection(embeddings, color_values, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_masking_comparison_writes_file(tmp_path):
    torch.manual_seed(0)
    xyz = torch.randn(80, 3)
    _, spatial_mask = spatial_cluster_mask(xyz, mask_ratio=0.5, n_clusters=4)
    random_mask = uniform_random_mask(n=80, ratio=0.5)
    save_path = tmp_path / "fig4.png"
    plot_masking_comparison(xyz, spatial_mask, random_mask, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0
