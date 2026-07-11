import matplotlib

matplotlib.use("Agg")  # headless: Colab/CI safe, no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

_METHOD_COLORS = {
    "scratch": "tab:gray",
    "mae": "tab:orange",
    "jepa": "tab:blue",
}


def _color_for(method: str, index: int) -> str:
    return _METHOD_COLORS.get(method, f"C{index}")


def plot_angular_error_vs_energy(
    binned_errors: dict[str, dict[float, float]],
    save_path: str,
    title: str = "Angular resolution vs. energy",
) -> None:
    """Fig. 1A. binned_errors: {method_name: {bin_center: mean_angular_error}},
    typically one dict per method from eval.metrics.angular_error_by_energy."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (method, bins) in enumerate(binned_errors.items()):
        centers = sorted(bins.keys())
        errors = [bins[c] for c in centers]
        ax.plot(centers, errors, marker="o", label=method, color=_color_for(method, i))
    ax.set_xlabel("log10(energy proxy)")
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_method_comparison_bars(
    results: dict[str, dict[str, float]],
    save_path: str,
    title: str = "Direction reconstruction: probe vs. fine-tune",
) -> None:
    """Fig. 1B. results: {method_name: {"probe": error, "finetune": error}}.
    "probe" may be omitted for a from-scratch baseline (no pretrained
    encoder exists to freeze) -- that bar is simply skipped, not an error."""
    methods = list(results.keys())
    stages = ["probe", "finetune"]
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, stage in enumerate(stages):
        values = [results[m].get(stage, float("nan")) for m in methods]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=stage)
        ax.bar_label(bars, fmt="%.1f")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_sample_efficiency(
    results: dict[str, dict[int, float]],
    save_path: str,
    title: str = "Fine-tuning sample efficiency",
) -> None:
    """Fig. 2A. results: {method_name: {n_labeled_events: angular_error}}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (method, points) in enumerate(results.items()):
        n_events = sorted(points.keys())
        errors = [points[n] for n in n_events]
        ax.plot(n_events, errors, marker="o", label=method, color=_color_for(method, i))
    ax.set_xscale("log")
    ax.set_xlabel("# labeled fine-tuning events")
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_training_curves(
    histories: dict[str, list[float]],
    save_path: str,
    ylabel: str = "Validation angular error (degrees)",
    title: str = "Fine-tuning convergence",
) -> None:
    """Fig. 2B. histories: {method_name: [value_per_epoch, ...]}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (method, values) in enumerate(histories.items()):
        ax.plot(range(1, len(values) + 1), values, label=method, color=_color_for(method, i))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_embedding_projection(
    embeddings: dict[str, np.ndarray],
    color_values: dict[str, np.ndarray],
    save_path: str,
    color_label: str = "zenith (rad)",
    title: str = "Learned representations (t-SNE)",
    seed: int = 0,
) -> None:
    """Fig. 3. embeddings: {method_name: (N, d) array}, color_values matching
    (N,) arrays of a physical quantity to color each point by. One t-SNE
    panel per method, shared color scale, side by side for direct comparison."""
    methods = list(embeddings.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), squeeze=False)
    axes = axes[0]

    vmin = min(v.min() for v in color_values.values())
    vmax = max(v.max() for v in color_values.values())

    scatter = None
    for ax, method in zip(axes, methods):
        emb = embeddings[method]
        n = emb.shape[0]
        perplexity = min(30, max(1, n - 1))
        projected = TSNE(
            n_components=2, perplexity=perplexity, random_state=seed, init="pca"
        ).fit_transform(emb)
        scatter = ax.scatter(
            projected[:, 0], projected[:, 1], c=color_values[method],
            vmin=vmin, vmax=vmax, cmap="viridis", s=12,
        )
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.colorbar(scatter, ax=axes.tolist(), label=color_label, shrink=0.8)
    fig.savefig(save_path)
    plt.close(fig)


def plot_masking_comparison(
    xyz: torch.Tensor,
    spatial_target_mask: torch.Tensor,
    random_mask: torch.Tensor,
    save_path: str,
    title: str = "Spatial cluster masking vs. uniform random masking",
) -> None:
    """Fig. 4. Same real event, 3 panels: (a) all real pulses, (b) JEPA's
    spatial_cluster_mask target set highlighted, (c) uniform_random_mask
    target set highlighted -- visualizes the architecture's central claim
    that spatial masking removes contiguous detector sub-volumes rather than
    scattered individual pulses."""
    xy = xyz[:, :2].detach().cpu().numpy()
    spatial_mask = spatial_target_mask.detach().cpu().numpy()
    rand_mask = random_mask.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(xy[:, 0], xy[:, 1], c="tab:blue", s=20)
    axes[0].set_title("(a) Full event")

    axes[1].scatter(xy[~spatial_mask, 0], xy[~spatial_mask, 1], c="tab:blue", s=20, label="context")
    axes[1].scatter(xy[spatial_mask, 0], xy[spatial_mask, 1], c="tab:red", s=30, marker="x", label="masked")
    axes[1].set_title("(b) Spatial cluster mask (JEPA)")
    axes[1].legend()

    axes[2].scatter(xy[~rand_mask, 0], xy[~rand_mask, 1], c="tab:blue", s=20, label="context")
    axes[2].scatter(xy[rand_mask, 0], xy[rand_mask, 1], c="tab:red", s=30, marker="x", label="masked")
    axes[2].set_title("(c) Uniform random mask (MAE)")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
