"""Generates the 6 poster figures + summary table from the 5 trained
checkpoints (jepa_probe, jepa_finetune, mae_probe, mae_finetune, scratch).

Run after all 3 experiments have finished:
    python -u eval/run_report.py --checkpoints-dir checkpoints --output-dir report

Fig 2A (sample efficiency) is NOT produced here -- it needs multiple
fine-tune runs at varying label budgets, which the standard 3-experiment
pipeline doesn't produce. That needs a separate sweep script.
"""

import argparse
import glob
import os

import pandas as pd
import torch

from data.masking import spatial_cluster_mask
from eval.metrics import angular_error_by_energy, mean_angular_error
from eval.plots import (
    plot_angular_error_vs_energy,
    plot_embedding_projection,
    plot_masking_comparison,
    plot_method_comparison_bars,
    plot_training_curves,
)
from eval.tables import build_summary_table, save_summary_table
from models.finetune import load_full_checkpoint
from train.config import flatten_sections, load_config
from train.data import build_dataloader, build_dataset
from train.pretrain_mae import uniform_random_mask


def latest_checkpoint(dirpath: str) -> str:
    files = glob.glob(os.path.join(dirpath, "*.ckpt"))
    if not files:
        raise FileNotFoundError(f"no checkpoints found in {dirpath}")

    def epoch_num(path: str) -> int:
        digits = "".join(c for c in os.path.basename(path) if c.isdigit())
        return int(digits) if digits else -1

    return max(files, key=epoch_num)


@torch.no_grad()
def evaluate_model(model, loader) -> dict:
    """Runs inference over a dataloader, collecting per-event predictions,
    ground truth, pooled embeddings, and a log10(pulse count) energy proxy
    (the real Kaggle data has no true energy column -- pulse count is the
    closest available stand-in; it's clipped by max_pulses truncation, so
    it under-represents the highest-energy events)."""
    model.eval()
    pred_az, pred_zen, true_az, true_zen = [], [], [], []
    log_n_pulses, embeddings, zeniths = [], [], []
    for batch in loader:
        g, az, zen, _ = model._forward(batch)
        pred_az.append(az)
        pred_zen.append(zen)
        true_az.append(batch.azimuth)
        true_zen.append(batch.zenith)
        embeddings.append(g)
        zeniths.append(batch.zenith)
        n_events = int(batch.batch.max().item()) + 1
        counts = torch.bincount(batch.batch, minlength=n_events).float()
        log_n_pulses.append(torch.log10(counts.clamp(min=1)))
    return {
        "pred_az": torch.cat(pred_az), "pred_zen": torch.cat(pred_zen),
        "true_az": torch.cat(true_az), "true_zen": torch.cat(true_zen),
        "log_n_pulses": torch.cat(log_n_pulses),
        "embeddings": torch.cat(embeddings).numpy(),
        "zenith": torch.cat(zeniths).numpy(),
    }


def read_val_angular_error_history(ckpt_dir: str) -> list[float] | None:
    """Reads the val/angular_error column CSVLogger wrote to
    <ckpt_dir>/metrics.csv during training (Fig 2B). Returns None if the
    run used fast_dev_run or WandbLogger only -- not an error, just nothing
    to plot for that method."""
    path = os.path.join(ckpt_dir, "metrics.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "val/angular_error" not in df.columns:
        return None
    df = df.dropna(subset=["val/angular_error"])
    if df.empty:
        return None
    return df.groupby("epoch")["val/angular_error"].last().tolist()


def run_report(
    test_batches: list[int],
    batch_dir: str,
    geometry_path: str,
    meta_path: str,
    max_pulses: int,
    checkpoint_dirs: dict[str, str],
    model_cfg,
    output_dir: str,
    batch_size: int = 256,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    dataset = build_dataset(batch_dir, geometry_path, meta_path, test_batches, max_pulses)
    loader = build_dataloader(dataset, batch_size=batch_size)

    evaluations = {}
    for name, ckpt_dir in checkpoint_dirs.items():
        ckpt_path = latest_checkpoint(ckpt_dir)
        model = load_full_checkpoint(model_cfg, ckpt_path)
        evaluations[name] = evaluate_model(model, loader)
        print(f"Evaluated {name} from {ckpt_path}")

    # Fig 1A: energy-resolved angular error, one line per pretraining objective
    binned = {}
    for label, key in [("jepa", "jepa_finetune"), ("mae", "mae_finetune"), ("scratch", "scratch")]:
        ev = evaluations[key]
        binned[label] = angular_error_by_energy(
            ev["pred_az"], ev["pred_zen"], ev["true_az"], ev["true_zen"], ev["log_n_pulses"]
        )
    plot_angular_error_vs_energy(binned, os.path.join(output_dir, "fig1a_angular_error_vs_energy.png"))

    # Fig 1B + Table 1: probe vs. finetune comparison (scratch has no probe stage)
    def err(key):
        ev = evaluations[key]
        return mean_angular_error(ev["pred_az"], ev["pred_zen"], ev["true_az"], ev["true_zen"])

    bar_results = {
        "jepa": {"probe": err("jepa_probe"), "finetune": err("jepa_finetune")},
        "mae": {"probe": err("mae_probe"), "finetune": err("mae_finetune")},
        "scratch": {"finetune": err("scratch")},
    }
    plot_method_comparison_bars(bar_results, os.path.join(output_dir, "fig1b_method_comparison.png"))

    table_results = {
        "PET+scratch": {
            "pretrain_objective": "none",
            "probe_angular_error": None,
            "finetune_angular_error": bar_results["scratch"]["finetune"],
        },
        "NeutrinoPET": {
            "pretrain_objective": "MAE",
            "probe_angular_error": bar_results["mae"]["probe"],
            "finetune_angular_error": bar_results["mae"]["finetune"],
        },
        "NeutrinoJEPAPET": {
            "pretrain_objective": "JEPA",
            "probe_angular_error": bar_results["jepa"]["probe"],
            "finetune_angular_error": bar_results["jepa"]["finetune"],
        },
    }
    df = build_summary_table(table_results)
    save_summary_table(
        df, os.path.join(output_dir, "table1.csv"), os.path.join(output_dir, "table1.md"),
    )

    # Fig 2B: fine-tuning convergence, read from CSVLogger's metrics.csv per run
    histories = {}
    for label, key in [("jepa", "jepa_finetune"), ("mae", "mae_finetune"), ("scratch", "scratch")]:
        history = read_val_angular_error_history(checkpoint_dirs[key])
        if history:
            histories[label] = history
    if histories:
        plot_training_curves(histories, os.path.join(output_dir, "fig2b_training_curves.png"))
    else:
        print("Fig 2B skipped: no metrics.csv found under any finetune checkpoint dir "
              "(only written by non-fast_dev_run training runs).")

    print("Fig 2A (sample efficiency) skipped: needs multiple fine-tune runs at varying "
          "label budgets, not derivable from these 5 checkpoints alone -- run a separate sweep.")

    # Fig 3: t-SNE of pooled embeddings, one panel per finetuned model
    embeddings = {
        label: evaluations[key]["embeddings"]
        for label, key in [("scratch", "scratch"), ("mae", "mae_finetune"), ("jepa", "jepa_finetune")]
    }
    colors = {
        label: evaluations[key]["zenith"]
        for label, key in [("scratch", "scratch"), ("mae", "mae_finetune"), ("jepa", "jepa_finetune")]
    }
    plot_embedding_projection(embeddings, colors, os.path.join(output_dir, "fig3_embeddings.png"))

    # Fig 4: spatial-cluster vs. uniform-random masking on one real test event
    sample = dataset[0]
    xyz = sample.x[:, :3]
    _, spatial_mask = spatial_cluster_mask(xyz, mask_ratio=0.5, n_clusters=4)
    random_mask = uniform_random_mask(n=xyz.shape[0], ratio=0.5)
    plot_masking_comparison(
        xyz, spatial_mask, random_mask, os.path.join(output_dir, "fig4_masking_comparison.png"),
    )

    print(f"Report written to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--data-config", default="configs/finetune_jepa.yaml")
    parser.add_argument("--output-dir", default="report")
    args = parser.parse_args()

    cfg = load_config(args.data_config)
    model_cfg = flatten_sections(cfg, "model", "training")
    model_cfg.freeze_encoder = False  # irrelevant under eval-mode no_grad, but SupervisedFineTune requires it

    checkpoint_dirs = {
        "jepa_probe": os.path.join(args.checkpoints_dir, "jepa_probe_v1"),
        "jepa_finetune": os.path.join(args.checkpoints_dir, "jepa_finetune_v1"),
        "mae_probe": os.path.join(args.checkpoints_dir, "mae_probe_v1"),
        "mae_finetune": os.path.join(args.checkpoints_dir, "mae_finetune_v1"),
        "scratch": os.path.join(args.checkpoints_dir, "scratch_v1"),
    }
    run_report(
        test_batches=cfg.data.test_batches,
        batch_dir=cfg.data.batch_dir,
        geometry_path=cfg.data.geometry_path,
        meta_path=cfg.data.meta_path,
        max_pulses=cfg.data.max_pulses,
        checkpoint_dirs=checkpoint_dirs,
        model_cfg=model_cfg,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
