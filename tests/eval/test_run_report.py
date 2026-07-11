import os
from types import SimpleNamespace

import torch

from eval.run_report import latest_checkpoint, read_val_angular_error_history, run_report
from models.finetune import SupervisedFineTune


def make_model_cfg():
    return SimpleNamespace(
        d=16, L=2, k=4, freeze_encoder=False,
        lr_encoder=1e-4, lr_heads=1e-3, weight_decay=0.01,
        lambda_direction=1.0, lambda_classification=0.5,
    )


def write_fake_checkpoint(dirpath, epoch=0, cfg=None):
    os.makedirs(dirpath, exist_ok=True)
    model = SupervisedFineTune(cfg or make_model_cfg())
    torch.save({"state_dict": model.state_dict()}, os.path.join(dirpath, f"epoch{epoch}.ckpt"))
    return model


def test_latest_checkpoint_picks_highest_epoch(tmp_path):
    d = tmp_path / "ckpts"
    write_fake_checkpoint(d, epoch=0)
    write_fake_checkpoint(d, epoch=1)
    write_fake_checkpoint(d, epoch=11)
    picked = latest_checkpoint(str(d))
    assert picked.endswith("epoch11.ckpt")


def test_latest_checkpoint_raises_when_empty(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(str(tmp_path / "empty"))


def test_read_val_angular_error_history_returns_none_when_missing(tmp_path):
    assert read_val_angular_error_history(str(tmp_path)) is None


def test_read_val_angular_error_history_parses_csv(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "epoch,step,val/angular_error\n"
        "0,0,30.0\n"
        "0,1,28.0\n"
        "1,2,20.0\n"
    )
    history = read_val_angular_error_history(str(tmp_path))
    assert history == [28.0, 20.0]


def test_run_report_produces_all_expected_outputs(
    tmp_path, tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet
):
    checkpoints_dir = tmp_path / "checkpoints"
    cfg = make_model_cfg()
    for name in ["jepa_probe_v1", "jepa_finetune_v1", "mae_probe_v1", "mae_finetune_v1", "scratch_v1"]:
        write_fake_checkpoint(checkpoints_dir / name, epoch=0, cfg=cfg)

    # give the jepa finetune run a fake training history, to exercise Fig 2B's real path
    (checkpoints_dir / "jepa_finetune_v1" / "metrics.csv").write_text(
        "epoch,step,val/angular_error\n0,0,40.0\n1,1,25.0\n2,2,15.0\n"
    )

    output_dir = tmp_path / "report"
    run_report(
        test_batches=[1, 3],
        batch_dir=str(tiny_batch_dir),
        geometry_path=str(tiny_sensor_geometry_csv),
        meta_path=str(tiny_meta_parquet),
        max_pulses=256,
        checkpoint_dirs={
            "jepa_probe": str(checkpoints_dir / "jepa_probe_v1"),
            "jepa_finetune": str(checkpoints_dir / "jepa_finetune_v1"),
            "mae_probe": str(checkpoints_dir / "mae_probe_v1"),
            "mae_finetune": str(checkpoints_dir / "mae_finetune_v1"),
            "scratch": str(checkpoints_dir / "scratch_v1"),
        },
        model_cfg=cfg,
        output_dir=str(output_dir),
        batch_size=8,
    )

    for fname in [
        "fig1a_angular_error_vs_energy.png",
        "fig1b_method_comparison.png",
        "fig2b_training_curves.png",
        "fig3_embeddings.png",
        "fig4_masking_comparison.png",
        "table1.csv",
        "table1.md",
    ]:
        path = output_dir / fname
        assert path.exists(), f"missing {fname}"
        assert path.stat().st_size > 0


def test_run_report_skips_fig2b_when_no_history_present(
    tmp_path, tiny_batch_dir, tiny_sensor_geometry_csv, tiny_meta_parquet, capsys
):
    checkpoints_dir = tmp_path / "checkpoints"
    cfg = make_model_cfg()
    for name in ["jepa_probe_v1", "jepa_finetune_v1", "mae_probe_v1", "mae_finetune_v1", "scratch_v1"]:
        write_fake_checkpoint(checkpoints_dir / name, epoch=0, cfg=cfg)

    output_dir = tmp_path / "report"
    run_report(
        test_batches=[1, 3],
        batch_dir=str(tiny_batch_dir),
        geometry_path=str(tiny_sensor_geometry_csv),
        meta_path=str(tiny_meta_parquet),
        max_pulses=256,
        checkpoint_dirs={
            "jepa_probe": str(checkpoints_dir / "jepa_probe_v1"),
            "jepa_finetune": str(checkpoints_dir / "jepa_finetune_v1"),
            "mae_probe": str(checkpoints_dir / "mae_probe_v1"),
            "mae_finetune": str(checkpoints_dir / "mae_finetune_v1"),
            "scratch": str(checkpoints_dir / "scratch_v1"),
        },
        model_cfg=cfg,
        output_dir=str(output_dir),
        batch_size=8,
    )
    assert not (output_dir / "fig2b_training_curves.png").exists()
    assert "Fig 2B skipped" in capsys.readouterr().out
