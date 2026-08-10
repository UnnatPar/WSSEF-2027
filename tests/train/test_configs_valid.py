import glob

from train.config import flatten_sections, load_config

CONFIG_FILES = sorted(glob.glob("configs/*.yaml"))
SUPERVISED_CONFIGS = [
    "probe_jepa.yaml", "probe_mae.yaml", "finetune_jepa.yaml", "finetune_mae.yaml", "scratch.yaml",
]


def test_at_least_seven_configs_exist():
    assert len(CONFIG_FILES) == 7


def test_all_configs_parse():
    for path in CONFIG_FILES:
        cfg = load_config(path)
        assert cfg is not None


def test_pretrain_config_flattens_to_jepa_required_fields():
    cfg = load_config("configs/pretrain.yaml")
    flat = flatten_sections(cfg, "model", "masking", "training")
    for field in ["d", "L", "k", "ema_decay", "ratio_min", "ratio_max",
                  "n_clusters", "lr", "weight_decay", "epochs"]:
        assert hasattr(flat, field), f"missing {field}"


def test_pretrain_mae_config_flattens_to_required_fields():
    cfg = load_config("configs/pretrain_mae.yaml")
    flat = flatten_sections(cfg, "model", "training")
    for field in ["d", "L", "k", "mask_ratio", "lr", "weight_decay", "epochs"]:
        assert hasattr(flat, field), f"missing {field}"


def test_supervised_configs_flatten_to_required_fields():
    for name in SUPERVISED_CONFIGS:
        cfg = load_config(f"configs/{name}")
        flat = flatten_sections(cfg, "model", "training")
        for field in ["d", "L", "k", "freeze_encoder", "lr_encoder", "lr_heads",
                      "weight_decay", "lambda_direction", "lambda_classification",
                      "num_workers"]:
            assert hasattr(flat, field), f"{name} missing {field}"


def test_supervised_configs_do_not_use_the_full_594_batch_train_pool():
    # Real scale: 660 files, ~200k events/batch, ~130M total (per spec). The
    # full [1, 594] train-pool range would try to index ~117M events and
    # thrash KaggleParquetDataset's file cache -- the spec's own guidance is
    # "use 5M = ~25 batches" for fine-tuning, not the entire pool.
    for name in SUPERVISED_CONFIGS:
        cfg = load_config(f"configs/{name}")
        start, end = cfg.data.train_batches
        n_batches = end - start + 1
        assert n_batches <= 50, (
            f"{name}'s train_batches spans {n_batches} batches -- "
            "did this regress back to the full [1, 594] pool?"
        )


def test_scratch_config_has_no_checkpoint():
    cfg = load_config("configs/scratch.yaml")
    assert not hasattr(cfg.model, "checkpoint")


def test_probe_configs_freeze_encoder():
    for name in ["probe_jepa.yaml", "probe_mae.yaml"]:
        cfg = load_config(f"configs/{name}")
        assert cfg.model.freeze_encoder is True, f"{name} should freeze the encoder"


def test_finetune_and_scratch_configs_do_not_freeze_encoder():
    for name in ["finetune_jepa.yaml", "finetune_mae.yaml", "scratch.yaml"]:
        cfg = load_config(f"configs/{name}")
        assert cfg.model.freeze_encoder is False, f"{name} should not freeze the encoder"


def test_probe_and_finetune_configs_point_at_the_right_pretrain_checkpoint():
    jepa_source = "pretrain_jepa_v1"
    mae_source = "pretrain_mae_v3"  # v1 was the abandoned Colab run; v2 trained under the collapsed-ReLU dynamics fixed in models/pet.py; v3 is the real one
    for name in ["probe_jepa.yaml", "finetune_jepa.yaml"]:
        cfg = load_config(f"configs/{name}")
        assert jepa_source in cfg.model.checkpoint, f"{name} should point at a {jepa_source} checkpoint"
    for name in ["probe_mae.yaml", "finetune_mae.yaml"]:
        cfg = load_config(f"configs/{name}")
        assert mae_source in cfg.model.checkpoint, f"{name} should point at a {mae_source} checkpoint"


def test_probe_and_finetune_configs_reference_a_directory_not_a_hardcoded_epoch():
    # A hardcoded "epoch99.ckpt" would not exist if a real Colab run got
    # interrupted before reaching that epoch -- train/checkpoints.py resolves
    # the latest available checkpoint in this directory at runtime instead.
    for name in ["probe_jepa.yaml", "probe_mae.yaml", "finetune_jepa.yaml", "finetune_mae.yaml"]:
        cfg = load_config(f"configs/{name}")
        assert cfg.model.checkpoint.endswith("/"), (
            f"{name}'s model.checkpoint should be a directory, not a specific epoch file"
        )
        assert ".ckpt" not in cfg.model.checkpoint


def test_all_supervised_and_pretrain_configs_have_distinct_checkpoint_dirs():
    # This is the exact bug this config split fixes: jepa-probe and mae-probe
    # (and jepa-finetune / mae-finetune) used to share a checkpoint.dirpath
    # and silently overwrite each other's checkpoints when both experiments
    # ran in the same notebook session.
    dirpaths = set()
    for path in CONFIG_FILES:
        cfg = load_config(path)
        assert hasattr(cfg.checkpoint, "dirpath"), f"{path} missing checkpoint.dirpath"
        dirpaths.add(cfg.checkpoint.dirpath)
    assert len(dirpaths) == len(CONFIG_FILES)


def test_all_data_sections_have_geometry_path():
    for path in CONFIG_FILES:
        cfg = load_config(path)
        assert hasattr(cfg.data, "geometry_path"), f"{path} missing data.geometry_path"
