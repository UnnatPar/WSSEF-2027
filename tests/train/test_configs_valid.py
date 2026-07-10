import glob

from train.config import flatten_sections, load_config

CONFIG_FILES = sorted(glob.glob("configs/*.yaml"))


def test_at_least_five_configs_exist():
    assert len(CONFIG_FILES) == 5


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


def test_probe_finetune_scratch_configs_flatten_to_supervised_fields():
    for name in ["probe.yaml", "finetune.yaml", "scratch.yaml"]:
        cfg = load_config(f"configs/{name}")
        flat = flatten_sections(cfg, "model", "training")
        for field in ["d", "L", "k", "freeze_encoder", "lr_encoder", "lr_heads",
                      "weight_decay", "lambda_direction", "lambda_classification"]:
            assert hasattr(flat, field), f"{name} missing {field}"


def test_scratch_config_has_no_checkpoint():
    cfg = load_config("configs/scratch.yaml")
    assert not hasattr(cfg.model, "checkpoint")


def test_probe_config_freezes_encoder():
    cfg = load_config("configs/probe.yaml")
    assert cfg.model.freeze_encoder is True


def test_finetune_config_does_not_freeze_encoder():
    cfg = load_config("configs/finetune.yaml")
    assert cfg.model.freeze_encoder is False


def test_probe_finetune_scratch_configs_have_distinct_checkpoint_dirs():
    dirpaths = set()
    for name in ["probe.yaml", "finetune.yaml", "scratch.yaml"]:
        cfg = load_config(f"configs/{name}")
        assert hasattr(cfg.checkpoint, "dirpath"), f"{name} missing checkpoint.dirpath"
        dirpaths.add(cfg.checkpoint.dirpath)
    assert len(dirpaths) == 3


def test_all_data_sections_have_geometry_path():
    for path in CONFIG_FILES:
        cfg = load_config(path)
        assert hasattr(cfg.data, "geometry_path"), f"{path} missing data.geometry_path"
