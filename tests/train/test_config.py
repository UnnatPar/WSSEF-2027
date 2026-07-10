from train.config import flatten_sections, load_config


def test_load_config_nested_access(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model:\n"
        "  d: 256\n"
        "  L: 6\n"
        "masking:\n"
        "  ratio_min: 0.4\n"
    )
    cfg = load_config(str(path))
    assert cfg.model.d == 256
    assert cfg.model.L == 6
    assert cfg.masking.ratio_min == 0.4


def test_flatten_sections_merges_and_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model:\n"
        "  d: 256\n"
        "  shared: 1\n"
        "training:\n"
        "  lr: 0.001\n"
        "  shared: 2\n"
    )
    cfg = load_config(str(path))
    flat = flatten_sections(cfg, "model", "training")
    assert flat.d == 256
    assert flat.lr == 0.001
    assert flat.shared == 2  # later section wins on collision
