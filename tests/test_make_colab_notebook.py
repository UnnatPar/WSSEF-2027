import nbformat as nbf

from scripts.make_colab_notebook import build_notebook


def test_notebook_is_valid_nbformat():
    nb = build_notebook()
    nbf.validate(nb)  # raises if malformed


def test_notebook_has_expected_cell_count():
    nb = build_notebook()
    assert len(nb["cells"]) == 11


def test_notebook_never_reimplements_training_logic():
    # The notebook must only orchestrate via subprocess calls into the
    # repo's own scripts -- it must not import/define models, losses, or
    # data-loading logic inline. Only code cells count; the markdown title
    # cell naturally mentions the project/experiment names.
    code_source = "\n".join(
        cell["source"] for cell in build_notebook()["cells"] if cell["cell_type"] == "code"
    )
    forbidden_substrings = [
        "PETEncoder", "NeutrinoJEPA(", "SupervisedFineTune", "MAEPretrain",
        "spatial_cluster_mask", "VonMisesFisher", "ParquetDataset",
    ]
    for forbidden in forbidden_substrings:
        assert forbidden not in code_source, (
            f"notebook code reimplements {forbidden} instead of shelling out"
        )


def test_notebook_references_every_stage_script_and_config():
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    for script in [
        "train/pretrain.py", "train/pretrain_mae.py",
        "train/probe.py", "train/finetune.py",
        "scripts/download_data.sh",
    ]:
        assert script in source, f"missing reference to {script}"
    for config in [
        "configs/pretrain.yaml", "configs/pretrain_mae.yaml",
        "configs/probe.yaml", "configs/finetune.yaml", "configs/scratch.yaml",
    ]:
        assert config in source, f"missing reference to {config}"


def test_notebook_installs_torch_last_after_requirements():
    # Verifies install ordering matches the real finding from building this
    # repo: graphnet's dependency chain can silently upgrade torch past the
    # pin if torch isn't reinstalled after requirements.txt. Find the cell
    # with the actual `pip install -r requirements.txt` command (not just
    # any cell that mentions the filename in a comment).
    cells = build_notebook()["cells"]
    sources = [c["source"] for c in cells if c["cell_type"] == "code"]
    requirements_idx = next(
        i for i, s in enumerate(sources) if '"-r",' in s and "requirements.txt" in s
    )
    assert "torch==2.3.0" in sources[requirements_idx]
    assert "--force-reinstall" in sources[requirements_idx]


def test_notebook_covers_all_3_experiments():
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    assert "Experiment 2" in source
    assert "Experiment 1" in source
    assert "Experiment 3" in source


def test_main_writes_a_loadable_notebook_file(tmp_path):
    from scripts.make_colab_notebook import main

    out_path = tmp_path / "test_notebook.ipynb"
    main(str(out_path))
    assert out_path.exists()

    with open(out_path) as f:
        loaded = nbf.read(f, as_version=4)
    nbf.validate(loaded)
    assert len(loaded["cells"]) == 11
