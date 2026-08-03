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
        "plot_angular_error_vs_energy", "plot_method_comparison_bars",
        "plot_embedding_projection", "plot_masking_comparison", "TSNE",
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
        "configs/probe_jepa.yaml", "configs/probe_mae.yaml",
        "configs/finetune_jepa.yaml", "configs/finetune_mae.yaml",
        "configs/scratch.yaml",
    ]:
        assert config in source, f"missing reference to {config}"


def test_notebook_gives_each_experiment_stage_a_distinct_checkpoint_dir():
    # This is the exact bug the config split fixed: jepa-probe and mae-probe
    # (and jepa/mae-finetune) must never share a watch_dir/checkpoint folder,
    # or one experiment's checkpoints silently overwrite the other's.
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    for watch_dir in [
        "checkpoints/pretrain_jepa_v1", "checkpoints/pretrain_mae_v1",
        "checkpoints/jepa_probe_v1", "checkpoints/mae_probe_v1",
        "checkpoints/jepa_finetune_v1", "checkpoints/mae_finetune_v1",
        "checkpoints/scratch_v1",
    ]:
        assert watch_dir in source, f"missing distinct checkpoint dir {watch_dir}"


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


def test_notebook_runs_eval_report_after_all_experiments():
    cells = build_notebook()["cells"]
    sources = [c["source"] for c in cells if c["cell_type"] == "code"]

    exp3_idx = next(i for i, s in enumerate(sources) if "Experiment 3" in s)
    report_idx = next(i for i, s in enumerate(sources) if "eval/run_report.py" in s)
    assert report_idx > exp3_idx, "eval/run_report.py must run after all 3 experiments"

    report_source = sources[report_idx]
    assert "--checkpoints-dir" in report_source
    assert "--output-dir" in report_source


def test_notebook_does_not_use_google_drive():
    # drive.mount() blocks on an interactive OAuth prompt a headless/automated
    # colab exec session has no way to answer -- confirmed by direct testing
    # (ledger 2026-08-03), and independently already hit once before by the
    # sibling lewm-jamba project (COLAB_RUNBOOK.md, 2026-07-30). Worse than a
    # clean failure: if drive.mount() never completes, /content/drive/... is
    # just an ordinary path on the VM's own ephemeral disk, so code that
    # writes there "succeeds" with no error while silently producing a second
    # local-only copy that dies with the VM exactly like the original. The
    # durable store is unnat-brain/bin/colab-pull-ckpt.sh, run externally.
    # Only code cells matter here -- the title cell's markdown explains, in
    # prose, why drive.mount() specifically isn't used, which legitimately
    # mentions the string "drive.mount()".
    code_source = "\n".join(
        cell["source"] for cell in build_notebook()["cells"] if cell["cell_type"] == "code"
    )
    assert "drive.mount(" not in code_source
    assert "google.colab import drive" not in code_source
    assert "DRIVE_CHECKPOINT_DIR" not in code_source
    assert "DRIVE_DATA_DIR" not in code_source


def test_notebook_documents_the_external_pull_based_checkpoint_story():
    # There's no in-notebook checkpoint-sync code anymore (see
    # test_notebook_does_not_use_google_drive) -- the durable copy comes from
    # running colab-pull-ckpt.sh against this session from outside the VM.
    # The notebook must at least point at that, or a reader has no way to
    # know checkpoints aren't otherwise going anywhere durable.
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    assert "colab-pull-ckpt.sh" in source
    assert "last.ckpt" in source


def test_main_writes_a_loadable_notebook_file(tmp_path):
    from scripts.make_colab_notebook import main

    out_path = tmp_path / "test_notebook.ipynb"
    main(str(out_path))
    assert out_path.exists()

    with open(out_path) as f:
        loaded = nbf.read(f, as_version=4)
    nbf.validate(loaded)
    assert len(loaded["cells"]) == 11
