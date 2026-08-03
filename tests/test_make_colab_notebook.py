import nbformat as nbf

from scripts.make_colab_notebook import build_notebook


def test_notebook_is_valid_nbformat():
    nb = build_notebook()
    nbf.validate(nb)  # raises if malformed


def test_notebook_has_expected_cell_count():
    nb = build_notebook()
    assert len(nb["cells"]) == 12


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
    assert "shutil.copytree" in report_source  # syncs the report dir to Drive


def test_notebook_watcher_does_not_let_drive_sync_errors_kill_the_thread():
    # An unhandled OSError (e.g. Drive quota exhausted) inside the daemon
    # watcher thread would silently kill it -- checkpoint syncing stops for
    # the rest of that stage with no visible error to the user.
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    copy_idx = source.index("shutil.copy2(path, drive_dst)")
    try_idx = source.rindex("try:", 0, copy_idx)
    except_idx = source.index("except OSError", copy_idx)
    assert try_idx < copy_idx < except_idx, (
        "shutil.copy2 in the watcher must be inside a try/except OSError"
    )


def test_notebook_restores_checkpoints_from_drive_before_starting_a_stage():
    # A fresh session's local checkpoint dir is always empty (fresh clone).
    # Without restoring from Drive first, train/*.py's auto-resume (which only
    # checks the local dir for last.ckpt) would silently restart every stage
    # from step 0 on every session death, even though Drive has the progress.
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    restore_idx = source.index("Restored checkpoints from Drive")
    copytree_idx = source.rindex("shutil.copytree(drive_src, watch_dir", 0, restore_idx + 200)
    start_stage_idx = source.index('proc = subprocess.Popen(\n        ["python", "-u", script,')
    assert copytree_idx < start_stage_idx, (
        "checkpoints must be restored from Drive before the training subprocess starts"
    )


def test_notebook_restores_dataset_from_drive_instead_of_always_redownloading():
    # Re-downloading ~20GB from Kaggle on every session death (expected, not
    # rare, given the ~1.5-2hr session lifetime) burns a large chunk of each
    # new session's short lifetime before training can even resume.
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    assert "DRIVE_DATA_DIR" in source
    assert "drive_train_dir" in source
    restore_idx = source.index("Restoring dataset from Drive")
    download_idx = source.index('["bash", "scripts/download_data.sh"')
    assert restore_idx < download_idx, (
        "must check for a Drive-persisted dataset before falling back to download_data.sh"
    )


def test_notebook_watcher_resyncs_last_ckpt_when_it_changes_not_just_once():
    # last.ckpt (save_last=True) is the SAME filename rewritten every
    # checkpoint interval, not a new file each time. A plain "seen this path
    # before" set would sync it to Drive exactly once, ever, leaving Drive
    # with a permanently stale snapshot for the rest of the stage. Sync state
    # must be keyed on mtime (or something that changes on rewrite), not path.
    source = "\n".join(cell["source"] for cell in build_notebook()["cells"])
    assert "_synced_mtime" in source
    assert "getmtime" in source


def test_main_writes_a_loadable_notebook_file(tmp_path):
    from scripts.make_colab_notebook import main

    out_path = tmp_path / "test_notebook.ipynb"
    main(str(out_path))
    assert out_path.exists()

    with open(out_path) as f:
        loaded = nbf.read(f, as_version=4)
    nbf.validate(loaded)
    assert len(loaded["cells"]) == 12
