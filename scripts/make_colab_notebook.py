"""Generates colab_train_neutrinojepa.ipynb.

Modeled on the reference colab_train.ipynb pattern (a different project,
LeWM/stable-worldmodel): every substantive operation -- installing deps,
downloading data, training -- is a subprocess.run/Popen call into a script
that already lives in this repo (scripts/download_data.sh, train/*.py). The
notebook itself only orchestrates: it never reimplements data loading,
training loops, or checkpoint logic inline. That's the same reason
train/dataset.py, train/pretrain.py etc. exist as real .py files instead of
being pasted into notebook cells -- this notebook just calls them.
"""

import nbformat as nbf

TITLE_CELL = """\
# NeutrinoJEPA Training

Run cells top to bottom. Checkpoints are written locally under `checkpoints/`
inside this VM -- they are NOT synced to Google Drive. `drive.mount()` blocks
on an interactive OAuth prompt a headless/automated session has no way to
answer, and (worse) silently degrades to writing a second local-only copy
instead of raising an error if it never completes -- confirmed by direct
testing, documented in the ledger's 2026-08-03 entry, and independently
already hit once before by the sibling `lewm-jamba` project. The durable
store is `unnat-brain/bin/colab-pull-ckpt.sh`, run from a normal machine (not
inside Colab), which mirrors `*.ckpt` files off the VM via `colab download`
on a timer. See that script's own header comment for why.

Runs all 3 experiments, each via the repo's own scripts (this notebook does
not reimplement any training logic):

1. **NeutrinoPET**: `train/pretrain_mae.py` -> `train/probe.py` -> `train/finetune.py`
2. **NeutrinoJEPAPET**: `train/pretrain.py` -> `train/probe.py` -> `train/finetune.py`
3. **PET+heads from scratch**: `train/finetune.py` on `configs/scratch.yaml` directly

Then evaluates all 5 resulting checkpoints (`jepa_probe`, `jepa_finetune`,
`mae_probe`, `mae_finetune`, `scratch`) via `eval/run_report.py`, producing
the 6 poster figures + summary table under `report/` (also pulled off the VM
externally, same as checkpoints, not synced to Drive).

Requires a Kaggle API token in Colab secrets as `KAGGLE_USERNAME` / `KAGGLE_KEY`.
"""

CONFIG_CELL = """\
REPO_URL = "https://github.com/UnnatPar/WSSEF-2027.git"
REPO_DIR = "/content/WSSEF-2027"
"""

INSTALL_CELL = """\
# -- Install torch + PyG (before cloning; these don't depend on the repo) --
# Install order matters: graphnet's own dependency chain has no upper bound
# on torch and will silently upgrade it past the pin if installed afterward
# (verified by direct testing while building this repo) -- torch/PyG are
# installed LAST, after requirements.txt, to guarantee the pinned versions win.
import subprocess, os, sys, glob

subprocess.run(
    ["pip", "install", "-q", "torch==2.3.0", "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=True,
)
print("torch installed")
"""

CLONE_CELL = """\
# -- Clone repo --
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
print("Repo ready at", REPO_DIR)
"""

REQUIREMENTS_CELL = """\
# -- Install the rest of the repo's dependencies, then re-pin torch + PyG last --
subprocess.run(
    ["pip", "install", "-q", "-r", os.path.join(REPO_DIR, "requirements.txt")],
    check=True,
)
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", "--no-deps", "torch==2.3.0",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=True,
)
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", "--no-deps",
     "torch-geometric==2.5.0", "torch-cluster==1.6.3", "torch-scatter==2.1.2",
     "torch-sparse==0.6.18",
     "-f", "https://data.pyg.org/whl/torch-2.3.0+cu121.html"],
    check=True,
)
print("Dependencies installed")
"""

DOWNLOAD_CELL = """\
# -- Kaggle credentials + data download (via the repo's own script) --
# `google.colab.userdata.get()` is also interactive-UI-only (same story as
# the Drive mount this notebook deliberately doesn't use -- see TITLE_CELL)
# and fails outright under headless/automated `colab exec` -- unlike Drive,
# this one fails loudly (an exception), so upload
# ~/.kaggle/access_token as a real file first (`colab upload`) instead of
# relying on this cell to fetch it from Colab secrets.
subprocess.run(["pip", "install", "-q", "-U", "kaggle"], check=True)  # 2.0.2 (Colab's stock
# version) can't auth with the KGAT_... access-token format; needs 2.2.4+.

DATA_DIR = os.path.join(REPO_DIR, "data")
train_dir = os.path.join(DATA_DIR, "train")

# Only the 116 batches configs/*.yaml actually reference (1-50 pretrain,
# 595-660 val+test) -- ~23M events, ~20GB, not the full 660-file/118GB/130M-
# event competition dataset, which doesn't fit on a Colab disk. Re-downloaded
# on every fresh session (no cross-session persistence for this -- see
# TITLE_CELL for why Drive isn't used, and note Kaggle's own download path is
# reliable, just not free time-wise, unlike the checkpoint-upload direction).
BATCH_RANGES = "1-50 595-660"
REQUIRED_BATCHES = 116

dl = subprocess.Popen(
    ["bash", "scripts/download_data.sh", DATA_DIR, BATCH_RANGES],
    cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
for line in iter(dl.stdout.readline, b""):
    sys.stdout.write(line.decode(errors="replace")); sys.stdout.flush()
dl.wait()
if dl.returncode != 0:
    raise SystemExit("download_data.sh failed")

n_batches = len(glob.glob(os.path.join(train_dir, "*.parquet")))
print(f"{n_batches} batch files ready")
if n_batches < REQUIRED_BATCHES:
    raise SystemExit(f"Expected {REQUIRED_BATCHES} batch files, found {n_batches} -- download incomplete")
"""

HELPERS_CELL = """\
# -- Shared helper: run a training stage --
def run_stage(script, config, watch_dir=None):
    \"\"\"Runs `python -u <script> --config <config>` via subprocess, streaming
    stdout. `watch_dir` (checkpoints/<run>) is informational only here -- this
    notebook does not sync checkpoints anywhere itself. The durable copy is
    pulled from OUTSIDE this VM by `unnat-brain/bin/colab-pull-ckpt.sh`, run
    from a normal machine against this session (see TITLE_CELL for why this
    isn't Drive-based). If resuming a prior session's progress, the caller is
    responsible for `colab upload`-ing the last pulled checkpoint into
    `watch_dir` as `last.ckpt` BEFORE invoking this cell -- train/*.py's own
    auto-resume picks it up from there with no notebook-side logic needed.
    \"\"\"
    if watch_dir is not None:
        os.makedirs(watch_dir, exist_ok=True)

    proc = subprocess.Popen(
        ["python", "-u", script, "--config", config],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in iter(proc.stdout.readline, b""):
        sys.stdout.write(line.decode(errors="replace")); sys.stdout.flush()
    proc.wait()

    if proc.returncode != 0:
        raise SystemExit(f"{script} failed with code {proc.returncode}")
    print(f"=== {script} ({config}) COMPLETE ===")
"""

EXPERIMENT_2_CELL = """\
# -- Experiment 2: NeutrinoJEPAPET (JEPA pre-training) --
# Each stage has its own dedicated config with its own checkpoint.dirpath
# (configs/probe_jepa.yaml, configs/finetune_jepa.yaml) -- no runtime config
# patching needed, and no risk of colliding with the MAE experiment's
# checkpoints below.
run_stage("train/pretrain.py", "configs/pretrain.yaml",
          watch_dir=os.path.join(REPO_DIR, "checkpoints/pretrain_jepa_v1"))
run_stage("train/probe.py", "configs/probe_jepa.yaml",
          watch_dir=os.path.join(REPO_DIR, "checkpoints/jepa_probe_v1"))
run_stage("train/finetune.py", "configs/finetune_jepa.yaml",
          watch_dir=os.path.join(REPO_DIR, "checkpoints/jepa_finetune_v1"))
"""

EXPERIMENT_1_CELL = """\
# -- Experiment 1: NeutrinoPET (MAE pre-training) --
run_stage("train/pretrain_mae.py", "configs/pretrain_mae.yaml",
          watch_dir=os.path.join(REPO_DIR, "checkpoints/pretrain_mae_v1"))
run_stage("train/probe.py", "configs/probe_mae.yaml",
          watch_dir=os.path.join(REPO_DIR, "checkpoints/mae_probe_v1"))
run_stage("train/finetune.py", "configs/finetune_mae.yaml",
          watch_dir=os.path.join(REPO_DIR, "checkpoints/mae_finetune_v1"))
"""

EXPERIMENT_3_CELL = """\
# -- Experiment 3: PET+heads from scratch (no pre-training, no checkpoint) --
run_stage("train/finetune.py", "configs/scratch.yaml",
          watch_dir=os.path.join(REPO_DIR, "checkpoints/scratch_v1"))
print("=== All 3 experiments complete ===")
print("Final checkpoint directories: jepa_probe_v1, jepa_finetune_v1,")
print("mae_probe_v1, mae_finetune_v1, scratch_v1")
"""

REPORT_CELL = """\
# -- Evaluation: turn the 5 checkpoints into the 6 poster figures + table --
# All the actual eval logic (inference, metrics, plotting) lives in
# eval/run_report.py -- this cell only runs it. Not run_stage(): this is a
# one-shot batch job, not an incremental training run, so there's no
# checkpoint directory to watch -- and its CLI takes different flags
# (--checkpoints-dir/--data-config/--output-dir, not --config). Output stays
# local under report/ -- pull it off the VM the same way as checkpoints
# (`colab download`), not via Drive; see TITLE_CELL for why.
REPORT_DIR = os.path.join(REPO_DIR, "report")
report_proc = subprocess.Popen(
    ["python", "-u", "eval/run_report.py",
     "--checkpoints-dir", "checkpoints",
     "--data-config", "configs/finetune_jepa.yaml",
     "--output-dir", "report"],
    cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
for line in iter(report_proc.stdout.readline, b""):
    sys.stdout.write(line.decode(errors="replace")); sys.stdout.flush()
report_proc.wait()
if report_proc.returncode != 0:
    raise SystemExit(f"eval/run_report.py failed with code {report_proc.returncode}")

print(f"Report written to {REPORT_DIR}:")
for fname in sorted(os.listdir(REPORT_DIR)):
    print(" -", fname)
"""


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(TITLE_CELL),
        nbf.v4.new_code_cell(CONFIG_CELL),
        nbf.v4.new_code_cell(INSTALL_CELL),
        nbf.v4.new_code_cell(CLONE_CELL),
        nbf.v4.new_code_cell(REQUIREMENTS_CELL),
        nbf.v4.new_code_cell(DOWNLOAD_CELL),
        nbf.v4.new_code_cell(HELPERS_CELL),
        nbf.v4.new_code_cell(EXPERIMENT_2_CELL),
        nbf.v4.new_code_cell(EXPERIMENT_1_CELL),
        nbf.v4.new_code_cell(EXPERIMENT_3_CELL),
        nbf.v4.new_code_cell(REPORT_CELL),
    ]
    return nb


def main(output_path: str = "colab_train_neutrinojepa.ipynb"):
    nb = build_notebook()
    with open(output_path, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
