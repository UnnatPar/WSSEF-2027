"""Generates colab_train_neutrinojepa.ipynb.

Modeled on the reference colab_train.ipynb pattern (a different project,
LeWM/stable-worldmodel): every substantive operation -- installing deps,
downloading data, training -- is a subprocess.run/Popen call into a script
that already lives in this repo (scripts/download_data.sh, train/*.py). The
notebook itself only orchestrates: it never reimplements data loading,
training loops, or checkpoint logic inline. That's the same reason
train/data.py, train/pretrain.py etc. exist as real .py files instead of
being pasted into notebook cells -- this notebook just calls them.
"""

import nbformat as nbf

TITLE_CELL = """\
# NeutrinoJEPA Training

Run cells top to bottom. Checkpoints sync to Google Drive as they're written.

Runs all 3 experiments, each via the repo's own scripts (this notebook does
not reimplement any training logic):

1. **NeutrinoPET**: `train/pretrain_mae.py` -> `train/probe.py` -> `train/finetune.py`
2. **NeutrinoJEPAPET**: `train/pretrain.py` -> `train/probe.py` -> `train/finetune.py`
3. **PET+heads from scratch**: `train/finetune.py` on `configs/scratch.yaml` directly

Requires a Kaggle API token in Colab secrets as `KAGGLE_USERNAME` / `KAGGLE_KEY`.
"""

CONFIG_CELL = """\
REPO_URL = "https://github.com/UnnatPar/WSSEF-2027.git"
REPO_DIR = "/content/WSSEF-2027"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/neutrinojepa_checkpoints"
"""

INSTALL_CELL = """\
# -- Install torch + PyG (before cloning; these don't depend on the repo) --
# Install order matters: graphnet's own dependency chain has no upper bound
# on torch and will silently upgrade it past the pin if installed afterward
# (verified by direct testing while building this repo) -- torch/PyG are
# installed LAST, after requirements.txt, to guarantee the pinned versions win.
import subprocess, os, sys, threading, shutil, glob, time

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
     "-f", "https://data.pyg.org/whl/torch-2.3.0+cu121.html"],
    check=True,
)
print("Dependencies installed")
"""

DRIVE_CELL = """\
# -- Mount Google Drive --
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
print(f"Checkpoints will sync to: {DRIVE_CHECKPOINT_DIR}")
"""

DOWNLOAD_CELL = """\
# -- Kaggle credentials + data download (via the repo's own script) --
from google.colab import userdata

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
with open(kaggle_json, "w") as f:
    f.write('{"username": "%s", "key": "%s"}' % (
        userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY"),
    ))
os.chmod(kaggle_json, 0o600)
subprocess.run(["pip", "install", "-q", "kaggle"], check=True)

DATA_DIR = os.path.join(REPO_DIR, "data")
train_dir = os.path.join(DATA_DIR, "train")

if not os.path.exists(train_dir) or len(glob.glob(os.path.join(train_dir, "*.parquet"))) < 660:
    dl = subprocess.Popen(
        ["bash", "scripts/download_data.sh", DATA_DIR],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in iter(dl.stdout.readline, b""):
        sys.stdout.write(line.decode(errors="replace")); sys.stdout.flush()
    dl.wait()
    if dl.returncode != 0:
        raise SystemExit("download_data.sh failed")

n_batches = len(glob.glob(os.path.join(train_dir, "*.parquet")))
print(f"{n_batches} batch files ready")
if n_batches < 660:
    raise SystemExit(f"Expected 660 batch files, found {n_batches} -- download incomplete")
"""

HELPERS_CELL = """\
# -- Shared helper: run a training stage, syncing new checkpoints to Drive --
def run_stage(script, config, watch_dir=None):
    \"\"\"Runs `python -u <script> --config <config>` via subprocess, streaming
    stdout, while a background thread mirrors any *.ckpt files written under
    watch_dir to Drive as they appear (debounced on stable file size, same
    pattern as the reference notebook's checkpoint watcher).
    \"\"\"
    _stop = threading.Event()
    _seen = set()

    def watcher():
        if watch_dir is None:
            return
        while not _stop.is_set():
            for path in glob.glob(os.path.join(watch_dir, "*.ckpt")):
                if path in _seen:
                    continue
                try:
                    s1 = os.path.getsize(path); time.sleep(5); s2 = os.path.getsize(path)
                    if s1 != s2 or s1 == 0:
                        continue
                except OSError:
                    continue
                drive_dst = os.path.join(DRIVE_CHECKPOINT_DIR, os.path.basename(watch_dir), os.path.basename(path))
                os.makedirs(os.path.dirname(drive_dst), exist_ok=True)
                shutil.copy2(path, drive_dst)
                _seen.add(path)
                print(f"=== CHECKPOINT SYNCED: {os.path.basename(path)} -> Drive ===", flush=True)
            _stop.wait(30)

    threading.Thread(target=watcher, daemon=True).start()

    proc = subprocess.Popen(
        ["python", "-u", script, "--config", config],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in iter(proc.stdout.readline, b""):
        sys.stdout.write(line.decode(errors="replace")); sys.stdout.flush()
    proc.wait()
    _stop.set()

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


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(TITLE_CELL),
        nbf.v4.new_code_cell(CONFIG_CELL),
        nbf.v4.new_code_cell(INSTALL_CELL),
        nbf.v4.new_code_cell(CLONE_CELL),
        nbf.v4.new_code_cell(REQUIREMENTS_CELL),
        nbf.v4.new_code_cell(DRIVE_CELL),
        nbf.v4.new_code_cell(DOWNLOAD_CELL),
        nbf.v4.new_code_cell(HELPERS_CELL),
        nbf.v4.new_code_cell(EXPERIMENT_2_CELL),
        nbf.v4.new_code_cell(EXPERIMENT_1_CELL),
        nbf.v4.new_code_cell(EXPERIMENT_3_CELL),
    ]
    return nb


def main(output_path: str = "colab_train_neutrinojepa.ipynb"):
    nb = build_notebook()
    with open(output_path, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
