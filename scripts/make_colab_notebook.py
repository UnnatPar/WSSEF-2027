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

Run cells top to bottom. Checkpoints sync to Google Drive as they're written.

Runs all 3 experiments, each via the repo's own scripts (this notebook does
not reimplement any training logic):

1. **NeutrinoPET**: `train/pretrain_mae.py` -> `train/probe.py` -> `train/finetune.py`
2. **NeutrinoJEPAPET**: `train/pretrain.py` -> `train/probe.py` -> `train/finetune.py`
3. **PET+heads from scratch**: `train/finetune.py` on `configs/scratch.yaml` directly

Then evaluates all 5 resulting checkpoints (`jepa_probe`, `jepa_finetune`,
`mae_probe`, `mae_finetune`, `scratch`) via `eval/run_report.py`, producing
the 6 poster figures + summary table under `report/`, synced to Drive.

Requires a Kaggle API token in Colab secrets as `KAGGLE_USERNAME` / `KAGGLE_KEY`.
"""

CONFIG_CELL = """\
REPO_URL = "https://github.com/UnnatPar/WSSEF-2027.git"
REPO_DIR = "/content/WSSEF-2027"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/neutrinojepa_checkpoints"
DRIVE_DATA_DIR = "/content/drive/MyDrive/neutrinojepa_data"
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
     "torch-sparse==0.6.18",
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

# Only the 116 batches configs/*.yaml actually reference (1-50 pretrain,
# 595-660 val+test) -- ~23M events, ~20GB, not the full 660-file/118GB/130M-
# event competition dataset, which doesn't fit on a Colab disk.
BATCH_RANGES = "1-50 595-660"
REQUIRED_BATCHES = 116

# A fresh session's local disk is always empty -- re-downloading ~20GB from
# Kaggle on every session death (a session dying is the expected case, not
# an edge case, per the ~1.5-2hr lifetime limit) would burn a large chunk of
# each new session's short lifetime before training even resumes. Restore
# from Drive first if a prior session already did this download.
drive_train_dir = os.path.join(DRIVE_DATA_DIR, "train")
if os.path.exists(drive_train_dir) and len(glob.glob(os.path.join(drive_train_dir, "*.parquet"))) >= REQUIRED_BATCHES:
    print(f"Restoring dataset from Drive ({drive_train_dir}) instead of re-downloading...")
    shutil.copytree(DRIVE_DATA_DIR, DATA_DIR, dirs_exist_ok=True)

if not os.path.exists(train_dir) or len(glob.glob(os.path.join(train_dir, "*.parquet"))) < REQUIRED_BATCHES:
    dl = subprocess.Popen(
        ["bash", "scripts/download_data.sh", DATA_DIR, BATCH_RANGES],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in iter(dl.stdout.readline, b""):
        sys.stdout.write(line.decode(errors="replace")); sys.stdout.flush()
    dl.wait()
    if dl.returncode != 0:
        raise SystemExit("download_data.sh failed")
    # Persist to Drive so the next session (after this one dies) restores
    # instead of re-downloading.
    os.makedirs(DRIVE_DATA_DIR, exist_ok=True)
    shutil.copytree(DATA_DIR, DRIVE_DATA_DIR, dirs_exist_ok=True)
    print(f"Dataset backed up to Drive at {DRIVE_DATA_DIR}")

n_batches = len(glob.glob(os.path.join(train_dir, "*.parquet")))
print(f"{n_batches} batch files ready")
if n_batches < REQUIRED_BATCHES:
    raise SystemExit(f"Expected {REQUIRED_BATCHES} batch files, found {n_batches} -- download incomplete")
"""

HELPERS_CELL = """\
# -- Shared helper: run a training stage, syncing new checkpoints to Drive --
def run_stage(script, config, watch_dir=None):
    \"\"\"Runs `python -u <script> --config <config>` via subprocess, streaming
    stdout, while a background thread mirrors any *.ckpt files written under
    watch_dir to Drive as they appear (debounced on stable file size, same
    pattern as the reference notebook's checkpoint watcher).

    Restores from Drive into watch_dir FIRST, before starting the subprocess:
    a fresh session's local checkpoint dir is always empty (fresh clone), but
    train/*.py's auto-resume logic only checks the LOCAL dir for last.ckpt --
    without this restore, every session death would silently restart this
    stage from step 0 instead of resuming, even though Drive has the progress.
    \"\"\"
    _synced_mtime = {}  # path -> mtime already copied to Drive

    if watch_dir is not None:
        drive_src = os.path.join(DRIVE_CHECKPOINT_DIR, os.path.basename(watch_dir))
        if os.path.isdir(drive_src) and glob.glob(os.path.join(drive_src, "*.ckpt")):
            os.makedirs(watch_dir, exist_ok=True)
            shutil.copytree(drive_src, watch_dir, dirs_exist_ok=True)
            print(f"=== Restored checkpoints from Drive: {drive_src} -> {watch_dir} ===", flush=True)
            # These are already in sync with Drive -- seed the watcher so it
            # doesn't immediately re-copy them back on its first poll.
            for path in glob.glob(os.path.join(watch_dir, "*.ckpt")):
                try:
                    _synced_mtime[path] = os.path.getmtime(path)
                except OSError:
                    pass

    _stop = threading.Event()

    def watcher():
        if watch_dir is None:
            return
        while not _stop.is_set():
            for path in glob.glob(os.path.join(watch_dir, "*.ckpt")):
                # last.ckpt (from save_last=True) is the SAME filename
                # rewritten every checkpoint interval, not a new file each
                # time -- keying sync-state on path alone (as a plain "seen"
                # set) would sync it to Drive exactly once, ever, and then
                # silently never again for the rest of this stage, leaving
                # Drive with a permanently stale snapshot. Keying on mtime
                # instead means every real rewrite is detected and re-synced.
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if _synced_mtime.get(path) == mtime:
                    continue
                try:
                    s1 = os.path.getsize(path); time.sleep(5); s2 = os.path.getsize(path)
                    if s1 != s2 or s1 == 0 or os.path.getmtime(path) != mtime:
                        continue  # still being written, or changed again mid-debounce; catch it next poll
                except OSError:
                    continue
                drive_dst = os.path.join(DRIVE_CHECKPOINT_DIR, os.path.basename(watch_dir), os.path.basename(path))
                try:
                    os.makedirs(os.path.dirname(drive_dst), exist_ok=True)
                    shutil.copy2(path, drive_dst)
                except OSError as e:
                    # An unhandled exception here would silently kill this
                    # daemon thread (e.g. Drive quota exhausted mid-run) --
                    # checkpoint syncing would stop for the rest of this
                    # stage with no visible error. Log and retry next poll
                    # instead of leaving `path` unmarked-but-silently-lost.
                    print(f"=== CHECKPOINT SYNC FAILED for {os.path.basename(path)}: {e!r} "
                          f"(will retry) ===", flush=True)
                    continue
                _synced_mtime[path] = mtime
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

REPORT_CELL = """\
# -- Evaluation: turn the 5 checkpoints into the 6 poster figures + table --
# All the actual eval logic (inference, metrics, plotting) lives in
# eval/run_report.py -- this cell only runs it and syncs the output to Drive.
# Not run_stage(): this is a one-shot batch job, not an incremental training
# run, so there's no checkpoint directory to watch -- and its CLI takes
# different flags (--checkpoints-dir/--data-config/--output-dir, not --config).
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

drive_report_dir = os.path.join(DRIVE_CHECKPOINT_DIR, "report")
shutil.copytree(REPORT_DIR, drive_report_dir, dirs_exist_ok=True)
print(f"Report synced to {drive_report_dir}")
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
        nbf.v4.new_code_cell(DRIVE_CELL),
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
