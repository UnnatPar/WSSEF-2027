import glob
import os


def latest_checkpoint(dirpath: str) -> str:
    """Returns the most recently WRITTEN *.ckpt file in dirpath (by mtime).

    Used instead of hardcoding a specific epoch filename (e.g. "epoch99.ckpt")
    in configs: a real Colab run can be interrupted by session limits or
    manual stops well before the configured epoch count, and a hardcoded
    filename would then not exist -- a probe/finetune run would fail before
    it even starts. Resolving "whatever's actually there" at runtime instead
    means partial pretraining runs still produce usable checkpoints.

    Previously this preferred a literal "last.ckpt" by name, on the
    assumption that save_last=True keeps it at least as recent as any
    numbered file. That assumption is false in a real, reproduced scenario:
    a checkpoint dir seeded with an old "last.ckpt" (e.g. manually uploaded
    to resume a session) can make Lightning detect a naming collision with
    the *current* run's own last-checkpoint tracking and write "last-v1.ckpt"
    instead of overwriting it -- silently leaving the seeded, stale
    "last.ckpt" in place while real training moves on. Verified directly: a
    production run reached step 147,454 while "last.ckpt" still pointed at
    the step-8,154 seed checkpoint from ~139k steps earlier (confirmed by
    md5: "last-v1.ckpt" matched the numbered step-147454 file exactly,
    "last.ckpt" did not) -- name-based preference would have silently loaded
    the wrong, far-less-trained encoder into every probe/finetune run.
    mtime is the actual ground truth for "most recently written" and isn't
    fooled by any naming/versioning quirk.
    """
    files = glob.glob(os.path.join(dirpath, "*.ckpt"))
    if not files:
        raise FileNotFoundError(f"no checkpoints found in {dirpath}")
    return max(files, key=os.path.getmtime)
