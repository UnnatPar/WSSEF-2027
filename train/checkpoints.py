import glob
import os


def latest_checkpoint(dirpath: str) -> str:
    """Returns the most recent *.ckpt file in dirpath.

    Used instead of hardcoding a specific epoch filename (e.g. "epoch99.ckpt")
    in configs: a real Colab run can be interrupted by session limits or
    manual stops well before the configured epoch count, and a hardcoded
    filename would then not exist -- a probe/finetune run would fail before
    it even starts. Resolving "whatever's actually there" at runtime instead
    means partial pretraining runs still produce usable checkpoints.

    Training uses time-interval checkpointing (every 15 min, not per-epoch --
    a real epoch can run for hours) plus `save_last=True`, which writes a
    `last.ckpt` alongside the numbered file at every save. `last.ckpt` is
    always at least as recent as the numbered file, but has no digits in its
    name -- prefer it explicitly rather than falling through to the epoch/step
    sort below, which would otherwise pick a numbered checkpoint that's stale
    by up to one checkpoint interval.
    """
    last_ckpt = os.path.join(dirpath, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    files = glob.glob(os.path.join(dirpath, "*.ckpt"))
    if not files:
        raise FileNotFoundError(f"no checkpoints found in {dirpath}")

    def epoch_num(path: str) -> int:
        digits = "".join(c for c in os.path.basename(path) if c.isdigit())
        return int(digits) if digits else -1

    return max(files, key=epoch_num)
