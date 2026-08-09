import os

import pytest

from train.checkpoints import latest_checkpoint


def test_latest_checkpoint_picks_highest_epoch(tmp_path):
    for epoch in [0, 1, 11, 2]:
        (tmp_path / f"epoch{epoch}.ckpt").write_bytes(b"fake")
    picked = latest_checkpoint(str(tmp_path))
    assert picked.endswith("epoch11.ckpt")


def test_latest_checkpoint_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(str(tmp_path / "empty"))


def test_latest_checkpoint_works_when_only_one_checkpoint_exists(tmp_path):
    (tmp_path / "epoch0.ckpt").write_bytes(b"fake")
    picked = latest_checkpoint(str(tmp_path))
    assert picked.endswith("epoch0.ckpt")


def test_latest_checkpoint_prefers_a_more_recently_written_last_ckpt(tmp_path):
    # last.ckpt has no digits in its name, so the epoch/step sort alone would
    # never pick it on name alone -- but when it really is the most recently
    # written file (the normal case under save_last=True), it must still win.
    (tmp_path / "epoch0099-step000999999.ckpt").write_bytes(b"fake")
    (tmp_path / "last.ckpt").write_bytes(b"fake")
    picked = latest_checkpoint(str(tmp_path))
    assert picked.endswith("last.ckpt")


def test_latest_checkpoint_ignores_a_stale_last_ckpt_by_name(tmp_path):
    """Real, reproduced production bug: a checkpoint dir seeded with an old
    "last.ckpt" (e.g. manually uploaded to resume a session) can make
    Lightning write "last-v1.ckpt" instead of overwriting the stale file when
    it detects a naming collision with the current run's own tracking --
    leaving "last.ckpt" pointing at a far older step than the real latest
    checkpoint. Verified directly: a production run reached step 147,454
    while "last.ckpt" still pointed at a step-8,154 seed checkpoint from
    ~139k steps earlier. Name-based preference of "last.ckpt" would silently
    pick the wrong, far-less-trained checkpoint; mtime must win instead."""
    stale_last = tmp_path / "last.ckpt"
    stale_last.write_bytes(b"stale, seeded from an old session")
    real_latest = tmp_path / "epoch0003-step000147454.ckpt"
    real_latest.write_bytes(b"real latest, from actual training progress")
    # Simulate the stale file genuinely being older on disk, not just
    # alphabetically/numerically "earlier" -- mtime is what the fix checks.
    old_time = os.path.getmtime(stale_last) - 3600
    os.utime(stale_last, (old_time, old_time))

    picked = latest_checkpoint(str(tmp_path))
    assert picked.endswith("epoch0003-step000147454.ckpt")
