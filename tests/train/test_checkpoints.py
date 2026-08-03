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


def test_latest_checkpoint_prefers_last_ckpt_over_a_newer_looking_numbered_file(tmp_path):
    # last.ckpt has no digits in its name, so the epoch/step sort alone would
    # never pick it -- but save_last=True means it's always at least as
    # recent as any numbered file, even one whose embedded epoch/step number
    # looks larger. Preferring it explicitly avoids loading a stale
    # checkpoint that's up to one time-interval old.
    (tmp_path / "epoch0099-step000999999.ckpt").write_bytes(b"fake")
    (tmp_path / "last.ckpt").write_bytes(b"fake")
    picked = latest_checkpoint(str(tmp_path))
    assert picked.endswith("last.ckpt")
