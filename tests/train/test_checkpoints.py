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
