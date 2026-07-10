import torch

from data.masking import spatial_cluster_mask


def test_masks_are_disjoint_and_covering():
    torch.manual_seed(0)
    xyz = torch.randn(100, 3)
    context, target = spatial_cluster_mask(xyz, mask_ratio=0.5, n_clusters=4)
    assert context.dtype == torch.bool
    assert target.dtype == torch.bool
    assert not (context & target).any()
    assert (context | target).all()


def test_target_ratio_is_approximately_requested():
    torch.manual_seed(0)
    xyz = torch.randn(500, 3)
    _, target = spatial_cluster_mask(xyz, mask_ratio=0.5, n_clusters=4)
    ratio = target.float().mean().item()
    assert 0.45 <= ratio <= 0.55


def test_deterministic_given_seed():
    xyz = torch.randn(80, 3)
    torch.manual_seed(7)
    ctx1, tgt1 = spatial_cluster_mask(xyz, mask_ratio=0.4, n_clusters=4)
    torch.manual_seed(7)
    ctx2, tgt2 = spatial_cluster_mask(xyz, mask_ratio=0.4, n_clusters=4)
    assert torch.equal(ctx1, ctx2)
    assert torch.equal(tgt1, tgt2)


def test_handles_small_n_without_error():
    torch.manual_seed(0)
    xyz = torch.randn(5, 3)
    context, target = spatial_cluster_mask(xyz, mask_ratio=0.5, n_clusters=4)
    assert context.shape == (5,)
    assert target.shape == (5,)
    assert (context | target).all()


def test_zero_ratio_masks_nothing():
    xyz = torch.randn(30, 3)
    context, target = spatial_cluster_mask(xyz, mask_ratio=0.0, n_clusters=4)
    assert target.sum().item() == 0
    assert context.all()
