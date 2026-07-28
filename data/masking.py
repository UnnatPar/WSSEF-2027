import torch
from torch import Tensor


def spatial_cluster_mask(
    xyz: Tensor,
    mask_ratio: float,
    n_clusters: int,
) -> tuple[Tensor, Tensor]:
    """Greedily grow `n_clusters` spatial clusters of masked pulses until
    `mask_ratio * N` pulses are covered, forcing the model to predict entire
    unobserved detector sub-volumes instead of trivially interpolating
    adjacent unmasked pulses.

    Counts are tracked as plain Python ints via tensor `.numel()` (metadata
    only -- never syncs) instead of `.sum().item()`, and each cluster's
    growth is one vectorized selection instead of a per-pulse Python loop.
    Same nearest-anchor-first growth order and stopping condition as a
    literal per-pulse loop would produce -- this is an implementation
    change, not an algorithm change. Matters because this runs once per
    event inside a Python loop over the whole batch (models/jepa.py) --
    `.item()` there means one synchronizing GPU round trip per pulse, per
    event, per step.
    """
    n = xyz.shape[0]
    target = torch.zeros(n, dtype=torch.bool, device=xyz.device)
    target_size = int(mask_ratio * n)

    if target_size == 0 or n == 0:
        context = ~target
        return context, target

    n_masked = 0
    for _ in range(n_clusters):
        if n_masked >= target_size or n_masked >= n:
            break
        # Uniform-random choice of an unmasked anchor without materializing
        # `remaining` via boolean-mask indexing (`all_indices[~target]`),
        # which calls a synchronizing nonzero() every cluster, every event,
        # every step (up to n_clusters * batch_size times/step -- the
        # dominant remaining cost after the fixes above). Argmax of i.i.d.
        # random keys, masked to -inf at already-target positions, is
        # exactly a uniform-random draw from the unmasked positions.
        keys = torch.rand(n, device=xyz.device).masked_fill(target, float("-inf"))
        anchor = keys.argmax(dim=0, keepdim=True)
        dists = torch.norm(xyz - xyz[anchor], dim=-1)
        order = torch.argsort(dists)

        # `order` ranges over ALL n indices (nearest-to-anchor first), same
        # as the original -- already-masked ones are no-ops that still
        # occupy a loop slot in the reference formulation. `cum` counts how
        # many *new* pulses have been seen by each position in that order;
        # `keep` selects the prefix of new pulses needed to reach
        # target_size, in the same order they'd have been assigned one by
        # one.
        not_yet_masked = ~target[order]
        needed = target_size - n_masked
        cum = torch.cumsum(not_yet_masked.long(), dim=0)
        keep = not_yet_masked & (cum <= needed)

        # `order[keep]` (boolean-mask indexing) would call a synchronizing
        # nonzero() -- scatter `keep` back to original-index positions via
        # integer-index assignment instead (order is a permutation, so this
        # is a plain no-conflict scatter) and OR it in.
        keep_full = torch.zeros(n, dtype=torch.bool, device=xyz.device)
        keep_full[order] = keep
        target |= keep_full

        # Exactly how many new positions `keep` selects: the first
        # min(needed, available) not-yet-masked entries in `order`, where
        # available = n - n_masked (total not-yet-masked count, which is
        # what `cum`'s final value would be). Plain Python ints -- no tensor
        # read needed.
        available = n - n_masked
        n_masked += min(needed, available)

    context = ~target
    return context, target
