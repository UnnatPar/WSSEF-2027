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
    """
    n = xyz.shape[0]
    target = torch.zeros(n, dtype=torch.bool, device=xyz.device)
    target_size = int(mask_ratio * n)

    if target_size == 0 or n == 0:
        context = ~target
        return context, target

    all_indices = torch.arange(n, device=xyz.device)
    for _ in range(n_clusters):
        if target.sum().item() >= target_size:
            break
        remaining = all_indices[~target]
        if remaining.numel() == 0:
            break
        anchor = remaining[torch.randint(len(remaining), (1,)).item()]
        dists = torch.norm(xyz - xyz[anchor], dim=-1)
        order = torch.argsort(dists)
        for idx in order.tolist():
            if target.sum().item() >= target_size:
                break
            target[idx] = True

    context = ~target
    return context, target
