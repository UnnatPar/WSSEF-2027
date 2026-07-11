import torch


def to_cartesian(az: torch.Tensor, zen: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        torch.sin(zen) * torch.cos(az),
        torch.sin(zen) * torch.sin(az),
        torch.cos(zen),
    ], dim=-1)


def mean_angular_error(pred_az, pred_zen, true_az, true_zen) -> float:
    """Great-circle distance in degrees. Identical to the Kaggle metric."""
    dot = (to_cartesian(pred_az, pred_zen) * to_cartesian(true_az, true_zen)).sum(-1)
    dot = dot.clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(dot).mean().item() * 180 / torch.pi


def angular_error_by_energy(pred_az, pred_zen, true_az, true_zen, log_energies) -> dict:
    """Bin events into 10 log10-energy bins across [2, 7].
    Returns {bin_center: mean_angular_error}. Matches PolarBERT paper Fig 2 binning.
    """
    bins = torch.linspace(2, 7, 11)
    result = {}
    for i in range(10):
        mask = (log_energies >= bins[i]) & (log_energies < bins[i + 1])
        if mask.sum() > 0:
            result[float((bins[i] + bins[i + 1]) / 2)] = mean_angular_error(
                pred_az[mask], pred_zen[mask], true_az[mask], true_zen[mask]
            )
    return result
