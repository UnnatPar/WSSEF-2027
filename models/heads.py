import torch
from torch import Tensor, nn


class DirectionHead(nn.Module):
    """(B, d) -> (B,), (B,): predicted (azimuth, zenith) in radians."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 2),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.mlp(x)
        az = 2 * torch.pi * torch.sigmoid(out[:, 0])
        zen = torch.pi * torch.sigmoid(out[:, 1])
        return az, zen


class ClassificationHead(nn.Module):
    """(B, d) -> (B, 1): logit for track (1) vs cascade (0)."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class MAEHead(nn.Module):
    """Reconstructs (t, q) of masked DOMs from their node embeddings."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2)
        )

    def forward(self, node_embeddings: Tensor) -> Tensor:
        return self.proj(node_embeddings)
