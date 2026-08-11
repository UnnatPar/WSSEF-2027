import torch
from torch import Tensor, nn


class DirectionHead(nn.Module):
    """(B, d) -> (B,), (B,): predicted (azimuth, zenith) in radians."""

    def __init__(self, d: int = 256):
        super().__init__()
        # LayerNorm before each GELU -- root cause fixed here, confirmed
        # directly on a real run (scratch_v3, step 30,922): az/zen
        # predictions collapsed to an exact constant (std=0.0034/0.0000)
        # even though the encoder underneath was confirmed healthy
        # (node_emb std=0.57, not collapsed -- the LR-warmup fix for the
        # encoder in dd5609f held). Traced the collapse to this module
        # itself: tracing g's diversity through each of mlp's layers showed
        # it eroding through the stack (std 0.065 -> 0.027 -> 0.056 ->
        # 0.032) and the final Linear's raw (pre-sigmoid) output deeply
        # saturated (range [-16.3, -4.5], all strongly negative -> sigmoid
        # collapses to ~0 regardless of input). This is the same
        # no-normalization-anywhere-in-a-layer-stack pattern already found
        # and fixed three other times this session (PETEncoder.input_proj,
        # PETBlock, PETEncoder.pool_proj/pool_norm) -- DirectionHead was the
        # last unprotected spot, free to drift into internal saturation over
        # enough training steps with nothing to constrain its intermediate
        # activations' scale.
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(),
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
        # LayerNorm before GELU -- same fix, same reasoning as DirectionHead
        # above (proactive here: this head sees far less real gradient
        # signal than DirectionHead in production, since the real Kaggle
        # data has no `pid` label at all, but there's no reason to leave it
        # with the same unprotected structure that just collapsed).
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(), nn.Linear(d // 2, 1),
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
