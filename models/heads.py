import torch
from torch import Tensor, nn


def _direction_branch(d: int) -> nn.Sequential:
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
    return nn.Sequential(
        nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
        nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(),
        nn.Linear(d // 2, 1),
    )


class DirectionHead(nn.Module):
    """(B, d) -> (B,), (B,): predicted (azimuth, zenith) in radians.

    split=True: separate az/zen branches, no shared parameters. Directly
    confirmed by an A/B test on scratch_v4's checkpoint (frozen, already-
    trained encoder): a matched-capacity zenith-only head trained on the
    same frozen encoder embeddings reached 27.50 deg MAE vs the joint head's
    46.41 deg (+18.91 deg of pure interference cost; azimuth barely affected,
    +1.49 deg). A follow-up gradient-attribution test ruled out the shared
    trunk embedding `g` as the site of conflict (grad_az/grad_zen showed
    near-1.0 magnitude ratio and ~0 cosine similarity) -- pointing the
    mechanism specifically at DirectionHead's own shared parameters.

    split=False (the default): the original joint head, one shared trunk
    producing both outputs. Real regression found applying split=True
    unconditionally: scratch_v6/v7 (encoder AND heads both cold, fully
    random init) collapsed -- direct checkpoint inspection at step 9,551,
    still inside pure-cos-distance warmup (kappa confirmed untouched,
    std=0.1675, ruling out any vMF/kappa involvement), showed BOTH az and
    zen branches already collapsing toward near-constant, deeply-saturated
    pre-sigmoid outputs (az std=0.0114, zen std=0.0349) -- for comparison,
    scratch_v4's joint head reached az std=2.38 by step 4,017, an order of
    magnitude more diverse, dramatically earlier. The A/B evidence above was
    gathered with an already-good, frozen `g`; two independently-initialized
    branches coordinating on one joint 3D cos-similarity target through a
    fully-random `g` (scratch's actual cold-start regime) appears to be a
    genuinely harder problem the split evidence never covered. split=True is
    scoped to configs where the encoder starts from a real pretrained
    checkpoint (mae_finetune/jepa_finetune, closer to what was actually
    tested) until re-validated for the from-scratch regime.
    """

    def __init__(self, d: int = 256, split: bool = False):
        super().__init__()
        self.split = split
        if split:
            self.az_branch = _direction_branch(d)
            self.zen_branch = _direction_branch(d)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
                nn.Linear(d, d // 2), nn.LayerNorm(d // 2), nn.GELU(),
                nn.Linear(d // 2, 2),
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.split:
            az = 2 * torch.pi * torch.sigmoid(self.az_branch(x).squeeze(-1))
            zen = torch.pi * torch.sigmoid(self.zen_branch(x).squeeze(-1))
            return az, zen
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
