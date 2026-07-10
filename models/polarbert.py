import torch
from torch import Tensor, nn


class PolarBERTEncoder(nn.Module):
    """Standard pre-LN transformer encoder, no positional encoding
    (IceCube pulses are an unordered point set). Not used elsewhere in this
    plan (PolarBERT/Model B ablation is deferred) -- kept for parity with the
    spec's file layout and to unblock a future ablation plan."""

    def __init__(self, d: int = 256, n_layers: int = 6, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(6, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        b = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if padding_mask is not None:
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x[:, 0]


class MAEHead(nn.Module):
    """Reconstructs (t, q) of masked DOMs from their node embeddings."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2)
        )

    def forward(self, node_embeddings: Tensor) -> Tensor:
        return self.proj(node_embeddings)
