import torch
from torch import Tensor, nn
from torch_cluster import knn_graph
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_scatter import scatter_softmax, scatter_sum


class PETBlock(nn.Module):
    """Edge-conditioned local attention over a dynamic k-NN graph.

    PyG's DynamicEdgeConv only returns aggregated node output, not the
    per-edge embeddings this attention step needs, so the edge MLP is
    applied directly per-edge on a knn graph built with torch_cluster.
    """

    def __init__(self, d: int, k: int = 8):
        super().__init__()
        self.d = d
        self.k = k
        # GELU, not ReLU -- same fix, same reasoning as PETEncoder.input_proj
        # (see that comment for the full mechanism and A/B evidence), but
        # this collapse was measured even more severe here: this MLP has no
        # LayerNorm at all stabilizing its input (unlike input_proj), and
        # runs on every edge of every k-NN graph in every block. Checked
        # directly against the real trained checkpoint (step 272,088):
        # block 2's edge_mlp was 100.0% dead (every one of 256 units, zero
        # output for all 86,480 real edges tested); blocks 0/1/3 were
        # 83.6%/97.7%/98.8% dead.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )

    def forward(self, x: Tensor, batch: Tensor, batch_size: int | None = None) -> Tensor:
        n = x.shape[0]
        k = min(self.k, n - 1) if n > 1 else 1
        # Passing batch_size explicitly skips torch_cluster.knn's internal
        # `int(batch.max()) + 1`, a synchronizing GPU->CPU round trip it
        # otherwise repeats on every one of the L blocks' calls (verified via
        # torch.profiler on a real L4: aten::_local_scalar_dense + a chunk of
        # cudaStreamSynchronize traced directly to this line -- see ledger
        # 2026-07-28). The caller already knows batch_size from its own batch
        # vector at zero extra sync cost.
        edge_index = knn_graph(
            x, k=k, batch=batch, flow="source_to_target", batch_size=batch_size,
        )
        src, dst = edge_index[0], edge_index[1]

        e_ij = self.edge_mlp(torch.cat([x[dst], x[src] - x[dst]], dim=-1))

        Q = self.W_Q(x)
        K = self.W_K(x[src] + e_ij)
        V = self.W_V(x[src] + e_ij)

        scores = (Q[dst] * K).sum(-1) / (self.d ** 0.5)
        alpha = scatter_softmax(scores, dst, dim=0, dim_size=n)
        x_attn = scatter_sum(alpha.unsqueeze(-1) * V, dst, dim=0, dim_size=n)

        x = self.norm1(x + x_attn)
        x = self.norm2(x + self.ffn(x))
        return x


class PETEncoder(nn.Module):
    def __init__(self, d: int = 256, L: int = 6, k: int = 8):
        super().__init__()
        # GELU, not ReLU: verified directly against a real trained checkpoint
        # (step 272,088) that this exact LayerNorm -> ReLU combination
        # collapses catastrophically over the course of real training --
        # 87.1% of the 256 embedding dims were permanently dead (zero
        # pre-activation-negative for every one of 10,810 real test pulses),
        # up from 0.4-1.6% at random init (confirmed healthy) and climbing
        # monotonically throughout training (1.2% at step 12k -> 9.8% at 32k
        # -> 59.4% at 104k -> 87.1% at 272k, checked across the full
        # checkpoint history). A dead ReLU unit's gradient is exactly zero
        # for any negative pre-activation and can never recover, so this is
        # a permanent, accumulating capacity loss -- plausibly explaining
        # the exact symptom chased all session: fast initial convergence
        # (while the layer was still healthy) followed by a hard, unmovable
        # plateau (once most of the embedding died), unaffected by every
        # other real fix made (precision, LR schedule, checkpoint
        # resolution) because none of them address why units die in the
        # first place. Every other activation in this codebase already uses
        # GELU (PETBlock.ffn, MAEHead, DirectionHead) -- input_proj's plain
        # ReLU was the sole outlier. Verified directly with a controlled A/B:
        # training both variants on the same real data/optimizer for 100
        # steps, ReLU reached 4.7% dead units while GELU reached exactly
        # 0.0%, with statistically identical loss curves (~0.012 either
        # way) -- GELU is not a capacity/speed tradeoff here, it just
        # doesn't have ReLU's exact-zero-gradient trap.
        self.input_proj = nn.Sequential(
            nn.Linear(6, d), nn.LayerNorm(d), nn.GELU()
        )
        self.blocks = nn.ModuleList([PETBlock(d, k) for _ in range(L)])
        self.pool_proj = nn.Linear(3 * d, d)  # 3-way: mean + max + sum

    def forward(self, x: Tensor, batch: Tensor, batch_size: int | None = None) -> Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, batch, batch_size=batch_size)
        return x

    def encode_event(self, x: Tensor, batch: Tensor, batch_size: int | None = None) -> Tensor:
        x = self.forward(x, batch, batch_size=batch_size)
        g = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch),
        ], dim=-1)
        return self.pool_proj(g)
