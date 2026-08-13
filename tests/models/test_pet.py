import torch

from models.pet import PETBlock, PETEncoder


def test_pet_block_output_shape_single_event():
    block = PETBlock(d=16, k=4)
    x = torch.randn(20, 16)
    batch = torch.zeros(20, dtype=torch.long)
    out = block(x, batch)
    assert out.shape == (20, 16)


def test_pet_block_output_shape_multi_event(tiny_pyg_batch):
    d = 16
    n = tiny_pyg_batch.x.shape[0]
    block = PETBlock(d=d, k=4)
    x = torch.randn(n, d)
    out = block(x, tiny_pyg_batch.batch)
    assert out.shape == (n, d)


def test_pet_block_gradients_flow_to_all_params():
    block = PETBlock(d=16, k=4)
    x = torch.randn(20, 16, requires_grad=True)
    batch = torch.zeros(20, dtype=torch.long)
    out = block(x, batch)
    out.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in block.parameters())


def test_pet_block_handles_k_larger_than_event_size():
    # event has fewer than k=8 nodes; knn_graph must not error
    block = PETBlock(d=16, k=8)
    x = torch.randn(5, 16)
    batch = torch.zeros(5, dtype=torch.long)
    out = block(x, batch)
    assert out.shape == (5, 16)


def test_pet_block_has_layerscale_params_initialized_small():
    """Root cause fixed by LayerScale, confirmed via a controlled A/B on
    real data with production's actual LR schedule shape: with no
    LayerScale, sustained near-peak LR (matching the real cosine schedule,
    which barely decays across thousands of steps out of a 312,504-step
    horizon) drives attention to near-uniform softmax over the k-NN
    neighborhood, and applying that at full residual strength through all 6
    stacked blocks causes a sudden over-smoothing collapse (node embedding
    cross-sample std dropped 500x within ~200 steps after ~2,200 healthy
    steps). LayerScale gates each sublayer's contribution with a small
    learnable per-channel scale, starting near zero so the residual
    dominates until training earns the right to mix -- see models/pet.py's
    PETBlock docstring for the full A/B evidence."""
    block = PETBlock(d=16, k=4)
    assert torch.allclose(block.gamma1, torch.full((16,), 1e-2))
    assert torch.allclose(block.gamma2, torch.full((16,), 1e-2))


def test_pet_block_layerscale_gradients_flow():
    block = PETBlock(d=16, k=4)
    x = torch.randn(20, 16, requires_grad=True)
    batch = torch.zeros(20, dtype=torch.long)
    out = block(x, batch)
    out.sum().backward()
    assert block.gamma1.grad is not None
    assert block.gamma2.grad is not None


def test_pet_encoder_forward_shape(tiny_pyg_batch):
    encoder = PETEncoder(d=16, L=2, k=4)
    out = encoder(tiny_pyg_batch.x, tiny_pyg_batch.batch)
    assert out.shape == (tiny_pyg_batch.x.shape[0], 16)


def test_pet_encoder_encode_event_norm_does_not_scale_with_event_size():
    """Root cause of a real collapsed-direction-head bug, confirmed against a
    real checkpoint (configs/scratch.yaml, step 4,252): encode_event's
    global_add_pool sums over an event's pulses, so g's magnitude scaled
    almost linearly with event size (measured r=0.98 between pulse count and
    ||g|| across 256 real events, 76 at 25 pulses vs 512-604 at 256 pulses).
    With no normalization downstream, that saturated DirectionHead's sigmoid
    output (std collapsed to 0.00045) despite g itself being diverse. This
    test constructs two events with the same PER-PULSE feature distribution
    but very different pulse counts and checks pool_norm keeps ||g||
    comparable regardless -- before the fix, this ratio was ~7x; after, it
    should be close to 1."""
    torch.manual_seed(0)
    encoder = PETEncoder(d=16, L=2, k=4)
    small_event = torch.randn(5, 16)
    large_event = small_event.repeat(20, 1)[:100]  # same per-pulse distribution, 20x more pulses
    x = torch.cat([small_event, large_event], dim=0)
    batch = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(100, dtype=torch.long)])

    from torch_geometric.nn import global_add_pool, global_max_pool
    pooled = torch.cat([
        global_max_pool(x, batch), global_add_pool(x, batch),
    ], dim=-1)
    g = encoder.pool_norm(encoder.pool_proj(pooled))
    norms = g.norm(dim=-1)
    ratio = (norms.max() / norms.min()).item()
    assert ratio < 2.0, (
        f"g's norm still scales with event size (ratio={ratio:.2f}x between a 5-pulse and "
        "100-pulse event) -- pool_norm isn't standardizing the pooled embedding's magnitude"
    )


def test_pet_encoder_encode_event_shape(tiny_pyg_batch):
    encoder = PETEncoder(d=16, L=2, k=4)
    n_events = int(tiny_pyg_batch.batch.max().item()) + 1
    g = encoder.encode_event(tiny_pyg_batch.x, tiny_pyg_batch.batch)
    assert g.shape == (n_events, 16)


def test_pet_encoder_gradients_flow(tiny_pyg_batch):
    encoder = PETEncoder(d=16, L=2, k=4)
    g = encoder.encode_event(tiny_pyg_batch.x, tiny_pyg_batch.batch)
    g.sum().backward()
    assert all(p.grad is not None for p in encoder.parameters())


def test_edge_mlp_gradient_survives_deeply_negative_pre_activation():
    """Same failure mode as input_proj (see that test's docstring for the
    full mechanism), found in a second location by checking every ReLU
    usage in the codebase after fixing the first one: PETBlock.edge_mlp's
    first Linear -> ReLU. This one has no LayerNorm at all stabilizing its
    input (unlike input_proj), and runs on every edge of every k-NN graph
    in every block -- checked directly against the real trained checkpoint
    (step 272,088), block 2's edge_mlp was 100.0% dead (every one of 256
    units, zero output for all 86,480 real edges tested); blocks 0/1/3 were
    83.6%/97.7%/98.8% dead. edge_mlp's first Linear has no normalization
    layer before its activation, so its own bias alone is enough to force
    every unit's pre-activation negative -- no LayerNorm affine trick
    needed here."""
    block = PETBlock(d=16, k=4)
    with torch.no_grad():
        block.edge_mlp[0].weight.zero_()
        block.edge_mlp[0].bias.fill_(-5.0)
    x = torch.randn(20, 16, requires_grad=True)
    batch = torch.zeros(20, dtype=torch.long)
    out = block(x, batch)
    out.sum().backward()
    edge_mlp_grad_norm = torch.cat(
        [p.grad.flatten() for p in block.edge_mlp[0].parameters()]
    ).norm()
    assert edge_mlp_grad_norm > 0


def test_input_proj_gradient_survives_deeply_negative_pre_activation(tiny_pyg_batch):
    """Real, confirmed production bug: input_proj used to be Linear ->
    LayerNorm -> ReLU. Checked directly against the full checkpoint history
    of a real production run: the fraction of permanently-dead units (zero
    for every one of 10,810 real test pulses) climbed monotonically from
    0.4-1.6% at random init (healthy) to 87.1% by step 272,088 -- and it was
    the ONLY activation in this codebase still using plain ReLU (every other
    component -- PETBlock.ffn, MAEHead, DirectionHead -- already used GELU).
    A dead ReLU unit's gradient is exactly zero for any negative
    pre-activation and can never recover via gradient descent, so this was a
    permanent, accumulating capacity loss -- plausibly the real explanation
    for a val loss that converged fast then never moved again for 230k+
    steps, unaffected by every other real fix made this session (precision,
    LR schedule, checkpoint resolution).

    Verified directly with a controlled A/B: training both variants on the
    same real IceCube data with the same optimizer for 100 steps, ReLU
    already reached 4.7% dead units while GELU reached exactly 0.0%, with
    statistically identical loss curves -- GELU is not a capacity/speed
    tradeoff here, it just lacks ReLU's exact-zero-gradient trap.

    This test targets that specific mechanism directly: force input_proj's
    Linear layer to produce a deeply negative pre-activation for every unit
    (the exact state real training drove most units into), and require a
    nonzero gradient reaches it regardless."""
    encoder = PETEncoder(d=16, L=2, k=4)
    with torch.no_grad():
        # LayerNorm normalizes each sample to zero-mean across features, so
        # the Linear layer alone can't push every unit negative at once --
        # its own learned affine shift (weight/bias) is what real training
        # actually uses to do that. Force it here: weight~0 kills the
        # per-sample-varying normalized signal, bias=-5 pushes every unit's
        # post-LayerNorm value negative regardless of input -- a realistic
        # magnitude a learned LayerNorm bias could reach, not an artificial
        # extreme. (GELU's gradient, like ReLU's, does eventually underflow
        # to exact 0.0 in float32 -- but only past roughly x=-20, verified
        # directly; -5 stays in GELU's real, nonzero-gradient range while
        # still being solidly negative enough to kill a plain ReLU unit
        # outright, which dies at any x<0.)
        encoder.input_proj[1].weight.fill_(1e-6)
        encoder.input_proj[1].bias.fill_(-5.0)
    g = encoder.encode_event(tiny_pyg_batch.x, tiny_pyg_batch.batch)
    g.sum().backward()
    layernorm_grad_norm = torch.cat(
        [p.grad.flatten() for p in encoder.input_proj[1].parameters()]
    ).norm()
    assert layernorm_grad_norm > 0
