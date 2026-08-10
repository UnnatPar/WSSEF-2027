import math


def warmup_cosine_lr_lambda(trainer, total_steps, warmup_steps):
    """Linear LR warmup into a cosine decay, reading trainer.global_step
    directly rather than the step argument LambdaLR passes in -- same
    resume-safety reasoning as the plain cosine schedules in
    pretrain_mae.py/jepa.py (a scheduler's own internal step counter does
    not reliably track true cumulative progress across checkpoint resumes).

    Root cause this fixes, confirmed by a controlled A/B on real data: with
    no warmup (lr jumps to its full peak value on step 1), PETEncoder's node
    embedding cross-sample std collapsed from 0.68 (healthy, at random
    init) to 0.004 within 20 real AdamW steps and never recovered (0.0017
    by step 300) -- a full representation collapse, independent of and
    prior to the separate LayerNorm-gain-decay collapse (see
    split_decay_params above). With a 100-step linear warmup under
    otherwise identical conditions (same seed, data, architecture, step
    count), std stayed 60-100x higher throughout (0.45 at step 20, ~0.13-0.17
    by step 300) and reached a genuinely lower loss (~0.007-0.009 vs
    ~0.010-0.015 without warmup). Mechanism: AdamW's bias-corrected
    second-moment estimate is enormous in the first few steps (it divides by
    1 - beta2^step, ~1000x amplification at step 1 for beta2=0.999), so a
    full-strength first step on a deep (L=6) k-NN attention stack is large
    enough to immediately push the whole encoder toward the degenerate
    "predict the batch mean" fixed point -- a well-documented failure mode
    for deep transformer/GNN training generally, which is why warmup is
    standard practice (BERT, ViT, etc.) and was the one thing missing here.
    """
    warmup_steps = max(warmup_steps, 1)

    def lr_lambda(_):
        step = trainer.global_step
        if step < warmup_steps:
            return step / warmup_steps
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return lr_lambda


def split_decay_params(module):
    """Splits a module's trainable parameters into (decay, no_decay) groups
    for AdamW, following standard transformer training practice (GPT-2/3,
    BERT, ViT reference implementations all do this).

    Root cause this fixes, confirmed by direct measurement against a real
    trained checkpoint: applying weight_decay to LayerNorm's gain (weight)
    parameter, with no counteracting gradient signal, causes it to decay via
    pure exponential shrinkage -- (1 - lr*weight_decay) per step -- toward
    zero. At step 39,064 with lr~0.001, weight_decay=0.05, every LayerNorm's
    gain in the real checkpoint had decayed to ~0.144, matching
    exp(-lr*weight_decay*steps) almost exactly. As a LayerNorm's gain
    approaches zero, LayerNorm(x) approaches its (comparatively tiny but
    nonzero) bias regardless of x -- chained across 6 encoder blocks, this
    collapses the whole network toward an input-independent, near-constant
    output. Measured directly: node embeddings for 10,810 real, physically
    diverse pulses had collapsed to near-zero cross-sample variance
    (std~0.0002), for masked AND unmasked nodes alike, and even with no
    masking applied at all -- ruling out the masking mechanism itself and
    pointing squarely at this.

    A parameter is "no_decay" if it's 1-dimensional (biases and LayerNorm's
    weight/bias are both 1D vectors, matching this shape regardless of
    module type -- this is why dim() < 2, not an isinstance/module-name
    check, is the standard way to do this split).
    """
    decay, no_decay = [], []
    for param in module.parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def make_param_groups(module, lr, weight_decay):
    """Convenience wrapper: builds AdamW-ready param groups with weight_decay
    applied only to the decay group, no_decay group at weight_decay=0."""
    decay, no_decay = split_decay_params(module)
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]
