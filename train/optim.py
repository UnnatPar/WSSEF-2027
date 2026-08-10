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
