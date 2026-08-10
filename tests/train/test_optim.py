import torch
from torch import nn

from train.optim import make_param_groups, split_decay_params


def test_split_decay_params_puts_linear_weight_in_decay():
    linear = nn.Linear(8, 4)
    decay, no_decay = split_decay_params(linear)
    assert any(p is linear.weight for p in decay)
    assert not any(p is linear.weight for p in no_decay)


def test_split_decay_params_puts_linear_bias_in_no_decay():
    linear = nn.Linear(8, 4)
    decay, no_decay = split_decay_params(linear)
    assert any(p is linear.bias for p in no_decay)
    assert not any(p is linear.bias for p in decay)


def test_split_decay_params_puts_layernorm_weight_and_bias_in_no_decay():
    """The actual bug this whole module exists to fix: LayerNorm's weight
    (gain) is a 1D parameter just like a bias, and must NOT be decayed --
    see train/optim.py's module docstring for the real, measured collapse
    this caused when it was."""
    ln = nn.LayerNorm(16)
    decay, no_decay = split_decay_params(ln)
    assert decay == []
    assert any(p is ln.weight for p in no_decay)
    assert any(p is ln.bias for p in no_decay)


def test_split_decay_params_skips_frozen_parameters():
    linear = nn.Linear(8, 4)
    linear.weight.requires_grad_(False)
    decay, no_decay = split_decay_params(linear)
    assert not any(p is linear.weight for p in decay)
    assert not any(p is linear.weight for p in no_decay)


def test_make_param_groups_applies_weight_decay_only_to_decay_group():
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    groups = make_param_groups(model, lr=1e-3, weight_decay=0.05)
    assert len(groups) == 2
    decay_group = next(g for g in groups if g["weight_decay"] == 0.05)
    no_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)
    assert len(decay_group["params"]) > 0
    assert len(no_decay_group["params"]) > 0


def test_layernorm_gain_survives_many_steps_of_pure_weight_decay_with_the_fix():
    """Direct regression test for the real, measured failure: without this
    fix, a LayerNorm's gain decays via pure exponential shrinkage toward
    zero even with zero gradient signal opposing it. Simulates exactly that
    (a LayerNorm whose gain never receives a real gradient, matching the
    production model where weight decay dominated) across a realistic
    number of steps, and checks the gain survives with the fix applied."""
    ln = nn.LayerNorm(16)
    initial_gain = ln.weight.clone()

    groups = make_param_groups(ln, lr=1e-3, weight_decay=0.05)
    optimizer = torch.optim.AdamW(groups)

    for _ in range(2000):
        optimizer.zero_grad()
        # Zero gradient on purpose -- isolates weight decay's own effect,
        # matching the real scenario where gradient signal to the gain was
        # negligible compared to decay.
        ln.weight.grad = torch.zeros_like(ln.weight)
        ln.bias.grad = torch.zeros_like(ln.bias)
        optimizer.step()

    assert torch.allclose(ln.weight, initial_gain), (
        "LayerNorm gain moved under zero gradient -- weight decay is still "
        "being applied to it despite make_param_groups"
    )
