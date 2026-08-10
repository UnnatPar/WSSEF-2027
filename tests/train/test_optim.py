import torch
from torch import nn

from train.optim import make_param_groups, split_decay_params, warmup_cosine_lr_lambda


class _FakeTrainer:
    def __init__(self, global_step=0):
        self.global_step = global_step


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


def test_warmup_cosine_lr_lambda_ramps_linearly_from_zero_during_warmup():
    trainer = _FakeTrainer(global_step=0)
    lr_lambda = warmup_cosine_lr_lambda(trainer, total_steps=1000, warmup_steps=100)
    assert lr_lambda(None) == 0.0
    trainer.global_step = 50
    assert lr_lambda(None) == 0.5
    trainer.global_step = 99
    assert abs(lr_lambda(None) - 0.99) < 1e-9


def test_warmup_cosine_lr_lambda_reaches_full_lr_at_end_of_warmup():
    trainer = _FakeTrainer(global_step=100)
    lr_lambda = warmup_cosine_lr_lambda(trainer, total_steps=1000, warmup_steps=100)
    assert abs(lr_lambda(None) - 1.0) < 1e-9


def test_warmup_cosine_lr_lambda_decays_to_zero_at_total_steps():
    trainer = _FakeTrainer(global_step=1000)
    lr_lambda = warmup_cosine_lr_lambda(trainer, total_steps=1000, warmup_steps=100)
    assert abs(lr_lambda(None)) < 1e-9


def test_warmup_cosine_lr_lambda_never_exceeds_one():
    trainer = _FakeTrainer(global_step=0)
    lr_lambda = warmup_cosine_lr_lambda(trainer, total_steps=1000, warmup_steps=100)
    for step in range(0, 1001, 17):
        trainer.global_step = step
        assert 0.0 <= lr_lambda(None) <= 1.0 + 1e-9


def test_warmup_cosine_lr_lambda_handles_zero_warmup_steps_without_dividing_by_zero():
    trainer = _FakeTrainer(global_step=1)
    lr_lambda = warmup_cosine_lr_lambda(trainer, total_steps=1000, warmup_steps=0)
    assert abs(lr_lambda(None) - 1.0) < 1e-9  # warmup_steps clamped to 1, so step 1 is already past it
