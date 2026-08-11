import lightning as pl
import torch
import torch.nn.functional as F
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from torch import nn

from eval.metrics import mean_angular_error, to_cartesian
from models.heads import ClassificationHead, DirectionHead
from models.pet import PETEncoder
from train.optim import split_decay_params


class SupervisedFineTune(pl.LightningModule):
    """Encoder-agnostic supervised training: PETEncoder + DirectionHead +
    ClassificationHead. Serves all 3 experiments (probe / finetune / from
    scratch) -- they differ only in encoder init/freeze state and optimizer
    param groups, never in this training_step.

    VonMisesFisher3DLoss expects a [N,4] prediction (3D direction + a kappa
    concentration parameter) -- verified by direct testing, not the plain
    3-vector the spec's prose implies. DirectionHead's spec-defined interface
    only outputs (az, zen), so a small separate kappa_head supplies the 4th
    column rather than changing DirectionHead's interface.

    The real Kaggle competition data has no track/cascade ("pid") label at
    all -- verified against the competition's actual documented schema --
    so classification loss is only computed when the batch happens to carry
    a `pid` attribute.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = PETEncoder(cfg.d, cfg.L, cfg.k)
        self.direction_head = DirectionHead(cfg.d)
        self.classification_head = ClassificationHead(cfg.d)
        self.kappa_head = nn.Linear(cfg.d, 1)
        self.direction_loss_fn = VonMisesFisher3DLoss()

        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _forward(self, batch, batch_size: int | None = None):
        g = self.encoder.encode_event(batch.x, batch.batch, batch_size=batch_size)
        az, zen = self.direction_head(g)
        # Softplus(x) underflows to exactly 0.0 for a sufficiently negative
        # pre-activation -- easier to hit than it sounds under precision=
        # "16-mixed" AMP, since fp16's subnormal floor (~6e-8) is much higher
        # than fp32's (~1e-38). At kappa=0.0 exactly, graphnet's LogCMK.forward
        # computes log(kappa) - log(iv(0.5, kappa)) = -inf - (-inf) = NaN in
        # floating point, even though the mathematical limit is finite.
        # Verified directly: this NaN'd ~80-100% of steps in a real production
        # run.
        #
        # A tiny epsilon (1e-4) stopped the NaN but not a second, worse
        # failure: verified by direct gradient measurement, d(loss)/d(kappa)
        # is POSITIVE whenever direction predictions are still wrong (as they
        # are at the start of training), so gradient descent pushes kappa
        # toward zero from step one. Once kappa is near zero, the gradient
        # reaching direction_head collapses ~10,000x (measured: 0.131887 at
        # kappa=1.0 vs 0.000013 at kappa=1e-4) -- a self-reinforcing collapse
        # that pinned real production runs at a constant loss for thousands
        # of steps with zero learning.
        #
        # Flooring the *value* at 1.0 stops that, but doesn't stop a third
        # failure: plain Softplus's own gradient (sigmoid of the
        # pre-activation) also underflows to exactly 0.0 once the optimizer
        # -- which still wants to push kappa down, floor or not -- drives the
        # pre-activation far enough negative. Verified by direct step-by-step
        # tracing on a real batch: kappa_grad_norm hits exactly 0.0 by step 20
        # and never recovers, permanently dead-neuroning kappa_head at the
        # floor value regardless of how training proceeds afterward. The leak
        # term below (a LeakyReLU-style fix for the same dying-neuron
        # mechanism) keeps a tiny but nonzero gradient flowing no matter how
        # negative the pre-activation gets, while being numerically identical
        # to plain Softplus everywhere in the normal operating range
        # (verified: diff <= 0.005 for pre-activations in [-5, 5], exactly 0
        # for pre-activations >= 0).
        kappa_raw = self.kappa_head(g)
        kappa = F.softplus(kappa_raw) + 1e-3 * F.relu(-kappa_raw) + 1.0
        return g, az, zen, kappa

    def _compute_loss(self, batch, g, az, zen, kappa):
        pred_vec = to_cartesian(az, zen)
        true_vec = to_cartesian(batch.azimuth, batch.zenith)

        # Staged direction loss -- root cause fixed here, confirmed directly
        # on a real run (mae_finetune_v1, step 74,301): raw vMF loss applied
        # end-to-end from a cold (randomly initialized) direction_head/
        # kappa_head collapsed az/zen predictions to an exact constant
        # (std=0.0000), even though the encoder underneath was already
        # healthy and non-collapsed (a warm-started, real MAE-pretrained
        # encoder didn't help -- direction_head/kappa_head are separate,
        # freshly-initialized modules on top of it, so they hit the same
        # cold-start dynamics regardless). Mechanism: whenever a fresh
        # direction_head's predictions are worse than random
        # (cos(Delta theta) < 0, near-universal at cold start), the vMF loss
        # term -kappa*cos(Delta theta) is genuinely minimized by shrinking
        # kappa toward zero -- a real, mathematically correct gradient
        # signal, not a bug -- which drags direction_head's own gradient
        # down with it (the vMF loss's gradient w.r.t. direction scales with
        # kappa), creating a self-reinforcing trap: direction can't improve
        # without gradient, kappa won't rise without better direction. The
        # actual IceCube Kaggle competition winners hit and solved this
        # exact problem (arXiv:2310.15674): the 2nd place team used a plain
        # angular-distance loss for the first 2-3 epochs before blending in
        # vMF; the 3rd place team froze the backbone and trained a
        # classification-based direction estimate before ever training a
        # vMF-loss regression head. direction_warmup_steps mirrors this:
        # cos-distance loss (no kappa at all, cannot collapse this way) for
        # the first N steps, giving direction_head a chance to learn a
        # reasonable estimate before kappa's dynamics are introduced.
        try:
            global_step = self.trainer.global_step
        except (RuntimeError, TypeError, AttributeError):
            global_step = None
        warmup_steps = getattr(self.cfg, "direction_warmup_steps", 0)

        if global_step is not None and global_step < warmup_steps:
            cos_sim = (pred_vec * true_vec).sum(-1)
            direction_loss = (1 - cos_sim).mean()
        else:
            pred = torch.cat([pred_vec, kappa], dim=-1)
            # graphnet's VonMisesFisher3DLoss.log_cmk mixes its two internal
            # branches (log_cmk_approx, log_cmk_exact) in an index_put --
            # under AMP autocast they land in different dtypes (bf16 vs
            # fp32) and the assignment throws. Verified by direct testing on
            # a real training step, not a fixture.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                direction_loss = self.direction_loss_fn(pred.float(), true_vec.float())
        loss = self.cfg.lambda_direction * direction_loss

        if hasattr(batch, "pid"):
            cls_logits = self.classification_head(g)
            classification_loss = F.binary_cross_entropy_with_logits(
                cls_logits, batch.pid.float().view(-1, 1)
            )
            loss = loss + self.cfg.lambda_classification * classification_loss
        return loss

    def training_step(self, batch, batch_idx):
        n_events = int(batch.batch.max().item()) + 1
        g, az, zen, kappa = self._forward(batch, batch_size=n_events)
        loss = self._compute_loss(batch, g, az, zen, kappa)
        self.log("train/loss", loss, batch_size=n_events)
        return loss

    def validation_step(self, batch, batch_idx):
        n_events = int(batch.batch.max().item()) + 1
        g, az, zen, kappa = self._forward(batch, batch_size=n_events)
        loss = self._compute_loss(batch, g, az, zen, kappa)
        angular_error = mean_angular_error(az, zen, batch.azimuth, batch.zenith)
        self.log("val/loss", loss, batch_size=n_events)
        self.log("val/angular_error", angular_error, batch_size=n_events)
        return loss

    def configure_optimizers(self):
        # split_decay_params, NOT raw .parameters() lists: PETEncoder was
        # confirmed to suffer a severe representation collapse from
        # weight_decay being applied to LayerNorm's gain with no opposing
        # gradient -- see train/pretrain_mae.py's configure_optimizers and
        # train/optim.py for the full, precisely confirmed story (gain decay
        # matched exp(-lr*weight_decay*steps) almost exactly in a real
        # checkpoint). The heads (plain Linear/GELU stacks) don't have
        # LayerNorms, but excluding their biases from decay too matches
        # standard practice and costs nothing.
        head_modules = [self.direction_head, self.classification_head, self.kappa_head]
        head_decay, head_no_decay = [], []
        for m in head_modules:
            d, nd = split_decay_params(m)
            head_decay += d
            head_no_decay += nd

        if self.cfg.freeze_encoder:
            return torch.optim.AdamW([
                {"params": head_decay, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay},
                {"params": head_no_decay, "lr": self.cfg.lr_heads, "weight_decay": 0.0},
            ])

        encoder_decay, encoder_no_decay = split_decay_params(self.encoder)
        return torch.optim.AdamW([
            {"params": encoder_decay, "lr": self.cfg.lr_encoder, "weight_decay": self.cfg.weight_decay},
            {"params": encoder_no_decay, "lr": self.cfg.lr_encoder, "weight_decay": 0.0},
            {"params": head_decay, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay},
            {"params": head_no_decay, "lr": self.cfg.lr_heads, "weight_decay": 0.0},
        ])


def load_full_checkpoint(cfg, checkpoint_path: str) -> SupervisedFineTune:
    """Loads a complete, already-trained probe/finetune checkpoint (encoder
    AND heads) for evaluation. Unlike build_supervised_model, which only
    transplants encoder weights onto freshly-initialized heads to START a
    new probe/finetune run, this restores everything -- used by
    eval/run_report.py to actually score a finished model."""
    model = SupervisedFineTune(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_supervised_model(cfg, checkpoint_path: str | None) -> SupervisedFineTune:
    model = SupervisedFineTune(cfg)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("encoder.")
        }
        # strict=False: encoder.pool_norm didn't exist when earlier MAE
        # pretrain checkpoints (e.g. pretrain_mae_v6) were saved -- MAE
        # pretraining calls encoder.forward() directly and never touches
        # encode_event()/pool_proj/pool_norm at all, so those checkpoints
        # correctly have no gradient-trained values for pool_norm to give
        # here. Missing keys just mean pool_norm starts at its fresh
        # (identity-like) init, same as pool_proj already silently does for
        # every MAE checkpoint regardless of this fix. See models/pet.py's
        # PETEncoder.pool_norm docstring for why it was added.
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        assert not unexpected, f"unexpected keys in encoder checkpoint: {unexpected}"
        assert set(missing) <= {"pool_norm.weight", "pool_norm.bias"}, (
            f"unexpected missing keys in encoder checkpoint: {missing}"
        )
        if cfg.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
    return model
