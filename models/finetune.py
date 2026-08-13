import lightning as pl
import torch
import torch.nn.functional as F
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from torch import nn
from torch_geometric.nn import global_mean_pool

from eval.metrics import mean_angular_error, to_cartesian
from models.heads import ClassificationHead, DirectionHead
from models.pet import PETEncoder
from train.optim import split_decay_params, warmup_cosine_lr_lambda


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
        # cfg.d + 1, not cfg.d: an explicit event-level aux_frac (fraction of
        # "auxiliary"/less-certain pulses, see _forward) is concatenated onto
        # g before it reaches any head. Root cause fixed here, confirmed
        # directly on scratch_v4's checkpoint: aux_frac was the single
        # strongest predictor in the whole error breakdown (26 deg spread,
        # 50.6 deg at low aux_frac vs 76.5 deg at high), yet zeroing the raw
        # per-pulse `aux` feature only cost 3.39 deg in ablation -- the model
        # has access to real, highly predictive signal here and mostly isn't
        # using it. Root cause: aux only ever reaches the heads as a raw
        # per-pulse 0/1 flag that has to survive mean/max/add pooling across
        # an event, which is a lossy path for recovering a clean fraction/
        # rate statistic (sum-pooling in particular gives a count, confounded
        # with n_pulses, not a rate). Handing the heads the already-computed
        # ratio directly removes that reconstruction burden.
        self.direction_head = DirectionHead(cfg.d + 1)
        self.classification_head = ClassificationHead(cfg.d + 1)
        self.kappa_head = nn.Linear(cfg.d + 1, 1)
        self.direction_loss_fn = VonMisesFisher3DLoss()

        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _forward(self, batch, batch_size: int | None = None):
        g = self.encoder.encode_event(batch.x, batch.batch, batch_size=batch_size)
        # column 5 = auxiliary flag (see train/dataset.py's FEATURE_COLUMNS +
        # the appended auxiliary column) -- see __init__'s comment for why
        # this is appended explicitly rather than left for the encoder to
        # reconstruct implicitly.
        aux_frac = global_mean_pool(batch.x[:, 5:6], batch.batch, size=batch_size)
        g = torch.cat([g, aux_frac], dim=-1)
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

        # Staged + blended direction loss -- root cause fixed here, confirmed
        # directly on two real runs: raw vMF loss applied end-to-end from a
        # cold (randomly initialized) direction_head/kappa_head collapsed
        # az/zen predictions to an exact constant (mae_finetune_v1, step
        # 74,301: std=0.0000), even with an already-healthy, non-collapsed
        # MAE-pretrained encoder underneath (direction_head/kappa_head are
        # separate, freshly-initialized modules, so encoder quality doesn't
        # help). Mechanism: whenever direction predictions are worse than
        # random (cos(Delta theta) < 0), the vMF term -kappa*cos(Delta theta)
        # is genuinely minimized by shrinking kappa toward zero -- a real,
        # mathematically correct gradient signal -- which drags
        # direction_head's own gradient down with it (vMF's gradient w.r.t.
        # direction scales with kappa), a self-reinforcing trap.
        #
        # A first fix (direction_warmup_steps alone, hard-switching from pure
        # cos-distance to pure vMF at the boundary) was NOT sufficient:
        # confirmed on mae_finetune_v2/mae_probe_v1, cos-distance loss was
        # healthy and stable (~0.93-1.02, matching the ~1.0 random-guess
        # baseline) for the entire warmup window, but train/loss jumped from
        # 1.015 at step 2999 to 2.617 at step 3049 -- immediately upon
        # switching to pure vMF -- and stayed collapsed there indefinitely
        # (probe's whole multi-thousand-step visible history sat flat at
        # ~2.6-2.7 with no recovery). A hard switch just relocates the same
        # cold-start trap to the warmup boundary, since vMF alone can still
        # dominate the gradient once introduced.
        #
        # The actual IceCube Kaggle competition winners' proven fix
        # (arXiv:2310.15674) is not a hard switch but a permanent blend: the
        # 2nd place team used pure angular-distance loss for the first 2-3
        # epochs, then trained with `l = opening_angle + 0.05 * vMF_loss` for
        # the rest of training -- vMF never becomes the dominant term, only a
        # small always-present calibration signal for kappa, so it can never
        # again hijack direction_head's gradient the way a pure vMF loss can.
        # Mirrored here: pure cos-distance for the first
        # direction_warmup_steps, then cos-distance + vmf_blend_weight * vMF
        # (default 0.05, matching the real value) for the remainder --
        # cos-distance stays dominant and well-behaved forever, vMF only
        # contributes gradient to kappa_head, not full control over
        # direction_head.
        try:
            global_step = self.trainer.global_step
        except (RuntimeError, TypeError, AttributeError):
            global_step = None
        warmup_steps = getattr(self.cfg, "direction_warmup_steps", 0)

        cos_sim = (pred_vec * true_vec).sum(-1)
        direction_loss = (1 - cos_sim).mean()

        if global_step is None or global_step >= warmup_steps:
            pred = torch.cat([pred_vec, kappa], dim=-1)
            # graphnet's VonMisesFisher3DLoss.log_cmk mixes its two internal
            # branches (log_cmk_approx, log_cmk_exact) in an index_put --
            # under AMP autocast they land in different dtypes (bf16 vs
            # fp32) and the assignment throws. Verified by direct testing on
            # a real training step, not a fixture.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                vmf_loss = self.direction_loss_fn(pred.float(), true_vec.float())
            vmf_blend_weight = getattr(self.cfg, "vmf_blend_weight", 0.05)
            direction_loss = direction_loss + vmf_blend_weight * vmf_loss
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
            groups = [
                {"params": head_decay, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay},
                {"params": head_no_decay, "lr": self.cfg.lr_heads, "weight_decay": 0.0},
            ]
        else:
            encoder_decay, encoder_no_decay = split_decay_params(self.encoder)
            groups = [
                {"params": encoder_decay, "lr": self.cfg.lr_encoder, "weight_decay": self.cfg.weight_decay},
                {"params": encoder_no_decay, "lr": self.cfg.lr_encoder, "weight_decay": 0.0},
                {"params": head_decay, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay},
                {"params": head_no_decay, "lr": self.cfg.lr_heads, "weight_decay": 0.0},
            ]

        # warmup_cosine_lr_lambda, NOT a bare (unscheduled) AdamW: root cause
        # of a real, confirmed collapse in mae_finetune_v3 (step 11,561),
        # found after the direction-loss staging/blend fix (f1dd6e7) held up
        # (loss curve looked healthy) but direct measurement showed az/zen
        # predictions were still an exact constant. Traced upstream:
        # node_emb (the encoder's own per-pulse output) had degraded from a
        # healthy 0.61 (the pretrain_mae_v6 checkpoint this run started from)
        # to 0.19 after 11,561 unfrozen fine-tuning steps -- the exact same
        # GNN over-smoothing collapse diagnosed and fixed for MAE pretraining
        # in 799354b, recurring here because configure_optimizers used a
        # flat, unscheduled AdamW with zero LR warmup for the now-unfrozen
        # encoder, missing the exact protection pretrain_mae.py already has.
        # Applied to both the frozen and unfrozen cases for consistency --
        # even head-only training benefits from not starting at full LR on
        # step 1 given everything else learned about this architecture's
        # cold-start sensitivity this session.
        optimizer = torch.optim.AdamW(groups)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(1000, total_steps // 20)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine_lr_lambda(self.trainer, total_steps, warmup_steps)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


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
        # pool_proj/pool_norm always excluded here, not just loaded with
        # strict=False -- MAE pretraining calls encoder.forward() directly
        # and never touches encode_event()/pool_proj/pool_norm at all, so
        # any MAE checkpoint's pool_proj/pool_norm weights are untrained
        # garbage regardless of shape. Excluding them outright (rather than
        # relying on strict=False, which only tolerates *missing* keys, not
        # *shape-mismatched* ones) keeps this loader working even after
        # pool_proj's shape changed (3*d -> 2*d, mean-pool branch dropped,
        # see models/pet.py) -- an old MAE checkpoint's pool_proj would
        # otherwise hard-error on a shape mismatch instead of just leaving a
        # key out. Both always start at their fresh (identity-like) init,
        # same as pool_norm already did unconditionally before this fix.
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("encoder.") and not k.startswith("encoder.pool_proj.")
            and not k.startswith("encoder.pool_norm.")
        }
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        assert not unexpected, f"unexpected keys in encoder checkpoint: {unexpected}"
        assert set(missing) <= {
            "pool_proj.weight", "pool_proj.bias", "pool_norm.weight", "pool_norm.bias",
        }, f"unexpected missing keys in encoder checkpoint: {missing}"
        if cfg.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
    return model
