import lightning as pl
import torch
import torch.nn.functional as F
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from torch import nn

from eval.metrics import mean_angular_error, to_cartesian
from models.heads import ClassificationHead, DirectionHead
from models.pet import PETEncoder


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
        self.kappa_head = nn.Sequential(nn.Linear(cfg.d, 1), nn.Softplus())
        self.direction_loss_fn = VonMisesFisher3DLoss()

        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _forward(self, batch):
        g = self.encoder.encode_event(batch.x, batch.batch)
        az, zen = self.direction_head(g)
        kappa = self.kappa_head(g)
        return g, az, zen, kappa

    def _compute_loss(self, batch, g, az, zen, kappa):
        pred_vec = to_cartesian(az, zen)
        pred = torch.cat([pred_vec, kappa], dim=-1)
        true_vec = to_cartesian(batch.azimuth, batch.zenith)

        # graphnet's VonMisesFisher3DLoss.log_cmk mixes its two internal branches
        # (log_cmk_approx, log_cmk_exact) in an index_put -- under AMP autocast
        # they land in different dtypes (bf16 vs fp32) and the assignment throws.
        # Verified by direct testing on a real training step, not a fixture.
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
        g, az, zen, kappa = self._forward(batch)
        loss = self._compute_loss(batch, g, az, zen, kappa)
        n_events = int(batch.batch.max().item()) + 1
        self.log("train/loss", loss, batch_size=n_events)
        return loss

    def validation_step(self, batch, batch_idx):
        g, az, zen, kappa = self._forward(batch)
        loss = self._compute_loss(batch, g, az, zen, kappa)
        angular_error = mean_angular_error(az, zen, batch.azimuth, batch.zenith)
        n_events = int(batch.batch.max().item()) + 1
        self.log("val/loss", loss, batch_size=n_events)
        self.log("val/angular_error", angular_error, batch_size=n_events)
        return loss

    def configure_optimizers(self):
        head_params = (
            list(self.direction_head.parameters())
            + list(self.classification_head.parameters())
            + list(self.kappa_head.parameters())
        )
        if self.cfg.freeze_encoder:
            return torch.optim.AdamW(
                head_params, lr=self.cfg.lr_heads, weight_decay=self.cfg.weight_decay,
            )
        return torch.optim.AdamW([
            {"params": self.encoder.parameters(), "lr": self.cfg.lr_encoder},
            {"params": head_params, "lr": self.cfg.lr_heads},
        ], weight_decay=self.cfg.weight_decay)


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
        model.encoder.load_state_dict(encoder_state)
        if cfg.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
    return model
