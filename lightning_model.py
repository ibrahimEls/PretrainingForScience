import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup
from pytorch_lightning import LightningModule
from pytorch_optimizer import Lion

from network import PET2
from utils import (
    CLIPLoss,
    get_param_groups,
)


class PETLightning(LightningModule):
    def __init__(
        self,
        # model
        input_dim: int = 4,
        num_classes: int = 2,
        hidden_size: int = 64,
        num_transformers: int = 6,
        num_heads: int = 8,
        attn_drop: float = 0.1,
        mlp_drop: float = 0.1,
        mlp_ratio: int = 2,
        feature_drop: float = 0.0,
        num_tokens: int = 4,
        K: int = 15,
        radius: float = 0.4,
        # optimizer / scheduler
        lr: float = 5e-4,
        lr_factor: float = 10.0,
        b1: float = 0.95,
        b2: float = 0.98,
        weight_decay: float = 0.3,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        # training options
        use_clip: bool = False,
        use_event_loss: bool = False,
        event_threshold: int = 200,
        use_amp: bool = False,
        fine_tune: bool = False,
        ckpt_loaded: str = "",
        # misc
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- model ---
        self.model = PET2(
            input_dim=input_dim - 3,
            hidden_size=hidden_size,
            num_transformers=num_transformers,
            num_heads=num_heads,
            attn_drop=attn_drop,
            mlp_drop=mlp_drop,
            mlp_ratio=mlp_ratio,
            feature_drop=feature_drop,
            num_tokens=num_tokens,
            K=K,
            add_dim=4,
            conditional=False,
            use_time=kwargs.get("use_time", True),
            pid=kwargs.get("use_pid", False),
            num_classes=num_classes,
            mode=kwargs.get("mode", "classifier"),
        )

        # --- losses ---
        self.loss_class = nn.CrossEntropyLoss(reduction="none")
        self.loss_gen = nn.L1Loss(reduction="none")
        self.clip_loss = CLIPLoss()

        # --- flags & params ---
        self.use_clip = use_clip
        self.use_event_loss = use_event_loss
        self.event_threshold = event_threshold
        self.use_amp = use_amp
        self.fine_tune = fine_tune
        self.ckpt_loaded = ckpt_loaded

        # optimizer params
        self.lr = lr
        self.betas = (b1, b2)
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor

    def forward(self, x, y=None, **kwargs):
        return self.model(x, y, **kwargs)

    def _compute_losses(self, outputs, y):
        """
        outputs is a dict with keys: y_pred, z_pred, v, y_perturb, alpha, x_body, z_body
        """
        losses = {}
        loss = torch.tensor(0.0, device=self.device)

        # classification loss
        if outputs["y_pred"] is not None:
            if self.use_event_loss:
                mask = y >= self.event_threshold
                if mask.any():
                    ev = self.loss_class(
                        outputs["y_pred"][mask][:, self.event_threshold :],
                        y[mask] - self.event_threshold,
                    ).mean()
                    losses["loss_class_event"] = ev
                    loss = loss + ev
                inv = ~mask
                if inv.any():
                    cl = self.loss_class(
                        outputs["y_pred"][inv][:, : self.event_threshold], y[inv]
                    ).mean()
                    losses["loss_class"] = cl
                    loss = loss + cl
            else:
                cl = self.loss_class(outputs["y_pred"], y).mean()
                losses["loss_class"] = cl
                loss = loss + cl

        # generation loss
        if outputs["z_pred"] is not None:
            nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
            gen = (
                self.loss_gen(outputs["v"], outputs["z_pred"]).sum((1, 2)) / nonzero
            ).mean()
            losses["loss_gen"] = gen
            loss = loss + gen

        # perturbation loss
        if outputs["y_perturb"] is not None:
            if self.use_event_loss:
                mask = y >= self.event_threshold
                if mask.any():
                    lep = (
                        outputs["alpha"][mask].squeeze(1)
                        * self.loss_class(
                            outputs["y_perturb"][mask][:, self.event_threshold :],
                            y[mask] - self.event_threshold,
                        )
                    ).mean()
                    losses["loss_event_perturb"] = lep
                    loss = loss + lep
                inv = ~mask
                if inv.any():
                    lp = (
                        outputs["alpha"][inv].squeeze(1)
                        * self.loss_class(
                            outputs["y_perturb"][inv][:, : self.event_threshold], y[inv]
                        )
                    ).mean()
                    losses["loss_perturb"] = lp
                    loss = loss + lp
            else:
                lp = (
                    outputs["alpha"].squeeze(1)
                    * self.loss_class(outputs["y_perturb"], y)
                ).mean()
                losses["loss_perturb"] = lp
                loss = loss + lp

        # CLIP loss
        if self.use_clip and outputs.get("x_body") is not None:
            clip = self.clip_loss(
                outputs["x_body"].view(outputs["x_body"].shape[0], -1),
                outputs["z_body"].view(outputs["x_body"].shape[0], -1),
                weight=outputs["alpha"].squeeze(1),
            )
            losses["loss_clip"] = clip
            loss = loss + clip

        losses["loss"] = loss
        return losses

    def training_step(self, batch, batch_idx):
        X, y = batch["X"].float(), batch["y"]
        X, y = X.to(self.device), y.to(self.device)

        model_kwargs = {
            k: (batch[k].to(self.device) if batch[k] is not None else None)
            for k in ("pid", "add_info")
            if (k in batch)
        }

        with amp.autocast(enabled=self.use_amp, device_type="cuda"):
            out = self(X, y, **model_kwargs)
            # PET2 now returns a dict with expected keys
            losses = self._compute_losses(out, y)

        # log everything
        for k, v in losses.items():
            self.log(
                f"train_{k}",
                v,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        X, y = batch["X"].float(), batch["y"]
        X, y = X.to(self.device), y.to(self.device)

        model_kwargs = {
            k: (batch[k].to(self.device) if batch[k] is not None else None)
            for k in ("pid", "add_info")
            if (k in batch)
        }

        with torch.no_grad(), amp.autocast(enabled=self.use_amp, device_type="cuda"):
            out = self(X, y, **model_kwargs)
            losses = self._compute_losses(out, y)

        for k, v in losses.items():
            self.log(
                f"val_{k}",
                v,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return losses["loss"]

    def test_step(self, batch, batch_idx):
        # mirror validation
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # param groups
        pg = get_param_groups(
            self.model,
            wd=self.weight_decay,
            lr=self.lr,
            lr_factor=self.lr_factor,
            fine_tune=self.fine_tune,
        )
        optimizer = Lion(pg, betas=self.betas)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # if fine_tuning a pretrained body, filter shapes
        if self.fine_tune and self.ckpt_loaded:
            ck = torch.load(self.ckpt_loaded, map_location="cpu")
            body_sd = ck["body"]
            self.model.body.load_state_dict(body_sd, strict=False)

            # classifier_head / generator_head if present...
            for head in ("classifier_head", "generator_head"):
                if head in ck and getattr(self.model, head[:-5], None) is not None:
                    getattr(self.model, head[:-5]).load_state_dict(
                        ck[head], strict=False
                    )
