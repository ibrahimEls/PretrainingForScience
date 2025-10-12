import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup
from pytorch_lightning import LightningModule
from pytorch_optimizer import Lion

from ..diffusion import get_logsnr_alpha_sigma, perturb
from ..layers import DynamicTanh
from ..modules import PET_body, PET_classifier, PET_generator, PET_masked_predictor
from ..utils import (
    CLIPLoss,
    get_param_groups,
)


class PET2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        codebook_size,
        num_transformers=2,
        num_transformers_head=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        feature_drop=0.0,
        num_tokens=4,
        K=15,
        use_int=True,
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        use_time=False,
        mode="classifier",
        num_classes=2,
    ):
        super().__init__()
        self.mode = mode
        if self.mode not in [
            "classifier",
            "generator",
            "masked_predictor",
            "pretrain",
        ]:
            raise ValueError(f"Mode '{self.mode}' not supported.")

        self.body = PET_body(
            input_dim,
            hidden_size,
            num_transformers=num_transformers,
            num_transf_local=num_transformers_head,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            feature_drop=feature_drop,
            num_tokens=num_tokens,
            K=K,
            use_int=use_int,
            conditional=conditional,
            cond_dim=cond_dim,
            pid=pid,
            pid_dim=pid_dim,
            add_info=add_info,
            add_dim=add_dim,
            use_time=use_time,
        )

        self.num_add = self.body.num_add
        self.classifier = None
        self.generator = None
        if self.mode == "classifier" or self.mode == "pretrain":
            self.classifier = PET_classifier(
                hidden_size,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_classes=num_classes,
            )

        if self.mode == "generator" or self.mode == "pretrain":
            self.generator = PET_generator(
                input_dim,
                hidden_size,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_add=self.num_add,
                num_classes=num_classes,
            )

        if self.mode == "masked_predictor" or self.mode == "pretrain":
            self.masked_predictor = PET_masked_predictor(
                hidden_size,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                # num_tokens=num_tokens,
                # num_add=self.num_add,
                # num_classes=num_classes,
                codebook_size=codebook_size,
            )

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"norm", "scale", "token"}

    def forward(self, x, y, cond=None, pid=None, add_info=None):
        y_pred, y_perturb, z_pred, v, x_body, z_body, masked_pred, mask_body = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        time = torch.rand(size=(x.shape[0],)).to(x.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(time)
        if self.mode == "generator" or self.mode == "pretrain":
            z, v = perturb(x, time)
            z_body = self.body(z, cond, pid, add_info, time)
            z_pred = self.generator(z_body, y)

        if self.mode == "classifier" or self.mode == "pretrain":
            x_body = self.body(x, cond, pid, add_info, torch.zeros_like(time))
            y_pred = self.classifier(x_body)
            if self.mode == "pretrain":
                y_perturb = self.classifier(z_body)

        if self.mode == "masked_predictor" or self.mode == "pretrain":
            x_body, mask_body = self.body(
                x,
                cond,
                pid,
                add_info,
                torch.zeros_like(time),
                return_mask=True,
            )
            masked_pred = self.masked_predictor(x_body, mask=mask_body)

            # remove tokens and time from masked_pred
            masked_pred = masked_pred[:, self.body.num_tokens + 1 :, :]
            masked_mask = mask_body[:, self.body.num_tokens + 1 :]

        return {
            "y_pred": y_pred,
            "y_perturb": y_perturb,
            "z_pred": z_pred,
            "v": v,
            "x_body": x_body,
            "z_body": z_body,
            "alpha": alpha**2,
            "masked_pred": masked_pred,
            "masked_mask": masked_mask,
        }


class PETLightning(LightningModule):
    def __init__(
        self,
        # model
        tokenizer_ckpt: str,
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

        with open(tokenizer_ckpt, "rb") as f:
            print(f"Loading tokenizer from {tokenizer_ckpt}")
            self.tokenizer = torch.load(f, weights_only=False, map_location=self.device)

        # --- model ---
        self.model = PET2(
            input_dim=input_dim - 3,
            hidden_size=hidden_size,
            codebook_size=self.tokenizer.n_clusters,
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
        self.loss_masked = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
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

    def _compute_losses(self, outputs, y, y_masked=None):
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

        # masked prediction loss
        if outputs["masked_pred"] is not None and y_masked is not None:
            # remove tokens and time from masked_pred
            masked_pred = outputs["masked_pred"]
            mask_for_targets = outputs["masked_mask"][:, :, 0]
            # set the targets to -1 where the mask is 0
            y_masked[mask_for_targets == 0] = -1

            B, T, C = masked_pred.shape
            assert y_masked.shape == (B, T), (
                f"y_masked.shape: {y_masked.shape}, masked_pred.shape {masked_pred.shape}"
            )
            lmp = self.loss_masked(
                masked_pred.reshape(B * T, C),
                y_masked.reshape(B * T),
            )
            losses["loss_masked"] = lmp
            loss = loss + lmp

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

    def _shared_step(self, batch, batch_idx, stage):
        """Shared logic for train/val/test steps."""
        X, y = batch["X"].float(), batch["y"]
        X, y = X.to(self.device), y.to(self.device)

        # tokenize the input point clouds
        # make sure tokenizer is on the same device as the model
        if self.tokenizer.kmeans.centroids.device != X.device:
            self.tokenizer.kmeans.centroids = self.tokenizer.kmeans.centroids.to(
                X.device
            )
        y_masked = self.tokenizer.predict(X)

        model_kwargs = {
            k: (batch[k].to(self.device) if batch[k] is not None else None)
            for k in ("pid", "add_info")
            if (k in batch)
        }

        # Use torch.no_grad() for validation and test, allow gradients for training
        if stage == "train":
            with amp.autocast(enabled=self.use_amp, device_type="cuda"):
                out = self(X, y, **model_kwargs)
                losses = self._compute_losses(out, y, y_masked=y_masked)
        else:
            with (
                torch.no_grad(),
                amp.autocast(enabled=self.use_amp, device_type="cuda"),
            ):
                out = self(X, y, **model_kwargs)
                losses = self._compute_losses(out, y, y_masked=y_masked)

        # Log losses with appropriate prefix and settings
        on_step = stage == "train"  # <-- only log on step during training
        for k, v in losses.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=True,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )

        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

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
