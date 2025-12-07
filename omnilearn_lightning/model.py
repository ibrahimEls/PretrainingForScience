import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup
from omnilearned.diffusion import get_logsnr_alpha_sigma, perturb
from omnilearned.layers import DynamicTanh
from omnilearned.network import PET_body, PET_classifier, PET_generator
from omnilearned.utils import CLIPLoss, get_loss
from pytorch_lightning import LightningModule
from pytorch_optimizer import Lion
from torch.optim.lr_scheduler import OneCycleLR

from .array_utils import replace_masked_positions, set_fraction_ones_to_zeros
from .modules import MPM_head
from .utils import get_param_groups


def get_logs(device):
    logs_buff = torch.zeros((6), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_clip"] = logs_buff[3].view(-1)
    logs["loss_class_event"] = logs_buff[4].view(-1)
    logs["loss_masked_pid"] = logs_buff[5].view(-1)

    return logs


class PET2(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
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
        pos_encoding_type="sort_descending_in_masked_subset",
        skip=False,
        num_gen_classes=1,
        mpm_features="kin_pid_add",
    ):
        super().__init__()
        self.mode = mode
        self.mpm_features = mpm_features

        if self.mode not in [
            "classifier",
            "generator",
            "mpm",
            "classifier+generator",
            "classifier+mpm",
            "generator+mpm",
            "pretrain",
        ]:
            raise ValueError(f"Mode '{self.mode}' not supported.")

        self.body = PET_body(
            input_dim,
            base_dim,
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
            use_time=self.mode
            in ["classifier+generator", "generator+mpm", "generator", "pretrain"],
            skip=skip,
        )

        self.pos_encoding_type = pos_encoding_type
        print(f"Using pos_encoding_type: {self.pos_encoding_type}")

        self.num_add = self.body.num_add
        self.classifier = None
        self.generator = None
        # Initialize classifier if needed
        if "classifier" in self.mode or self.mode == "pretrain":
            self.classifier = PET_classifier(
                base_dim,
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

        # Initialize generator if needed
        if "generator" in self.mode or self.mode == "pretrain":
            self.generator = PET_generator(
                input_dim
                if num_gen_classes == 1
                else num_gen_classes,  # diffusion or segmentation
                base_dim,
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
                skip_pid=num_classes == 1,
            )

        # Initialize MPM head if needed
        if "mpm" in self.mode or self.mode == "pretrain":
            self.mpm_head = MPM_head(
                base_dim,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_add=self.num_add,
                # num_classes=num_classes,
                codebook_size_continuous=codebook_size,
                codebook_size_pid=pid_dim if "pid" in self.mpm_features else None,
            )
            # initialize trainable mask token embeddings
            # those are used to fill in the masked positions
            max_particles = 150  # maximum number of particles
            self.mask_embeddings = torch.nn.Parameter(
                torch.randn(max_particles, base_dim)
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

    def forward(self, x, y, cond=None, pid=None, add_info=None, masking_fraction=0.0):
        (
            y_pred,
            y_perturb,
            z_pred,
            v,
            x_body,
            z_body,
            masked_pred_continuous,
            masked_pred_pid,
            v_weight,
            mask_valid_particle,
            mask_valid_particle_but_masked,
        ) = (
            None,
            None,
            None,
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

        # Generator forward pass
        if "generator" in self.mode or self.mode == "pretrain":
            z, v, v_weight = perturb(x, time)
            z_body = self.body(z, cond, pid, add_info, time)
            z_pred = self.generator(z_body, y)

        # Classifier forward pass
        if "classifier" in self.mode or self.mode == "pretrain":
            x_body = self.body(x, cond, pid, add_info, torch.zeros_like(time))
            y_pred = self.classifier(x_body)
            # if self.mode == "pretrain":
            #     y_perturb = self.classifier(z_body)

        # MPM forward pass
        if "mpm" in self.mode or self.mode == "pretrain":
            # this is where the masking happens
            # --> we set the specified fraction of ones in the mask to zeros
            #     and use that to mask the input point cloud
            mask_before_masking = (x[:, :, 3] != 0).int()
            mask_after_masking = set_fraction_ones_to_zeros(
                mask_before_masking,
                fraction=masking_fraction,
            )
            # use pT as position encoding feature
            pos_encoding_feature = x[:, :, 2].clone()
            # multiply x by mask_after_masking as the padding mask in the body
            # is calculated from the input x (assuming 0-padded points)
            x = x * mask_after_masking.unsqueeze(-1)

            mask_valid_particle_but_masked = (
                (mask_before_masking - mask_after_masking).bool().unsqueeze(-1)
            )

            x_body = self.body(
                x,
                cond,
                pid,
                add_info,
                torch.zeros_like(time),
            )

            x_body_masked = x_body.clone()
            replace_masked_positions(
                x_body_masked[:, self.body.num_tokens + self.body.num_add :],
                mask_is_valid=mask_before_masking.int(),
                mask_is_valid_corrupted=mask_after_masking.int(),
                mask_is_valid_but_masked=mask_valid_particle_but_masked[:, :, 0].int(),
                pos_encoding_feature=pos_encoding_feature,
                pos_encoding_type=self.pos_encoding_type,
                vectors_to_insert=self.mask_embeddings,
            )

            # add tokens and additional points in the point cloud back to the mask
            mask_for_masked_pred_head = torch.cat(
                [
                    torch.ones(
                        (x.shape[0], self.body.num_tokens + self.body.num_add),
                        device=x.device,
                    ),
                    mask_before_masking,
                ],
                dim=1,
            ).unsqueeze(-1)

            masked_pred_continuous, masked_pred_pid, mask_valid_particle = (
                self.mpm_head(x_body_masked, mask=mask_for_masked_pred_head.bool())
            )

        return {
            "y_pred": y_pred,
            "y_perturb": y_perturb,
            "z_pred": z_pred,
            "v": v,
            "v_weight": v_weight,
            "x_body": x_body,
            "z_body": z_body,
            "alpha": alpha**2,
            "masked_pred": masked_pred_continuous,
            "masked_pred_pid": masked_pred_pid,
            "pid": pid,
            "mask_valid_particle": mask_valid_particle,
            "mask_valid_particle_but_masked": mask_valid_particle_but_masked,
        }


class PETLightning(LightningModule):
    def __init__(
        self,
        # model
        tokenizer_ckpt: str,
        masking_fraction: float = 0.4,
        pos_encoding_type: str = "sort_descending_in_masked_subset",
        num_feat: int = 4,
        num_classes: int = 2,
        attn_drop: float = 0.1,
        mlp_drop: float = 0.1,
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
        mode="pretrain",
        use_one_cycle=False,
        model_params={},
        use_add=False,
        use_pid=False,
        add_dim=4,
        num_gen_classes=1,
        all_head=False,
        use_weights_for_pid_loss=False,
        mpm_features="kin_pid_add",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = None
        self.use_mpm = "mpm" in mode or mode == "pretrain"
        self.mpm_features = mpm_features

        if self.use_mpm:
            # verify that mpm_features is valid
            supported_mpm_features = [
                "kin",
                # "kin_pid",
                # "kin_add",
                "kin_pid_add",
            ]
            if mpm_features not in supported_mpm_features:
                raise ValueError(
                    f"mpm_features '{mpm_features}' not supported. Supported: {supported_mpm_features}"
                )
            # if no tokenizer checkpoint is provided, use regression loss
            if tokenizer_ckpt is None:
                print(
                    "MPM will be trained with regression loss (no tokenizer ckpt provided)."
                )
            else:
                with open(tokenizer_ckpt, "rb") as f:
                    print(f"Loading tokenizer from {tokenizer_ckpt}")
                    self.tokenizer = torch.load(
                        f, weights_only=False, map_location=self.device
                    )
                    # verify that tokenizer corresponds to the requested mpm_features
                    if (
                        "add" in mpm_features
                        and self.tokenizer.scale_factors_add_info is None
                    ):
                        raise ValueError(
                            "Tokenizer was trained without add_info, but mpm_features "
                            f"'{mpm_features}' requires it."
                        )
                    if (
                        mpm_features == "kin"
                        and self.tokenizer.scale_factors_add_info is not None
                    ):
                        raise ValueError(
                            "Tokenizer has vertex information scale factors, but "
                            "mpm_features is 'kin'. Use a tokenizer trained "
                            "without those additional features."
                        )

        number_continuous_features = num_feat
        if use_add:
            number_continuous_features += add_dim

        # --- model ---
        self.model = PET2(
            input_dim=num_feat,
            cond_dim=3,
            pid=True,
            pid_dim=9,
            add_info=use_add,
            add_dim=add_dim,
            mode=mode,
            num_classes=num_classes,
            num_gen_classes=num_gen_classes,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            feature_drop=feature_drop,
            codebook_size=self.tokenizer.n_clusters
            if (self.use_mpm and self.tokenizer is not None)
            else number_continuous_features,
            pos_encoding_type=pos_encoding_type,
            mpm_features=mpm_features,
            **model_params,
        )

        # --- losses ---
        self.loss_class = nn.CrossEntropyLoss()
        self.loss_gen = nn.MSELoss()
        if self.tokenizer is not None:
            self.loss_masked_continuous = nn.CrossEntropyLoss(
                ignore_index=-1, reduction="mean"
            )
        else:
            self.loss_masked_continuous = nn.L1Loss(reduction="mean")
        self.loss_masked_pid = (
            nn.CrossEntropyLoss(
                ignore_index=-1,
                reduction="mean",
                weight=torch.tensor(
                    [4.2, 333, 234, 0, 221, 328, 2.4, 9.3, 4.2],
                    device=self.device,
                )
                if use_weights_for_pid_loss
                else None,
            )
            if "pid" in self.mpm_features
            else None
        )
        self.clip_loss = CLIPLoss()
        self.use_one_cycle = use_one_cycle

        # --- flags & params ---
        self.use_clip = use_clip
        self.use_event_loss = use_event_loss
        self.event_threshold = event_threshold
        self.use_amp = use_amp
        self.fine_tune = fine_tune
        self.all_head = all_head
        self.ckpt_loaded = ckpt_loaded
        self.use_pid = use_pid
        self.use_add = use_add

        # optimizer params
        self.lr = lr
        self.betas = (b1, b2)
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor

    def forward(self, x, y=None, **kwargs):
        return self.model(x, y, **kwargs)

    def _compute_MPM_loss(self, outputs, y, y_masked=None):
        """
        outputs is a dict with keys: y_pred, z_pred, v, y_perturb, alpha, x_body, z_body
        """
        mpm_logs = {}
        loss = torch.tensor(0.0, device=self.device)

        # masked prediction loss
        if outputs["masked_pred"] is not None and y_masked is not None:
            # remove tokens and time from masked_pred
            masked_pred = outputs["masked_pred"]
            mask_for_targets = outputs["mask_valid_particle_but_masked"][:, :, 0]
            # set the targets to -1 where the mask is 0
            y_masked[mask_for_targets == 0] = -1

            B, T, C = masked_pred.shape

            if self.tokenizer is not None:
                # predicting token-IDs
                assert y_masked.shape == (B, T), (
                    f"y_masked.shape: {y_masked.shape}, masked_pred.shape {masked_pred.shape}"
                )
                lmp = self.loss_masked_continuous(
                    masked_pred.reshape(B * T, C),
                    y_masked.reshape(B * T),
                )
            else:
                # regression loss on continuous features
                assert y_masked.shape == (B, T, C), (
                    f"y_masked.shape: {y_masked.shape}, masked_pred.shape {masked_pred.shape}"
                )
                lmp = self.loss_masked_continuous(
                    masked_pred[mask_for_targets], y_masked[mask_for_targets]
                )

            mpm_logs["loss_masked"] = lmp
            loss = loss + lmp

            # if pid is used, add its loss too
            if self.use_pid and outputs.get("masked_pred_pid") is not None:
                masked_pred_pid = outputs["masked_pred_pid"]
                y_masked_pid = outputs["pid"].clone()
                y_masked_pid[mask_for_targets == 0] = -1
                lmp_pid = self.loss_masked_pid(
                    masked_pred_pid.reshape(B * T, masked_pred_pid.shape[-1]),
                    y_masked_pid.reshape(B * T).long(),
                )
                mpm_logs["loss_masked_pid"] = lmp_pid
                loss = loss + lmp_pid
                mpm_logs["masked_pred_pid"] = masked_pred_pid if self.use_pid else None

            mpm_logs["masked_pred"] = masked_pred
            mpm_logs["mask_valid_particle"] = outputs["mask_valid_particle"]
            mpm_logs["mask_valid_particle_but_masked"] = outputs[
                "mask_valid_particle_but_masked"
            ]
            mpm_logs["y_masked"] = y_masked

        return mpm_logs, loss

    def _shared_step(self, batch, batch_idx, stage):
        """Shared logic for train/val/test steps."""
        X, y = batch["X"].float(), batch["y"]
        X, y = X.to(self.device), y.to(self.device)

        y_masked = None
        model_kwargs = {
            k: (batch[k].to(self.device) if batch[k] is not None else None)
            for k in ("pid", "add_info")
            if (k in batch)
        }

        if batch.get("data_pid") is not None:
            data_pid = batch["data_pid"].to(X.device)
        else:
            data_pid = None

        if self.use_mpm:
            # tokenize the input point clouds
            # make sure tokenizer is on the same device as the model
            model_kwargs["masking_fraction"] = self.hparams.masking_fraction
            if self.tokenizer is not None:
                if self.tokenizer.kmeans.centroids.device != X.device:
                    self.tokenizer.kmeans.centroids = (
                        self.tokenizer.kmeans.centroids.to(X.device)
                    )
                y_masked = self.tokenizer.predict(X, add_info=model_kwargs["add_info"])
            else:
                # regression loss on the continuous features
                y_masked = (
                    torch.cat([X.clone(), batch["add_info"].to(X.device)], dim=-1)
                    if self.use_add
                    else X.clone()
                )

        logs = get_logs(device=X.device)
        if stage == "train":
            with amp.autocast(enabled=self.use_amp, device_type="cuda"):
                out = self(X, y, **model_kwargs)

                # if we are in MPM only mode, skip the standard loss computation
                # all other modes are compatible with the `get_loss` function
                # as they have at least one of the default loss components
                # but when using only MPM, the code crashes when running
                # loss.detach() here: https://github.com/ViniciusMikuni/OmniLearned/blob/922355db6269061af9e8a36a7c6a118e7c6609c4/src/omnilearned/utils.py#L272
                # while being initialized as a standard python float here: https://github.com/ViniciusMikuni/OmniLearned/blob/922355db6269061af9e8a36a7c6a118e7c6609c4/src/omnilearned/utils.py#L221
                if self.model.mode == "mpm":
                    loss = torch.tensor(0.0, device=self.device)
                else:
                    loss = get_loss(
                        out,
                        y,
                        self.loss_class,
                        self.loss_gen,
                        self.use_event_loss,
                        self.use_clip,
                        CLIPLoss(),
                        logs,
                        data_pid=data_pid,
                    )

                mpm_logs, mpm_loss = self._compute_MPM_loss(out, y, y_masked=y_masked)
                logs.update(mpm_logs)
                logs["loss"] = loss + mpm_loss

        else:
            with (
                torch.no_grad(),
                amp.autocast(enabled=self.use_amp, device_type="cuda"),
            ):
                out = self(X, y, **model_kwargs)
                # same reason as above
                if self.model.mode == "mpm":
                    loss = torch.tensor(0.0, device=self.device)
                else:
                    loss = get_loss(
                        out,
                        y,
                        self.loss_class,
                        self.loss_gen,
                        self.use_event_loss,
                        self.use_clip,
                        CLIPLoss(),
                        logs,
                        data_pid=data_pid,
                    )

                mpm_logs, mpm_loss = self._compute_MPM_loss(out, y, y_masked=y_masked)
                logs.update(mpm_logs)
                logs["loss"] = loss + mpm_loss

        # Log losses with appropriate prefix and settings
        on_step = stage == "train"  # <-- only log on step during training
        for k, v in logs.items():
            if "loss" not in k:
                continue
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=True,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )

        return logs

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
            all_head=self.all_head,
        )

        if self.use_one_cycle:
            optimizer = torch.optim.AdamW(pg, betas=self.betas)
            group_max_lrs = [g["lr"] for g in optimizer.param_groups]
            total = self.trainer.estimated_stepping_batches

            scheduler = OneCycleLR(
                optimizer,
                max_lr=group_max_lrs,  # list, not a single float
                total_steps=total,
                pct_start=0.1,
                anneal_strategy="cos",
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "one_cycle",
                },
            }

        else:
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
                    "frequency": 1,
                    "name": "Lion",
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
