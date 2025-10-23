"""Callback for monitoring the masked prediction task during training."""

import os
from typing import Any, Dict

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import Callback, LightningModule

from omnilearn_lightning.array_utils import ak_subtract, np_to_ak, p4s_from_ptetaphimass
from omnilearn_lightning.plotting.feature_plotting import plot_features
from omnilearn_lightning.plotting.utils import set_mpl_style


def combine_features(x, pid=None, add_info=None):
    """Combine main features, PID, and additional info into a single tensor."""
    if pid is None and add_info is None:
        return x
    if pid is None:
        return torch.cat([x, add_info], dim=-1)
    if add_info is None:
        return torch.cat([x, pid.unsqueeze(-1)], dim=-1)
    return torch.cat([x, pid.unsqueeze(-1), add_info], dim=-1)


class MaskedPredictionCallback(Callback):
    # save the first 10 validation batches for visualization

    def __init__(
        self,
        n_batches: int = 100,
        n_example_jets: int = 8,
        image_path: str = "/tmp",
    ):
        super().__init__()
        self.n_batches = n_batches
        self.n_example_jets = n_example_jets
        self.image_path = image_path
        # make sure the image path exists
        os.makedirs(self.image_path, exist_ok=True)

    def on_validation_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        self.validation_batches = []
        self.validation_outputs = []

    def on_test_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        self.test_batches = []
        self.test_outputs = []

    def on_test_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # mirror validation behaviour for test loop
        if trainer.global_rank != 0:
            return  # only save from rank 0 to avoid duplication
        if len(self.test_batches) < self.n_batches:
            # only keep minimal items and move tensors to CPU immediately
            saved_batch = {}
            # keep X and y only (others are not needed for visualization)
            for key in ("X", "y", "add_info", "pid"):
                if key in batch:
                    val = batch[key]
                    saved_batch[key] = (
                        val.detach().cpu() if torch.is_tensor(val) else val
                    )
            self.test_batches.append(saved_batch)

            saved_outputs = {}
            # masks: move to cpu
            for key in (
                "mask_valid_particle",
                "mask_valid_particle_but_masked",
                "masked_pred",
                "masked_pred_pid",
            ):
                if key in outputs:
                    val = outputs[key]
                    saved_outputs[key] = (
                        val.detach().cpu() if torch.is_tensor(val) else val
                    )

            self.test_outputs.append(saved_outputs)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank != 0:
            return  # only save from rank 0 to avoid duplication
        if len(self.validation_batches) < self.n_batches:
            # only keep minimal items and move tensors to CPU immediately
            saved_batch = {}
            for key in ("X", "y", "add_info", "pid"):
                if key in batch:
                    val = batch[key]
                    saved_batch[key] = (
                        val.detach().cpu() if torch.is_tensor(val) else val
                    )
            self.validation_batches.append(saved_batch)

            saved_outputs = {}
            for key in (
                "mask_valid_particle",
                "mask_valid_particle_but_masked",
                "masked_pred",
                "masked_pred_pid",
            ):
                if key in outputs:
                    val = outputs[key]
                    saved_outputs[key] = (
                        val.detach().cpu() if torch.is_tensor(val) else val
                    )

            self.validation_outputs.append(saved_outputs)

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        # process validation batches and produce plots
        if trainer.global_rank != 0:
            return  # only save from rank 0 to avoid duplication
        self._process_and_log(
            trainer, pl_module, self.validation_batches, self.validation_outputs
        )

    def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        # process test batches and produce plots
        if trainer.global_rank != 0:
            return  # only save from rank 0 to avoid duplication
        # if no test batches were collected, skip
        if not getattr(self, "test_batches", None):
            return
        self._process_and_log(trainer, pl_module, self.test_batches, self.test_outputs)

    def _process_and_log(
        self,
        trainer,
        pl_module: LightningModule,
        batches_list,
        outputs_list,
    ) -> None:
        set_mpl_style()

        if pl_module.model.mode == "classifier":
            print(
                "MaskedPredictionCallback: model is in classifier mode, skipping visualization."
            )
            return

        feature_names = {
            "eta": "Particle $\\Delta\\eta$",
            "phi": "Particle $\\Delta\\phi$",
            "log_pt": "Particle $\\log(p_{T})$",
            "log_E": "Particle $\\log(E)$",
            "PID": "Particle ID",
            "d0val": "$d_{0}$",
            "d0err": "$d_{0}^{err}$",
            "dzval": "$d_{z}$",
            "dzerr": "$d_{z}^{err}$",
        }
        # Prepare awkward arrays for all batches
        batches_survived_ak = []
        batches_masked_pred_ak = []
        batches_masked_true_tokenized_ak = []
        batches_masked_true_ak = []
        batches_jet_labels = []
        batches_pred_token_ids = []

        for i, (batch, output) in enumerate(zip(batches_list, outputs_list)):
            # adjust feature names based on available features in the first batch
            if i == 0:
                if pl_module.use_pid is False or batch.get("pid") is None:
                    feature_names.pop("PID")
                if pl_module.use_add is False or batch.get("add_info") is None:
                    for add_info_name in ["d0val", "d0err", "dzval", "dzerr"]:
                        feature_names.pop(add_info_name)

            targets_fullres = combine_features(
                batch["X"],
                batch.get("pid"),
                batch.get("add_info"),
            )
            targets_transformed_x, targets_transformed_add_info = (
                pl_module.tokenizer.transform(batch["X"], add_info=batch["add_info"])
            )
            targets_transformed = combine_features(
                targets_transformed_x,
                batch.get("pid"),
                targets_transformed_add_info,
            )

            mask_valid_particle = output["mask_valid_particle"][:, :, 0].bool()
            mask_valid_particle_but_masked = output["mask_valid_particle_but_masked"][
                :, :, 0
            ].bool()
            targets_transformed[~mask_valid_particle] = 0  # set invalid points to 0

            # predicted_tokens = output["masked_pred"].argmax(dim=-1)
            probs = torch.nn.functional.softmax(output["masked_pred"], dim=-1)
            # reshape to (batch_size * num_points, num_tokens) for sampling
            probs = probs.view(-1, probs.shape[-1])
            # sample from the categorical distribution
            predicted_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # reshape back to (batch_size, num_points)
            predicted_tokens = predicted_tokens.view(output["masked_pred"].shape[0], -1)

            # same for pid-tokens if pid is used
            if pl_module.use_pid:
                # predicted_pid_tokens = output["masked_pred_pid"].argmax(dim=-1)
                probs_pid = torch.nn.functional.softmax(
                    output["masked_pred_pid"], dim=-1
                )
                probs_pid = probs_pid.view(-1, probs_pid.shape[-1])
                predicted_pid_tokens = torch.multinomial(
                    probs_pid, num_samples=1
                ).squeeze(-1)
                predicted_pid_tokens = predicted_pid_tokens.view(
                    output["masked_pred_pid"].shape[0], -1
                )

            # extract the reconstructed features from the predicted token IDs
            predictions_reconstructed_x, predictions_reconstructed_add_info = (
                pl_module.tokenizer.reconstruct(
                    predicted_tokens,
                    num_features_x=batch["X"].shape[-1],
                )
            )
            predictions_reconstructed = combine_features(
                predictions_reconstructed_x,
                pid=predicted_pid_tokens if pl_module.use_pid else None,
                add_info=predictions_reconstructed_add_info,
            )

            batches_pred_token_ids.append(
                np_to_ak(
                    predicted_tokens.unsqueeze(-1).cpu().numpy(),
                    names=["token_id"],
                    mask=mask_valid_particle_but_masked.cpu().numpy(),
                )
            )
            batches_masked_pred_ak.append(
                np_to_ak(
                    predictions_reconstructed.cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask_valid_particle_but_masked.cpu().numpy(),
                )
            )
            batches_masked_true_ak.append(
                np_to_ak(
                    targets_fullres.cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask_valid_particle_but_masked.cpu().numpy(),
                )
            )
            batches_survived_ak.append(
                np_to_ak(
                    targets_fullres.cpu().numpy(),
                    names=feature_names.keys(),
                    mask=(mask_valid_particle & ~mask_valid_particle_but_masked)
                    .cpu()
                    .numpy(),
                )
            )
            batches_masked_true_tokenized_ak.append(
                np_to_ak(
                    targets_transformed.cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask_valid_particle_but_masked.cpu().numpy(),
                )
            )

            batches_jet_labels.append(batch["y"].cpu().numpy())

        # Concatenate awkward arrays
        x_part_ak_pred = ak.concatenate(batches_masked_pred_ak)
        x_part_ak_true = ak.concatenate(batches_masked_true_ak)
        x_part_ak_survived = ak.concatenate(batches_survived_ak)
        x_part_ak_true_tokenized = ak.concatenate(batches_masked_true_tokenized_ak)
        x_part_ak_pred_token_ids = ak.concatenate(batches_pred_token_ids)
        jet_labels = np.concatenate(batches_jet_labels)

        # shuffle the jets
        rng = np.random.default_rng(seed=42)
        shuffle_idx = rng.permutation(len(x_part_ak_pred))
        x_part_ak_pred = x_part_ak_pred[shuffle_idx]
        x_part_ak_true = x_part_ak_true[shuffle_idx]
        x_part_ak_survived = x_part_ak_survived[shuffle_idx]
        x_part_ak_true_tokenized = x_part_ak_true_tokenized[shuffle_idx]
        x_part_ak_pred_token_ids = x_part_ak_pred_token_ids[shuffle_idx]
        jet_labels = jet_labels[shuffle_idx]

        p4_conversion_kwargs = dict(
            field_name_eta="eta",
            field_name_phi="phi",
            field_name_pt="log_pt",
            pt_is_log=True,
        )
        p4s_survived_and_target = p4s_from_ptetaphimass(
            ak.concatenate([x_part_ak_survived, x_part_ak_true_tokenized], axis=1),
            **p4_conversion_kwargs,
        )
        p4s_survived_and_target_fullres = p4s_from_ptetaphimass(
            ak.concatenate([x_part_ak_survived, x_part_ak_true], axis=1),
            **p4_conversion_kwargs,
        )
        p4s_survived_and_pred = p4s_from_ptetaphimass(
            ak.concatenate([x_part_ak_survived, x_part_ak_pred], axis=1),
            **p4_conversion_kwargs,
        )

        def sum_p4_and_get_feats(p4s_particles):
            p4s_sum = ak.sum(p4s_particles, axis=1)
            return ak.Array(
                {
                    "pt": p4s_sum.pt,
                    "mass": p4s_sum.mass,
                    "eta": p4s_sum.eta,
                    "phi": p4s_sum.phi,
                }
            )

        # calculate sum of each of them
        p4s_survived_and_target_sum = sum_p4_and_get_feats(p4s_survived_and_target)
        p4s_survived_and_target_fullres_sum = sum_p4_and_get_feats(
            p4s_survived_and_target_fullres
        )
        p4s_survived_and_pred_sum = sum_p4_and_get_feats(p4s_survived_and_pred)

        def save_and_log(fig, name):
            filename = f"{self.image_path}/mpm_visualization_epoch_{name}_step{trainer.global_step:06d}.png"
            print(f"Saving image to {filename}")
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            for logger in trainer.loggers:
                if logger.__class__.__name__ == "WandbLogger":
                    print(f"Logging image {filename} to wandb")
                    logger.experiment.log(
                        {f"mpm_visualization_{name}": wandb.Image(filename)}
                    )

        # jet-level distributions
        fig, axarr = plot_features(
            {
                "$\\mathrm{Jet} \\coloneq \\sum \\mathrm{survived} + \\sum \\mathrm{masked}_\\mathrm{full-resolution}$": p4s_survived_and_target_fullres_sum,
                "$\\mathrm{Jet} \\coloneq \\sum \\mathrm{survived} + \\sum \\mathrm{masked}_\\mathrm{tokenized}$": p4s_survived_and_target_sum,
                "$\\mathrm{Jet} \\coloneq \\sum \\mathrm{survived} + \\sum \\mathrm{predicted}$": p4s_survived_and_pred_sum,
            },
            names={
                "pt": "Jet $p_{T}$ [GeV]",
                "mass": "Jet mass [GeV]",
            },
            bins_dict={
                "pt": np.linspace(450, 1050, 50),
                "mass": np.linspace(0, 250, 50),
            },
            flatten=False,
            decorate_ax_kwargs={"yscale": 2.0},
            legend_kwargs={"loc": "upper left", "fontsize": 9},
            ax_size=(3.3, 2.5),
            ratio=True,
        )
        save_and_log(fig, "jet_distributions")

        # jet-level residuals
        fig_res, axarr_res = plot_features(
            {
                "$\\mathrm{Residual}(\\mathrm{Jet}_\\mathrm{predicted}, \\mathrm{Jet}_\\mathrm{full-resolution})$": ak_subtract(
                    p4s_survived_and_pred_sum, p4s_survived_and_target_fullres_sum
                ),
                "$\\mathrm{Residual}(\\mathrm{Jet}_\\mathrm{predicted}, \\mathrm{Jet}_\\mathrm{tokenized})$": ak_subtract(
                    p4s_survived_and_pred_sum, p4s_survived_and_target_sum
                ),
            },
            names={
                "pt": "Jet $p_{T}$ [GeV] residual",
                "mass": "Jet mass [GeV] residual",
            },
            bins_dict={
                "pt": np.linspace(-200, 200, 50),
                "mass": np.linspace(-100, 100, 50),
            },
            flatten=False,
            decorate_ax_kwargs={"yscale": 1.5},
            legend_kwargs={"loc": "upper left"},
        )
        save_and_log(fig_res, "jet_residuals")

        # calculate number of particles and number of unique selected token-IDs
        num_particles = np.sum(ak.num(x_part_ak_pred_token_ids.token_id))
        num_unique_tokens = len(
            np.unique(ak.flatten(x_part_ak_pred_token_ids.token_id))
        )

        # compare distributions of predicted vs true masked vs survived particles
        fig, axarr = plot_features(
            {
                "Survived": x_part_ak_survived,
                "Masked": x_part_ak_true,
                "Predicted": x_part_ak_pred,
            },
            names=feature_names,
            bins_dict={
                "eta": np.linspace(-1, 1, 50),
                "phi": np.linspace(-1, 1, 50),
                "log_pt": np.linspace(-3, 6.5, 50),
                "log_E": np.linspace(-3, 6.5, 50),
            },
            ratio=True,
            legend_kwargs={"loc": "upper left"},
            ax_rows=3 if len(feature_names) > 6 else 2,
        )
        fig.suptitle(
            f"Number of shown particles: {num_particles}, Number of unique tokens predicted: {num_unique_tokens}",
            fontsize=10,
            y=1.04,
        )
        fig.tight_layout()
        save_and_log(fig, "particle_distributions")

        # plot residuals of predicted vs true masked particles
        fig, axarr = plot_features(
            {
                "$\\mathrm{Residual}(\\mathrm{Particle}_\\mathrm{predicted}, \\mathrm{Particle}_\\mathrm{full-resolution})$": ak_subtract(
                    x_part_ak_pred, x_part_ak_true
                ),
                "$\\mathrm{Residual}(\\mathrm{Particle}_\\mathrm{predicted}, \\mathrm{Particle}_\\mathrm{tokenized})$": ak_subtract(
                    x_part_ak_pred, x_part_ak_true_tokenized
                ),
            },
            names={k: feature_names[k] + " residual" for k in feature_names.keys()},
            bins_dict={
                "eta": np.linspace(-1, 1, 50),
                "phi": np.linspace(-1, 1, 50),
                "log_pt": np.linspace(-3, 3, 50),
                "log_E": np.linspace(-3, 3, 50),
                "PID": np.linspace(-10, 10, 50),
                "d0val": np.linspace(-0.1, 0.1, 50),
                "d0err": np.linspace(-0.1, 0.1, 50),
                "dzval": np.linspace(-0.1, 0.1, 50),
                "dzerr": np.linspace(-0.1, 0.1, 50),
            },
            legend_kwargs={"loc": "upper left", "fontsize": 9},
            ax_size=(3.1, 2.0),
            decorate_ax_kwargs={"yscale": 1.4},
            ax_rows=3 if len(feature_names) > 6 else 2,
        )
        fig.suptitle(
            f"Number of shown particles: {num_particles}, Number of unique tokens predicted: {num_unique_tokens}",
            fontsize=10,
            y=1.08,
        )
        fig.tight_layout()
        save_and_log(fig, "particle_residuals")

        # ------ scatter plots for a few example jets ------

        total_width = 20 / 8 * self.n_example_jets
        fig, ax = plt.subplots(3, self.n_example_jets, figsize=(total_width, 8))

        # ax = ax.flatten()
        for i_jet in range(self.n_example_jets):
            ax_top = ax[0, i_jet]
            ax_middle = ax[1, i_jet]
            ax_bottom = ax[2, i_jet]

            ax_top.scatter(
                x_part_ak_survived[i_jet].phi,
                x_part_ak_survived[i_jet].eta,
                s=np.exp(x_part_ak_survived[i_jet].log_pt) * 1,
                c="gray",
                alpha=0.5,
                label="Survived",
            )
            ax_top.scatter(
                x_part_ak_true[i_jet].phi,
                x_part_ak_true[i_jet].eta,
                s=np.exp(x_part_ak_true[i_jet].log_pt) * 1,
                c="C0",
                alpha=0.5,
                label="Masked (full-res)",
            )

            ax_middle.scatter(
                x_part_ak_survived[i_jet].phi,
                x_part_ak_survived[i_jet].eta,
                s=np.exp(x_part_ak_survived[i_jet].log_pt) * 1,
                c="gray",
                alpha=0.5,
                label="Survived",
            )
            ax_middle.scatter(
                x_part_ak_true_tokenized[i_jet].phi,
                x_part_ak_true_tokenized[i_jet].eta,
                s=np.exp(x_part_ak_true_tokenized[i_jet].log_pt) * 1,
                c="C1",
                alpha=0.5,
                label="Masked (tokenized)",
            )

            ax_bottom.scatter(
                x_part_ak_survived[i_jet].phi,
                x_part_ak_survived[i_jet].eta,
                s=np.exp(x_part_ak_survived[i_jet].log_pt) * 1,
                c="gray",
                alpha=0.5,
                label="Survived",
            )
            ax_bottom.scatter(
                x_part_ak_pred[i_jet].phi,
                x_part_ak_pred[i_jet].eta,
                s=np.exp(x_part_ak_pred[i_jet].log_pt) * 1,
                c="C2",
                alpha=0.5,
                label="Predicted",
            )

            # jet type label
            jet_label = jet_labels[i_jet]

            for _ax in [ax_top, ax_middle, ax_bottom]:
                _ax.set_title(f"Jet {i_jet + 1}, $label={jet_label}$", fontsize=10)
                _ax.set_xlim(-1, 1)
                _ax.set_ylim(-1, 1.4)
                _ax.set_xlabel(feature_names["phi"])
                _ax.set_ylabel(feature_names["eta"])
                _ax.legend(loc="upper left", fontsize=8)

                # draw a circle to indicate the jet radius of 0.8
                circle = plt.Circle(
                    (0, 0),
                    0.8,
                    color="black",
                    fill=False,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )
                _ax.add_artist(circle)
                # make legend
                handles, labels = _ax.get_legend_handles_labels()
                handles = [
                    _ax.scatter(
                        [], [], s=20, alpha=0.5, label=label, color=h.get_edgecolor()[0]
                    )
                    for h, label in zip(handles, labels)
                ]
                _ax.legend(handles, labels, loc="upper left", ncol=1, fontsize=8)

        fig.tight_layout()

        save_and_log(fig, "example_jets")

        plt.show()
