"""Callback for monitoring the masked prediction task during training."""

from typing import Any, Dict

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pytorch_lightning import Callback, LightningModule

from omnilearn_lightning.array_utils import ak_subtract, np_to_ak
from omnilearn_lightning.plotting.feature_plotting import plot_features
from omnilearn_lightning.plotting.utils import set_mpl_style


class MaskedPredictionCallback(Callback):
    # save the first 10 validation batches for visualization

    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        self.validation_batches = []
        self.validation_outputs = []

    def on_validation_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if len(self.validation_batches) < 10:
            # add batch to list
            self.validation_batches.append(
                {k: v.detach() if torch.is_tensor(v) else v for k, v in batch.items()}
            )
            self.validation_outputs.append(
                {k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()}
            )

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        set_mpl_style()

        feature_names = {
            "eta": "Particle $\\Delta\\eta$",
            "phi": "Particle $\\Delta\\phi$",
            "log_pt": "Particle $\\log(p_{T})$",
            "log_E": "Particle $\\log(E)$",
        }

        # Prepare awkward arrays for all batches
        batches_survived_ak = []
        batches_masked_pred_ak = []
        batches_masked_true_tokenized_ak = []
        batches_masked_true_ak = []
        batches_jet_labels = []

        for batch, output in zip(self.validation_batches, self.validation_outputs):
            batch_tokenized = pl_module.tokenizer.transform(batch["X"])
            mask_valid_particle = output["mask_valid_particle"][:, :, 0].bool()
            mask_valid_particle_but_masked = output["mask_valid_particle_but_masked"][
                :, :, 0
            ].bool()
            batch_tokenized[~mask_valid_particle] = 0  # set invalid points to 0

            predicted_tokens = output["masked_pred"].argmax(dim=-1)
            # probs = torch.nn.functional.softmax(output["masked_pred"], dim=-1)
            # # reshape to (batch_size * num_points, num_tokens)
            # probs = probs.view(-1, probs.shape[-1])
            # # sample once from the categorical distribution
            # predicted_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # predicted_tokens = predicted_tokens.view(output["masked_pred"].shape[0], -1)
            predictions_reconstructed = pl_module.tokenizer.kmeans.centroids[
                predicted_tokens
            ]
            targets_reconstructed = pl_module.tokenizer.transform(batch["X"])

            batches_masked_pred_ak.append(
                np_to_ak(
                    predictions_reconstructed.cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask_valid_particle_but_masked.cpu().numpy(),
                )
            )
            batches_masked_true_ak.append(
                np_to_ak(
                    batch["X"].cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask_valid_particle_but_masked.cpu().numpy(),
                )
            )
            batches_survived_ak.append(
                np_to_ak(
                    batch["X"].cpu().numpy(),
                    names=feature_names.keys(),
                    mask=(mask_valid_particle & ~mask_valid_particle_but_masked)
                    .cpu()
                    .numpy(),
                )
            )
            batches_masked_true_tokenized_ak.append(
                np_to_ak(
                    batch_tokenized.cpu().numpy(),
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
        jet_labels = np.concatenate(batches_jet_labels)

        # shuffle the jets
        shuffle_idx = np.random.permutation(len(x_part_ak_pred))
        x_part_ak_pred = x_part_ak_pred[shuffle_idx]
        x_part_ak_true = x_part_ak_true[shuffle_idx]
        x_part_ak_survived = x_part_ak_survived[shuffle_idx]
        x_part_ak_true_tokenized = x_part_ak_true_tokenized[shuffle_idx]
        jet_labels = jet_labels[shuffle_idx]

        def save_and_log(fig, name):
            filename = (
                f"/tmp/mpm_visualization_epoch_{name}_step{trainer.global_step:06d}.png"
            )
            fig.savefig(filename, dpi=300)
            for logger in trainer.loggers:
                if logger.__class__.__name__ == "WandbLogger":
                    print(f"Logging image {filename} to wandb")
                    logger.experiment.log(
                        {f"mpm_visualization_{name}": wandb.Image(filename)}
                    )

        # compare distributions of predicted vs true masked vs survived particles
        fig, axarr = plot_features(
            {
                "Survived": x_part_ak_survived,
                "Predicted": x_part_ak_pred,
                "True Masked": x_part_ak_true,
            },
            names=feature_names,
            bins_dict={
                "eta": np.linspace(-1, 1, 50),
                "phi": np.linspace(-1, 1, 50),
                "log_pt": np.linspace(-3, 6.5, 50),
                "log_E": np.linspace(-3, 6.5, 50),
            },
            ratio=True,
        )
        save_and_log(fig, "particle_distributions")

        # plot residuals of predicted vs true masked particles
        fig, axarr = plot_features(
            {
                "Predicted - True Masked": ak_subtract(x_part_ak_pred, x_part_ak_true),
                "Predicted - True Masked (Tokenized)": ak_subtract(
                    x_part_ak_pred, x_part_ak_true_tokenized
                ),
            },
            names=feature_names,
            bins_dict={
                "eta": np.linspace(-1, 1, 50),
                "phi": np.linspace(-1, 1, 50),
                "log_pt": np.linspace(-3, 3, 50),
                "log_E": np.linspace(-3, 3, 50),
            },
        )
        save_and_log(fig, "particle_residuals")

        # ------ scatter plots for a few example jets ------

        fig, ax = plt.subplots(3, 8, figsize=(20, 7.5))

        # ax = ax.flatten()
        for i_jet in range(8):
            ax_top = ax[0, i_jet]
            ax_middle = ax[1, i_jet]
            ax_bottom = ax[2, i_jet]

            ax_top.scatter(
                x_part_ak_survived[i_jet].phi,
                x_part_ak_survived[i_jet].eta,
                s=np.exp(x_part_ak_survived[i_jet].log_pt) * 1,
                c="C0",
                alpha=0.5,
                label="Survived",
            )
            ax_top.scatter(
                x_part_ak_true[i_jet].phi,
                x_part_ak_true[i_jet].eta,
                s=np.exp(x_part_ak_true[i_jet].log_pt) * 1,
                c="C1",
                alpha=0.5,
                label="Masked",
            )

            ax_middle.scatter(
                x_part_ak_survived[i_jet].phi,
                x_part_ak_survived[i_jet].eta,
                s=np.exp(x_part_ak_survived[i_jet].log_pt) * 1,
                c="C0",
                alpha=0.5,
                label="Survived",
            )
            ax_middle.scatter(
                x_part_ak_true_tokenized[i_jet].phi,
                x_part_ak_true_tokenized[i_jet].eta,
                s=np.exp(x_part_ak_true_tokenized[i_jet].log_pt) * 1,
                c="C3",
                alpha=0.5,
                label="Masked (tokenized)",
            )

            ax_bottom.scatter(
                x_part_ak_survived[i_jet].phi,
                x_part_ak_survived[i_jet].eta,
                s=np.exp(x_part_ak_survived[i_jet].log_pt) * 1,
                c="C0",
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
