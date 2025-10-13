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
        all_batch_ak = []
        all_batch_tokenized_ak = []
        all_batch_pred_ak = []

        for batch, output in zip(self.validation_batches, self.validation_outputs):
            batch_tokenized = pl_module.tokenizer.transform(batch["X"])
            mask = output["masked_mask"]
            batch_tokenized[~mask] = 0  # set invalid points to 0

            predicted_tokens = output["masked_pred"].argmax(dim=-1)
            reconstructed = pl_module.tokenizer.kmeans.centroids[predicted_tokens]

            all_batch_tokenized_ak.append(
                np_to_ak(
                    batch_tokenized.cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask.cpu().numpy(),
                )
            )
            all_batch_ak.append(
                np_to_ak(
                    batch["X"].cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask.cpu().numpy(),
                )
            )
            all_batch_pred_ak.append(
                np_to_ak(
                    reconstructed.cpu().numpy(),
                    names=feature_names.keys(),
                    mask=mask.cpu().numpy(),
                )
            )

        # Concatenate awkward arrays
        x_part_ak_original = ak.concatenate(all_batch_ak)
        x_part_ak_tokenized = ak.concatenate(all_batch_tokenized_ak)
        x_part_ak_pred = ak.concatenate(all_batch_pred_ak)

        fig, axarr = plot_features(
            {
                "Original": x_part_ak_original,
                "Tokenized": x_part_ak_tokenized,
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
        )

        def save_and_log(fig, name):
            filename = (
                f"/tmp/mpm_visualization_epoch_{name}_step{trainer.global_step:06d}.png"
            )
            fig.savefig(filename)
            for logger in trainer.loggers:
                if logger.__class__.__name__ == "WandbLogger":
                    logger.experiment.log(
                        {f"mpm_visualization_{name}": wandb.Image(filename)}
                    )
                else:
                    print(
                        f"Logger {logger.__class__.__name__} not supported for image logging."
                    )

        save_and_log(fig, "particle_distributions")

        fig, axarr = plot_features(
            {
                "Tokenized - Predicted": ak_subtract(
                    x_part_ak_tokenized, x_part_ak_pred
                ),
                "Original - Predicted": ak_subtract(x_part_ak_original, x_part_ak_pred),
                "Original - Tokenized": ak_subtract(
                    x_part_ak_original, x_part_ak_tokenized
                ),
            },
            bins_dict={
                "eta": np.linspace(-1, 1, 50),
                "phi": np.linspace(-1, 1, 50),
                "log_pt": np.linspace(-3, 3, 50),
                "log_E": np.linspace(-3, 3, 50),
            },
            names=feature_names,
        )
        save_and_log(fig, "residuals")

        plt.show()
