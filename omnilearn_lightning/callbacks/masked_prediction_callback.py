"""Callback for monitoring the masked prediction task during training."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Callback, LightningModule

from omnilearn_lightning.array_utils import ak_subtract, np_to_ak
from omnilearn_lightning.plotting.feature_plotting import plot_features


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
        batch = self.validation_batches[0]
        output = self.validation_outputs[0]

        batch_tokenized = pl_module.tokenizer.transform(batch["X"])
        mask = output["masked_mask"]
        batch_tokenized[~mask] = 0  # set invalid points to 0
        print(f"batch_tokenized.shape: {batch_tokenized.shape}")

        predicted_tokens = output["masked_pred"].argmax(dim=-1)
        print(f"predicted_tokens.shape: {predicted_tokens.shape}")
        # reconstruct to original features
        reconstructed = pl_module.tokenizer.kmeans.centroids[predicted_tokens]
        print(f"reconstructed.shape: {reconstructed.shape}")

        feature_names = ["eta", "phi", "log_pt", "log_E"]

        # convert to awkward array for evaluation and plotting
        batch_tokenized_ak = np_to_ak(
            batch_tokenized.cpu().numpy(), names=feature_names, mask=mask.cpu().numpy()
        )
        batch_ak = np_to_ak(
            batch["X"].cpu().numpy(), names=feature_names, mask=mask.cpu().numpy()
        )
        batch_pred_ak = np_to_ak(
            reconstructed.cpu().numpy(), names=feature_names, mask=mask.cpu().numpy()
        )

        fig, axarr = plot_features(
            {
                "Original": batch_ak,
                "Tokenized": batch_tokenized_ak,
                "Predicted": batch_pred_ak,
            },
            names=batch_tokenized_ak.fields,
        )

        # plot the resolution of tokenized vs predicted
        fig, axarr = plot_features(
            {
                "Tokenized - Predicted": ak_subtract(batch_tokenized_ak, batch_pred_ak),
            },
            names=batch_tokenized_ak.fields,
        )

        plt.show()
