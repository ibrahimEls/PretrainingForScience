"""Callback for generating jets during the test loop and returning a comparison ak array.

Collects batches during the test loop, runs diffusion-based generation at test_epoch_end,
and produces the same output structure as ``generation_utils.generate_and_postprocess``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import awkward as ak
import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule

from omnilearn_lightning.array_utils import p4s_from_ptetaphimass
from omnilearn_lightning.generation_utils import (
    JetSubstructure,
    get_batch_and_generate,
    save_gen_output,
)


class JetGenerationCallback(Callback):
    """Generate jets via diffusion during the test loop and build a comparison ak array.

    At the end of the test epoch the callback stores its result in
    ``self.gen_output`` – an ``ak.Array`` with the same structure returned by
    ``generation_utils.generate_and_postprocess``::

        {
            "x_ak":         {"original": …, "generated": …},
            "p4s":          {"original": …, "generated": …},
            "p4s_sum":      {"original": …, "generated": …},
            "substructure": {"original": …, "generated": …},
            "y":            …,
        }

    Parameters
    ----------
    n_batches : int
        Number of test batches to collect and generate from.
    feature_names_x : dict[str, str]
        Mapping of feature keys (e.g. ``"log_pt"``) to display names.
        Must match the feature order in the data.
    n_sampling_steps : int
        Number of diffusion sampling steps.
    set_pid_to_zeros : bool
        Zero-out PID features before generation.
    set_add_info_to_zeros : bool
        Zero-out additional-info features before generation.
    output_path : str | Path | None
        If given, save the result to this parquet file.
    verbose : bool
        Forward to ``get_batch_and_generate``.
    """

    def __init__(
        self,
        n_batches: int = 10,
        feature_names_x: Optional[Dict[str, str]] = None,
        n_sampling_steps: int = 128,
        set_pid_to_zeros: bool = False,
        set_add_info_to_zeros: bool = False,
        output_path: Optional[str | Path] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.n_batches = n_batches
        self.feature_names_x = feature_names_x or {
            "eta": "Particle Δη",
            "phi": "Particle Δφ",
            "log_pt": "Particle log(pT)",
            "log_E": "Particle log(E)",
        }
        self.n_sampling_steps = n_sampling_steps
        self.set_pid_to_zeros = set_pid_to_zeros
        self.set_add_info_to_zeros = set_add_info_to_zeros
        self.output_path = Path(output_path) if output_path is not None else None
        self.verbose = verbose

        # Will be populated at test_epoch_end
        self.gen_output: Optional[ak.Array] = None

    # ------------------------------------------------------------------ #
    #  Lightning hooks                                                     #
    # ------------------------------------------------------------------ #

    def on_test_epoch_start(self, trainer, pl_module: LightningModule) -> None:
        self._batches: list[Dict[str, Any]] = []

    def on_test_batch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if len(self._batches) >= self.n_batches:
            return

        # Keep a CPU-only snapshot of the batch (same keys generation_utils needs)
        saved: Dict[str, Any] = {}
        for key in ("X", "y", "cond", "pid", "add_info"):
            if key in batch and batch[key] is not None:
                val = batch[key]
                saved[key] = val.detach().cpu() if torch.is_tensor(val) else val
        self._batches.append(saved)

    def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if not self._batches:
            print("JetGenerationCallback: no batches collected, skipping.")
            return

        print(
            f"JetGenerationCallback: generating from {len(self._batches)} "
            f"collected batches ({self.n_sampling_steps} sampling steps)..."
        )

        originals = []
        generateds = []
        y_values = []

        for batch in self._batches:
            x_orig, x_gen = get_batch_and_generate(
                batch=batch,
                lightning_model=pl_module,
                feature_names_x=self.feature_names_x,
                verbose=self.verbose,
                n_sampling_steps=self.n_sampling_steps,
                set_pid_to_zeros=self.set_pid_to_zeros,
                set_add_info_to_zeros=self.set_add_info_to_zeros,
            )
            originals.append(x_orig)
            generateds.append(x_gen)
            y_values.append(
                batch["y"].cpu().numpy()
                if torch.is_tensor(batch["y"])
                else np.asarray(batch["y"])
            )

        # ---- assemble the comparison ak array ---- #
        x_ak = {
            "original": ak.concatenate(originals, axis=0),
            "generated": ak.concatenate(generateds, axis=0),
        }

        p4_kwargs = dict(
            field_name_eta="eta",
            field_name_phi="phi",
            field_name_pt="log_pt",
            pt_is_log=True,
        )
        p4s = {
            "original": p4s_from_ptetaphimass(x_ak["original"], **p4_kwargs),
            "generated": p4s_from_ptetaphimass(x_ak["generated"], **p4_kwargs),
        }
        p4s_sum = {
            "original": ak.sum(p4s["original"], axis=1),
            "generated": ak.sum(p4s["generated"], axis=1),
        }

        print("JetGenerationCallback: calculating jet substructure...")
        substructure = {
            "original": JetSubstructure(p4s["original"]).get_substructure_as_ak_array(),
            "generated": JetSubstructure(
                p4s["generated"]
            ).get_substructure_as_ak_array(),
        }

        self.gen_output = ak.Array(
            {
                "x_ak": x_ak,
                "p4s": p4s,
                "p4s_sum": p4s_sum,
                "substructure": substructure,
                "y": ak.concatenate(y_values, axis=0),
            }
        )

        print(
            f"JetGenerationCallback: done – {len(self.gen_output['y'])} jets generated."
        )

        if self.output_path is not None:
            save_gen_output(self.gen_output, self.output_path)
