"""Callback for generating jets during the test loop and returning a comparison ak array.

Collects batches during the test loop, runs diffusion-based generation at test_epoch_end,
and produces the same output structure as ``generation_utils.generate_and_postprocess``.

Supports multi-GPU (DDP): every rank generates independently, then rank 0
merges all partial results via temporary parquet files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import awkward as ak
import numpy as np
import torch
import torch.distributed as dist
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
    ``self.gen_output`` (rank 0 only) – an ``ak.Array`` with the same
    structure returned by ``generation_utils.generate_and_postprocess``::

        {
            "x_ak":         {"original": …, "generated": …},
            "p4s":          {"original": …, "generated": …},
            "p4s_sum":      {"original": …, "generated": …},
            "substructure": {"original": …, "generated": …},
            "y":            …,
        }

    **Multi-GPU support**: every rank collects and generates from its own
    batches.  Per-rank particle-level results (``x_ak`` and ``y``) are
    written to temporary parquet files.  Rank 0 then loads, concatenates,
    and computes jet substructure on the merged result.

    Parameters
    ----------
    output_path : str | Path
        Directory to save the final parquet file and per-rank temporary files.
        Created automatically if it does not exist.
    n_batches : int
        Number of test batches **per rank** to collect and generate from.
    feature_names_x : dict[str, str]
        Mapping of feature keys (e.g. ``"log_pt"``) to display names.
        Must match the feature order in the data.
    n_sampling_steps : int
        Number of diffusion sampling steps.
    set_pid_to_zeros : bool
        Zero-out PID features before generation.
    set_add_info_to_zeros : bool
        Zero-out additional-info features before generation.
    verbose : bool
        Forward to ``get_batch_and_generate``.
    """

    def __init__(
        self,
        output_path: str | Path,
        n_batches: int = 10,
        feature_names_x: Optional[Dict[str, str]] = None,
        n_sampling_steps: int = 128,
        set_pid_to_zeros: bool = False,
        set_add_info_to_zeros: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
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
        self.verbose = verbose

        # Will be populated at test_epoch_end (rank 0 only)
        self.gen_output: Optional[ak.Array] = None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    @staticmethod
    def _world_size() -> int:
        return (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )

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
        # Every rank collects its own batches
        if len(self._batches) >= self.n_batches:
            return

        saved: Dict[str, Any] = {}
        for key in ("X", "y", "cond", "pid", "add_info"):
            if key in batch and batch[key] is not None:
                val = batch[key]
                saved[key] = val.detach().cpu() if torch.is_tensor(val) else val
        self._batches.append(saved)

    def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        rank = self._rank()
        world_size = self._world_size()

        # -- Phase 1: every rank generates from its own collected batches --
        if not self._batches:
            originals, generateds, y_values = [], [], []
        else:
            originals, generateds, y_values = self._generate_from_batches(pl_module)

        # -- Phase 2: merge across ranks --
        if world_size > 1:
            x_ak_all, y_all = self._gather_across_ranks(
                originals, generateds, y_values, rank, world_size
            )
        else:
            x_ak_all, y_all = self._build_x_ak(originals, generateds, y_values)

        # -- Phase 3: rank 0 computes substructure and assembles final output --
        if rank != 0:
            return

        if x_ak_all is None:
            print("JetGenerationCallback: no jets collected across ranks, skipping.")
            return

        self.gen_output = self._postprocess(x_ak_all, y_all)

        print(
            f"JetGenerationCallback: done – "
            f"{len(self.gen_output['y'])} jets generated "
            f"(from {world_size} rank(s))."
        )

        save_gen_output(self.gen_output, self.output_path / "gen_output.parquet")

    # ------------------------------------------------------------------ #
    #  Internal methods                                                    #
    # ------------------------------------------------------------------ #

    def _generate_from_batches(self, pl_module):
        """Run diffusion generation on each collected batch. Returns lists."""
        rank = self._rank()
        print(
            f"JetGenerationCallback [rank {rank}]: generating from "
            f"{len(self._batches)} batches "
            f"({self.n_sampling_steps} sampling steps)..."
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

        return originals, generateds, y_values

    @staticmethod
    def _build_x_ak(originals, generateds, y_values):
        """Concatenate per-batch lists into the x_ak dict and y array."""
        if not originals:
            return None, None
        x_ak = {
            "original": ak.concatenate(originals, axis=0),
            "generated": ak.concatenate(generateds, axis=0),
        }
        y = ak.concatenate(y_values, axis=0)
        return x_ak, y

    def _gather_across_ranks(self, originals, generateds, y_values, rank, world_size):
        """Merge particle-level results from all ranks via temporary parquet files.

        Every rank writes its ``x_ak`` and ``y`` to a parquet file under
        ``self.output_path``.  A barrier ensures all ranks have finished
        writing before rank 0 reads and merges them.
        """
        x_ak_local, y_local = self._build_x_ak(originals, generateds, y_values)

        rank_file = self.output_path / f"rank_{rank}.parquet"
        if x_ak_local is not None:
            ak.to_parquet(ak.Array({"x_ak": x_ak_local, "y": y_local}), rank_file)
        else:
            ak.to_parquet(ak.Array({"empty": [True]}), rank_file)

        # Wait for all ranks to finish writing
        dist.barrier()

        if rank != 0:
            return None, None

        # -- Rank 0: load and merge all partial results --
        all_originals = []
        all_generateds = []
        all_y = []

        for r in range(world_size):
            fpath = self.output_path / f"rank_{r}.parquet"
            partial = ak.from_parquet(fpath)
            if "empty" not in partial.fields:
                all_originals.append(partial["x_ak"]["original"])
                all_generateds.append(partial["x_ak"]["generated"])
                all_y.append(partial["y"])
            fpath.unlink()

        if not all_originals:
            return None, None

        x_ak = {
            "original": ak.concatenate(all_originals, axis=0),
            "generated": ak.concatenate(all_generateds, axis=0),
        }
        return x_ak, ak.concatenate(all_y, axis=0)

    @staticmethod
    def _postprocess(x_ak, y):
        """Compute p4 vectors and jet substructure from the merged x_ak dict."""
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

        return ak.Array(
            {
                "x_ak": x_ak,
                "p4s": p4s,
                "p4s_sum": p4s_sum,
                "substructure": substructure,
                "y": y,
            }
        )
