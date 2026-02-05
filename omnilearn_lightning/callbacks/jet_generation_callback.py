"""Callback for generating jets during validation/test and returning a comparison ak array.

Collects batches during the validation/test loop, runs diffusion-based generation
at epoch end, and produces the same output structure as
``generation_utils.generate_and_postprocess``.

Supports multi-GPU (DDP): every rank generates independently, then rank 0
merges all partial results via temporary parquet files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm

from omnilearn_lightning.array_utils import p4s_from_ptetaphimass
from omnilearn_lightning.generation_utils import (
    JetSubstructure,
    calc_metrics_for_dict,
    get_batch_and_generate,
    save_gen_output,
)
from omnilearn_lightning.plotting.feature_plotting import (
    plot_features,
    set_mpl_style,
)

# plot settings
bins_dict_substructure = {
    "tau21": np.linspace(0, 1.2, 50),
    "tau32": np.linspace(0, 1.2, 50),
    "jet_mass": np.linspace(0, 250, 50),
    "jet_pt": np.linspace(400, 1600, 50),
    "jet_n_constituents": np.linspace(-0.5, 101.5, 52),
    "d2": np.linspace(0, 5, 50),
}
names_substructure = {
    "jet_pt": "Jet $p_{T}$ [GeV]",
    "jet_mass": "Jet mass [GeV]",
    "jet_n_constituents": "Particle multiplicity",
    "tau21": "$\\tau_{21}$",
    "tau32": "$\\tau_{32}$",
    "d2": "$D_{2}$",
}
bins_dict = {
    "eta": np.linspace(-1, 1, 50),
    "phi": np.linspace(-1, 1, 50),
    "log_pt": np.linspace(-1.5, 6.5, 50),
    "log_E": np.linspace(-1.5, 6.5, 50),
}
names_dict = {
    "eta": "Particle $\\Delta\\eta$",
    "phi": "Particle $\\Delta\\phi$",
    "log_pt": "Particle $\\log(p_{T})$",
    "log_E": "Particle $\\log(E)$",
}

jet_types_dict_jetclass = {
    0: {"label": "QCD", "tex_label": "$q/g$", "color": "C0"},
    1: {
        "label": "Hbb",
        "tex_label": "$H\\rightarrow b\\bar{b}$",
        "color": "C1",
    },
    2: {
        "label": "Hcc",
        "tex_label": "$H\\rightarrow c\\bar{c}$",
        "color": "C2",
    },
    3: {
        "label": "Hgg",
        "tex_label": "$H\\rightarrow gg$",
        "color": "C3",
    },
    4: {
        "label": "H4q",
        "tex_label": "$H\\rightarrow 4q$",
        "color": "C4",
    },
    5: {
        "label": "Hlnuqq",
        "tex_label": "$H\\rightarrow \\ell\\nu qq'$",
        "color": "C5",
    },
    6: {
        "label": "Zqq",
        "tex_label": "$Z\\rightarrow q\\bar{q}$",
        "color": "C6",
    },
    7: {
        "label": "Wqq",
        "tex_label": "$W\\rightarrow qq'$",
        "color": "C7",
    },
    8: {
        "label": "Tbqq",
        "tex_label": "$t\\rightarrow bqq'$",
        "color": "C8",
    },
    9: {
        "label": "Tbl",
        "tex_label": "$t\\rightarrow b\\ell\\nu$",
        "color": "C9",
    },
}

# Feature names for JetNet (same kinematic features, no PID/add info)
feature_names_x = {
    "eta": "Particle $\\Delta\\eta$",
    "phi": "Particle $\\Delta\\phi$",
    "log_pt": "Particle $\\log(p_{T})$",
    "log_E": "Particle $\\log(E)$",
}

# JetNet has 5 jet types
jet_types_dict_jetnet = {
    0: {"label": "gluon", "tex_label": "$g$", "color": "C0"},
    1: {"label": "light_quark", "tex_label": "$q$", "color": "C1"},
    2: {"label": "top", "tex_label": "$t$", "color": "C2"},
    3: {"label": "W", "tex_label": "$W$", "color": "C3"},
    4: {"label": "Z", "tex_label": "$Z$", "color": "C4"},
}


class JetGenerationCallback(Callback):
    """Generate jets via diffusion during validation/test and build a comparison ak array.

    At the end of each validation/test epoch the callback stores its result in
    ``self.gen_output[phase]`` (rank 0 only) – an ``ak.Array`` with the same
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
        Directory to save the final parquet files and per-rank temporary files.
        Created automatically if it does not exist.
    n_batches : int
        Number of batches **per rank** to collect and generate from.
    n_sampling_steps : int
        Number of diffusion sampling steps.
    set_pid_to_zeros : bool
        Zero-out PID features before generation.
    set_add_info_to_zeros : bool
        Zero-out additional-info features before generation.
    every_n_epochs : int
        Run generation every n epochs. Default is 1 (every epoch).
    verbose : bool
        Forward to ``get_batch_and_generate``.
    """

    def __init__(
        self,
        output_path: str | Path,
        dataset_type: str,
        n_batches: int = -1,
        n_sampling_steps: int = 128,
        set_pid_to_zeros: bool = False,
        set_add_info_to_zeros: bool = False,
        every_n_epochs: int = 1,
        verbose: bool = False,
    ):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.n_batches = n_batches
        self.n_sampling_steps = n_sampling_steps
        self.set_pid_to_zeros = set_pid_to_zeros
        self.set_add_info_to_zeros = set_add_info_to_zeros
        self.every_n_epochs = every_n_epochs
        self.verbose = verbose

        supported_datasets = ["jetnet150", "jetnet30", "jetclass"]
        if dataset_type not in supported_datasets:
            raise ValueError(
                f"{dataset_type} not supported. Supported datasets: {supported_datasets}"
            )
        self.dataset_type = dataset_type

        if self.dataset_type == "jetclass":
            self.jet_types = range(10)
            self.jet_types_dict = jet_types_dict_jetclass
        elif "jetnet" in self.dataset_type:
            self.jet_types = range(5)
            self.jet_types_dict = jet_types_dict_jetnet
        else:
            self.jet_types = None
            self.jet_types_dict = None

        # Populated at epoch end (rank 0 only), keyed by phase
        self.gen_output: Dict[str, Optional[ak.Array]] = {
            "test": None,
            "validation": None,
        }

        # Per-phase batch storage, initialised at each epoch start
        self._batches: Dict[str, list] = {}

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

    @staticmethod
    def _get_wandb_logger(pl_module: LightningModule):
        """Return the WandbLogger if it exists, else None."""
        from pytorch_lightning.loggers import WandbLogger

        if pl_module.logger is None:
            return None
        if isinstance(pl_module.logger, WandbLogger):
            return pl_module.logger
        # Handle LoggerCollection (multiple loggers)
        if hasattr(pl_module.logger, "experiment"):
            # Check if it's a list of loggers
            loggers = getattr(pl_module, "loggers", [pl_module.logger])
            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    return logger
        return None

    # ------------------------------------------------------------------ #
    #  Lightning hooks – thin wrappers that delegate to shared methods     #
    # ------------------------------------------------------------------ #

    def on_test_epoch_start(self, trainer, pl_module):
        self._on_epoch_start("test")

    def on_validation_epoch_start(self, trainer, pl_module):
        self._on_epoch_start("validation")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._on_batch_end("test", batch)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._on_batch_end("validation", batch)

    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end("test", pl_module, trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end("validation", pl_module, trainer)

    # ------------------------------------------------------------------ #
    #  Shared hook logic                                                   #
    # ------------------------------------------------------------------ #

    def _on_epoch_start(self, phase: str) -> None:
        self._batches[phase] = []

    def _on_batch_end(self, phase: str, batch: Any) -> None:
        if self.n_batches != -1 and len(self._batches[phase]) >= self.n_batches:
            return

        saved: Dict[str, Any] = {}
        for key in ("X", "y", "cond", "pid", "add_info"):
            if key in batch and batch[key] is not None:
                val = batch[key]
                saved[key] = val.detach().cpu() if torch.is_tensor(val) else val
        self._batches[phase].append(saved)

    def _on_epoch_end(
        self, phase: str, pl_module: LightningModule, trainer=None
    ) -> None:
        epoch = trainer.current_epoch if trainer is not None else 0

        # Skip if not on the right epoch interval
        if epoch % self.every_n_epochs != 0:
            self._batches[phase] = []  # Clear batches to free memory
            return

        rank = self._rank()
        world_size = self._world_size()
        batches = self._batches.get(phase, [])
        step = trainer.global_step if trainer is not None else 0

        # -- Phase 1: every rank generates from its own collected batches --
        if not batches:
            originals, generateds, y_values = [], [], []
        else:
            originals, generateds, y_values = self._generate_from_batches(
                batches, pl_module
            )

        # -- Phase 2: merge across ranks --
        if world_size > 1:
            x_ak_all, y_all = self._gather_across_ranks(
                originals, generateds, y_values, rank, world_size, phase
            )
        else:
            x_ak_all, y_all = self._build_x_ak(originals, generateds, y_values)

        # -- Phase 3: rank 0 computes substructure and assembles final output --
        # Clear batch storage to free memory (all ranks)
        self._batches[phase] = []

        if rank != 0:
            return

        if x_ak_all is None:
            print(
                f"JetGenerationCallback [{phase}]: "
                "no jets collected across ranks, skipping."
            )
            return

        self.gen_output[phase] = self._postprocess(x_ak_all, y_all)

        print(
            f"JetGenerationCallback [{phase}]: done – "
            f"{len(self.gen_output[phase]['y'])} jets generated "
            f"(from {world_size} rank(s))."
        )

        save_gen_output(
            self.gen_output[phase],
            self.output_path
            / f"gen_output_{phase}_epoch{epoch:03d}_step{step:06d}.parquet",
        )

        self._plot(self.gen_output[phase], phase, pl_module, epoch=epoch, step=step)

    # ------------------------------------------------------------------ #
    #  Internal methods                                                    #
    # ------------------------------------------------------------------ #

    def _generate_from_batches(self, batches, pl_module):
        """Run diffusion generation on each collected batch. Returns lists."""
        rank = self._rank()
        print(
            f"JetGenerationCallback [rank {rank}]: generating from "
            f"{len(batches)} batches "
            f"({self.n_sampling_steps} sampling steps)..."
        )

        originals = []
        generateds = []
        y_values = []

        for batch in tqdm(batches):
            x_orig, x_gen = get_batch_and_generate(
                batch=batch,
                lightning_model=pl_module,
                feature_names_x=feature_names_x,
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

    def _gather_across_ranks(
        self, originals, generateds, y_values, rank, world_size, phase
    ):
        """Merge particle-level results from all ranks via temporary parquet files.

        Every rank writes its ``x_ak`` and ``y`` to a parquet file under
        ``self.output_path``.  A barrier ensures all ranks have finished
        writing before rank 0 reads and merges them.
        """
        x_ak_local, y_local = self._build_x_ak(originals, generateds, y_values)

        rank_file = self.output_path / f"{phase}_rank_{rank}.parquet"
        if x_ak_local is not None:
            ak.to_parquet(ak.Array({"x_ak": x_ak_local, "y": y_local}), rank_file)
        else:
            ak.to_parquet(ak.Array({"empty": [True]}), rank_file)

        print(f"JetGenerationCallback [rank {rank}]: wrote temporary file {rank_file}")

        # Wait for all ranks to finish writing
        dist.barrier()

        if rank != 0:
            return None, None

        print(
            f"JetGenerationCallback [rank {rank}]: combining results from all ranks..."
        )

        # -- Rank 0: load and merge all partial results --
        all_originals = []
        all_generateds = []
        all_y = []

        for r in range(world_size):
            fpath = self.output_path / f"{phase}_rank_{r}.parquet"
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

        # filter for jets with at least 3 constituents, otherwise substructure
        # calculation can fail
        mask_more_than_3_constituents = ak.num(x_ak["original"].eta) >= 3
        x_ak["original"] = x_ak["original"][mask_more_than_3_constituents]
        x_ak["generated"] = x_ak["generated"][mask_more_than_3_constituents]
        y = y[mask_more_than_3_constituents]

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

    def _plot(
        self,
        gen_output,
        stage: str,
        pl_module: LightningModule,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        """Plot jet substructure comparison between original and generated jets."""

        if self._rank() != 0:
            return

        set_mpl_style()

        # loop over all jet types, and produce those plots for each jet type individually

        for jet_type_i in self.jet_types:
            mask_this_type = gen_output.y == jet_type_i

            plot_kwargs_common = dict(
                fig_legend_kwargs=dict(
                    bbox_to_anchor=(0.5, 1.10),
                    loc="upper center",
                    fontsize=10,
                    title=f"{self.jet_types_dict[jet_type_i]['tex_label']} jets",
                    ncol=2,
                ),
                ratio=True,
                ax_rows=2,
                decorate_ax_kwargs=dict(
                    yscale=1.05,
                ),
            )

            # jet-level features
            fig, axarr = plot_features(
                {
                    "Target": gen_output[mask_this_type].substructure.original,
                    "Generated": gen_output[mask_this_type].substructure.generated,
                },
                names=names_substructure,
                bins_dict=bins_dict_substructure,
                flatten=False,
                **plot_kwargs_common,
            )

            outfile_suffix = (
                f"_{stage}_epoch{epoch:03d}_step{step:06d}_jettype{jet_type_i}"
            )
            jet_type_label = self.jet_types_dict[jet_type_i]["label"]

            outfile = (
                self.output_path / f"jet_substructure_comparison{outfile_suffix}.png"
            )
            fig.savefig(outfile, bbox_inches="tight", dpi=150)
            print(f"Saved figure to {outfile}")

            # Log to wandb if available
            wandb_logger = self._get_wandb_logger(pl_module)
            if wandb_logger is not None:
                import wandb

                wandb_logger.experiment.log(
                    {
                        f"gen/{stage}/jet_substructure/{jet_type_label}": wandb.Image(
                            outfile
                        ),
                    },
                    commit=False,
                )
            plt.close(fig)

            # constituent-level features
            fig, axarr = plot_features(
                {
                    "Target": gen_output[mask_this_type].x_ak.original,
                    "Generated": gen_output[mask_this_type].x_ak.generated,
                },
                names=feature_names_x,
                bins_dict=bins_dict,
                flatten=True,
                **plot_kwargs_common,
            )
            outfile = (
                self.output_path / f"constituent_feature_comparison{outfile_suffix}.png"
            )
            fig.savefig(outfile, bbox_inches="tight", dpi=150)
            print(f"Saved figure to {outfile}")

            # Log to wandb if available
            if wandb_logger is not None:
                import wandb

                wandb_logger.experiment.log(
                    {
                        f"gen/{stage}/constituent_features/{jet_type_label}": wandb.Image(
                            outfile
                        ),
                    },
                    commit=False,
                )
            plt.close(fig)

            metrics_calc_kwargs = dict(
                return_zero_if_nan_or_inf=True,
                num_eval_samples=10_000,
                num_batches=10,
            )

            # Calculate and log metrics for jet-level features
            jet_level_metrics = calc_metrics_for_dict(
                dict_reference=gen_output[mask_this_type].substructure.original,
                dict_approx=gen_output[mask_this_type].substructure.generated,
                names=list(names_substructure.keys()),
                **metrics_calc_kwargs,
            )
            print(f"Jet-level metrics for {jet_type_label}:")
            for metric_name, metric_values in jet_level_metrics.items():
                for feature_name, (mean, std) in metric_values.items():
                    print(f"  {metric_name}_{feature_name}: {mean:.4f} ± {std:.4f}")
                    pl_module.log(
                        f"gen/{stage}/jet/{jet_type_label}/{metric_name}_{feature_name}",
                        mean,
                        # sync_dist=True,
                    )

            # Calculate and log metrics for particle-level features
            particle_level_metrics = calc_metrics_for_dict(
                dict_reference=gen_output[mask_this_type].x_ak.original,
                dict_approx=gen_output[mask_this_type].x_ak.generated,
                names=list(feature_names_x.keys()),
                **metrics_calc_kwargs,
            )
            print(f"Particle-level metrics for {jet_type_label}:")
            for metric_name, metric_values in particle_level_metrics.items():
                for feature_name, (mean, std) in metric_values.items():
                    print(f"  {metric_name}_{feature_name}: {mean:.4f} ± {std:.4f}")
                    pl_module.log(
                        f"gen/{stage}/particle/{jet_type_label}/{metric_name}_{feature_name}",
                        mean,
                        # sync_dist=True,
                    )
