"""Multi-GPU jet generation script using PyTorch Lightning.

Uses trainer.predict() with a minimal callback that reuses existing generation utilities.
"""

import logging
import time
from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import pytorch_lightning as pl
import torch
import torch.distributed as dist

from omnilearn_lightning.array_utils import p4s_from_ptetaphimass
from omnilearn_lightning.dataloader import PETDataModule
from omnilearn_lightning.generation_utils import (
    JetSubstructure,
    get_batch_and_generate,
    save_gen_output,
)
from omnilearn_lightning.model import PETLightning

logger = logging.getLogger(__name__)

FEATURE_NAMES_X = {
    "eta": "Particle $\\Delta\\eta$",
    "phi": "Particle $\\Delta\\phi$",
    "log_pt": "Particle $\\log(p_{T})$",
    "log_E": "Particle $\\log(E)$",
}


class JetGenerationCallback(pl.Callback):
    """Callback for multi-GPU jet generation with result merging on rank 0."""

    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.gen_output = None

    def on_test_epoch_end(self, trainer, pl_module):
        rank, world_size = trainer.global_rank, trainer.world_size

        # Collect all batch results from this rank
        originals, generateds, y_values, cond_values = [], [], [], []
        for orig, gen, y, cond in pl_module.results:
            originals.append(orig)
            generateds.append(gen)
            y_values.append(y)
            cond_values.append(cond)

        # Write this rank's results to temp file
        if originals:
            x_ak = {
                "original": ak.concatenate(originals, axis=0),
                "generated": ak.concatenate(generateds, axis=0),
            }
            y = ak.concatenate(y_values, axis=0)
            cond = ak.concatenate(cond_values, axis=0)
            rank_file = self.output_path / f"_rank_{rank}.parquet"
            ak.to_parquet(ak.Array({"x_ak": x_ak, "y": y, "cond": cond}), rank_file)
            logger.info("[Rank %d] Wrote %d jets to %s", rank, len(y), rank_file)

        if world_size > 1:
            dist.barrier()

        if rank != 0:
            return

        # Rank 0: merge all results
        all_orig, all_gen, all_y, all_cond = [], [], [], []
        for r in range(world_size):
            fpath = self.output_path / f"_rank_{r}.parquet"
            if fpath.exists():
                partial = ak.from_parquet(fpath)
                all_orig.append(partial["x_ak"]["original"])
                all_gen.append(partial["x_ak"]["generated"])
                all_y.append(partial["y"])
                all_cond.append(partial["cond"])
                fpath.unlink()
                # print average y to verify each rank has different jets
                logger.info(
                    "[Rank %d] Loaded %d jets from %s, avg y=%.4f",
                    rank,
                    len(partial["y"]),
                    fpath,
                    ak.mean(partial["y"]),
                )

        if not all_orig:
            logger.warning("No jets generated.")
            return

        x_ak = {
            "original": ak.concatenate(all_orig, axis=0),
            "generated": ak.concatenate(all_gen, axis=0),
        }
        y = ak.concatenate(all_y, axis=0)
        cond = ak.concatenate(all_cond, axis=0)

        # Postprocess: filter, compute p4s and substructure
        mask = ak.num(x_ak["original"].eta) >= 3
        x_ak = {k: v[mask] for k, v in x_ak.items()}
        y = y[mask]
        cond = cond[mask]

        p4_kwargs = dict(
            field_name_eta="eta",
            field_name_phi="phi",
            field_name_pt="log_pt",
            pt_is_log=True,
        )
        p4s = {k: p4s_from_ptetaphimass(v, **p4_kwargs) for k, v in x_ak.items()}

        logger.info("Computing jet substructure...")
        self.gen_output = ak.Array(
            {
                "x_ak": x_ak,
                "p4s": p4s,
                "p4s_sum": {k: ak.sum(v, axis=1) for k, v in p4s.items()},
                "substructure": {
                    k: JetSubstructure(v).get_substructure_as_ak_array()
                    for k, v in p4s.items()
                },
                "y": y,
                "cond": cond,
            }
        )
        logger.info("Generated %d jets total from %d GPU(s)", len(y), world_size)


def generate_jets(
    ckpt_path: str,
    output_filename: str,
    dataset_type: str = "jetnet150",
    n_jets_to_generate: int = 1000,
    batch_size: int = 200,
    n_sampling_steps: int = 128,
    num_nodes: int = 1,
):
    num_gpus = torch.cuda.device_count()
    logger.info("Detected %d GPU(s) on %d node(s)", num_gpus, num_nodes)
    logger.info("Loading model from checkpoint: %s", ckpt_path)
    lightning_model = PETLightning.load_from_checkpoint(ckpt_path, fine_tune=False)

    output_path = Path(output_filename).parent
    output_path.mkdir(parents=True, exist_ok=True)

    datamodule = PETDataModule(
        dataset=dataset_type,
        path="/pscratch/sd/j/jobirk/omnilearn_datasets_dev/",
        batch_size=batch_size,
        num_samples=n_jets_to_generate,
        use_pid=dataset_type == "jetclass",
        use_add=dataset_type == "jetclass",
        use_cond=True,
        shuffle_val_test_indices=True,
        seed_for_initial_shuffling=42,
        load_val=True,
    )

    # Use the loaded model directly - just need to wrap test_step
    class GenModule(pl.LightningModule):
        def __init__(self, model, n_batches_per_gpu: int):
            super().__init__()
            self._model = model
            self.results = []
            self.n_batches_per_gpu = n_batches_per_gpu
            self._start_time = None
            self._jets_generated = 0

        def test_step(self, batch, batch_idx):
            this_rank = dist.get_rank() if dist.is_initialized() else 0
            if self._start_time is None:
                self._start_time = time.time()

            result = get_batch_and_generate(
                batch=batch,
                lightning_model=self._model,
                feature_names_x=FEATURE_NAMES_X,
                n_sampling_steps=n_sampling_steps,
                verbose=False,
            ) + (batch["y"].cpu().numpy(), batch["cond"].cpu().numpy())
            self.results.append(result)

            # --- Progress logging ---
            n_jets_this_batch = len(batch["y"])
            self._jets_generated += n_jets_this_batch
            elapsed = time.time() - self._start_time
            jets_per_sec = self._jets_generated / elapsed if elapsed > 0 else 0
            jets_per_hour = jets_per_sec * 3600

            # Total jets across all devices (GPUs * nodes)
            total_devices = max(1, num_gpus) * num_nodes
            total_jets_so_far = self._jets_generated * total_devices
            remaining_jets = max(0, n_jets_to_generate - total_jets_so_far)
            eta_seconds = (
                remaining_jets / (jets_per_sec * total_devices)
                if jets_per_sec > 0
                else float("inf")
            )

            def _fmt_time(seconds):
                if seconds == float("inf"):
                    return "unknown"
                h, remainder = divmod(int(seconds), 3600)
                m, s = divmod(remainder, 60)
                if h > 0:
                    return f"{h}h {m:02d}m {s:02d}s"
                return f"{m}m {s:02d}s"

            logger.info(
                "[Rank %d] Batch %d/%d | %d/%d jets generated | %.0f jets/hour (this GPU) | Elapsed: %s | ETA: %s",
                this_rank,
                batch_idx + 1,
                self.n_batches_per_gpu,
                total_jets_so_far,
                n_jets_to_generate,
                jets_per_hour,
                _fmt_time(elapsed),
                _fmt_time(eta_seconds),
            )

            return result

    # Calculate number of batches needed (each device processes batches in parallel)
    total_devices = max(1, num_gpus) * num_nodes
    effective_batch_size = batch_size * total_devices
    num_batches = (
        n_jets_to_generate + effective_batch_size - 1
    ) // effective_batch_size

    logger.info(
        "Generating %d jets with effective batch size %d across %d device(s) (%d GPU(s) x %d node(s)) => %d batches per device",
        n_jets_to_generate,
        effective_batch_size,
        total_devices,
        num_gpus,
        num_nodes,
        num_batches,
    )

    use_ddp = total_devices > 1
    writer = JetGenerationCallback(output_path)
    trainer = pl.Trainer(
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else 1,
        num_nodes=num_nodes,
        strategy="ddp" if use_ddp else "auto",
        callbacks=[writer],
        enable_progress_bar=False,
        logger=False,
        limit_test_batches=num_batches,
    )

    gen_module = GenModule(lightning_model, n_batches_per_gpu=num_batches)
    trainer.test(gen_module, datamodule=datamodule)

    if trainer.global_rank == 0 and writer.gen_output is not None:
        save_gen_output(writer.gen_output, output_filename)

    return writer.gen_output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = ArgumentParser(description="Multi-GPU jet generation")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="jetnet150")
    parser.add_argument("--n_jets_to_generate", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_sampling_steps", type=int, default=128)
    parser.add_argument("--num_nodes", type=int, default=1)

    args = parser.parse_args()
    generate_jets(**vars(args))
