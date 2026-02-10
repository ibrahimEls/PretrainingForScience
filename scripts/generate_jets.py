from argparse import ArgumentParser
from pathlib import Path

import awkward as ak

from omnilearn_lightning.dataloader import PETDataModule
from omnilearn_lightning.generation_utils import (
    generate_and_postprocess,
)
from omnilearn_lightning.model import PETLightning


def eval_jet_generation(
    ckpt_path: str,
    output_filename: str,
    dataset_type: str = "jetnet150",
    n_jets_to_generate: int = 1000,
    batch_size: int = 200,
):
    print(f"Loading model from checkpoint: {ckpt_path}")
    lightning_model = PETLightning.load_from_checkpoint(ckpt_path)
    print(lightning_model.model.generator)

    output_filename = Path(output_filename)
    # ensure output directory exists
    output_path = output_filename.parent
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Generated output will be saved to: {output_filename}")

    datamodule = PETDataModule(
        dataset=dataset_type,
        path="/pscratch/sd/j/jobirk/omnilearn_datasets_dev/",
        batch_size=batch_size,
        num_samples=n_jets_to_generate,
        use_pid=True if dataset_type == "jetclass" else False,
        use_add=True if dataset_type == "jetclass" else False,
        use_cond=True,
        shuffle_val_test_indices=True,
        seed_for_initial_shuffling=42,
        load_val=True,
    )

    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    n_batches = (n_jets_to_generate + batch_size - 1) // batch_size

    feature_names_x = {  # <-- will be used to convert to ak arrays with names
        "eta": "Particle $\\Delta\\eta$",
        "phi": "Particle $\\Delta\\phi$",
        "log_pt": "Particle $\\log(p_{T})$",
        "log_E": "Particle $\\log(E)$",
    }

    gen_output = generate_and_postprocess(
        n_batches=n_batches,
        dataloader=dataloader,
        lightning_model=lightning_model,
        feature_names_x=feature_names_x,
        verbose=False,
    )
    print(f"Generated {len(gen_output)} jets")  # <-- gen_output is an ak array of jets

    # save to parquet
    ak.to_parquet(gen_output, output_filename)
    print(f"Saved generated jets to {output_filename}")

    return None


parser = ArgumentParser(description="Evaluate jet generation from a checkpoint")
parser.add_argument(
    "--ckpt_path",
    type=str,
    required=True,
    help="Path to the checkpoint to evaluate",
)
parser.add_argument(
    "--output_filename",
    type=str,
    required=True,
    help="Filename (including path) to save the generated output (in parquet format)",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    default="jetnet150",
    help="Dataset type to use for evaluation (default: jetnet150)",
)
parser.add_argument(
    "--n_jets_to_generate",
    type=int,
    default=1000,
    help="Number of jets to generate for evaluation (default: 1000)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=200,
    help="Batch size to use for generation (default: 200)",
)


if __name__ == "__main__":
    args = parser.parse_args()
    eval_jet_generation(
        ckpt_path=args.ckpt_path,
        output_filename=args.output_filename,
        dataset_type=args.dataset_type,
        n_jets_to_generate=args.n_jets_to_generate,
        batch_size=args.batch_size,
    )
