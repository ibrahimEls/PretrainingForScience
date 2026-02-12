import os
from pathlib import Path

import awkward as ak
import omnilearned
import pandas as pd
import torch
import vector
from tqdm import tqdm

from omnilearn_lightning.array_utils import np_to_ak, p4s_from_ptetaphimass
from omnilearn_lightning.jet_substructure import JetSubstructure

vector.register_awkward()


def get_batch_and_generate(
    batch,
    lightning_model=None,
    feature_names_x=None,
    verbose=True,
    n_sampling_steps=128,
    set_pid_to_zeros=False,
    set_add_info_to_zeros=False,
):
    X, y = (
        batch["X"].float().to(lightning_model.device),
        batch["y"].to(lightning_model.device),
    )
    model_kwargs = {
        k: (batch[k].to(lightning_model.device) if batch[k] is not None else None)
        for k in ("cond", "pid", "add_info")
        if (k in batch)
    }
    if verbose:
        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")
        for k, v in model_kwargs.items():
            if v is not None:
                print(f"{k}.shape: {v.shape}")
    mask = X[:, :, 0] != 0.0

    # if "cond" is not in batch, then create the multiplicity entry
    if model_kwargs.get("cond", None) is None:
        multiplicity = torch.sum(mask, dim=1).float() / 100.0
        model_kwargs["cond"] = multiplicity.unsqueeze(-1)

    # set pid and add_info to zeros
    if "pid" in model_kwargs and set_pid_to_zeros:
        # print(f"Setting pid to zeros with shape: {model_kwargs['pid'].shape}")
        model_kwargs["pid"] = torch.zeros_like(model_kwargs["pid"])
    if "add_info" in model_kwargs and set_add_info_to_zeros:
        # print(f"Setting add_info to zeros with shape: {model_kwargs['add_info'].shape}")
        model_kwargs["add_info"] = torch.zeros_like(model_kwargs["add_info"])

    with torch.no_grad():
        gen_output = omnilearned.diffusion.generate(
            lightning_model.model,
            y=y,
            shape=X.shape,
            nsteps=n_sampling_steps,
            **model_kwargs,
        )

    x_ak_original = np_to_ak(
        X.cpu().numpy(), names=feature_names_x.keys(), mask=mask.cpu().numpy()
    )
    x_ak_generated = np_to_ak(
        gen_output.cpu().numpy(), names=feature_names_x.keys(), mask=mask.cpu().numpy()
    )

    return x_ak_original, x_ak_generated


def generate_and_postprocess(
    n_batches,
    dataloader,
    **gen_kwargs,
):
    print(f"Generating {n_batches} batches...")
    n_batches_generated = 0
    original = []
    generated = []
    y_values = []

    for batch_i in tqdm(dataloader, total=n_batches):
        if n_batches_generated >= n_batches:
            break
        n_batches_generated += 1
        output = get_batch_and_generate(
            batch=batch_i,
            **gen_kwargs,
        )
        original.append(output[0])
        generated.append(output[1])
        y_values.append(batch_i["y"].cpu().numpy())

    # concatenate the lists into a single ak.Array per n_steps
    # n_steps=0 corresponds to the original data
    x_ak_dict = {
        "original": None,
        "generated": None,
    }
    x_ak_dict["original"] = ak.concatenate(original, axis=0)
    x_ak_dict["generated"] = ak.concatenate(generated, axis=0)

    # convert to p4s, based on pt, eta, phi, mass=0
    p4_conversion_kwargs = dict(
        field_name_eta="eta",
        field_name_phi="phi",
        field_name_pt="log_pt",
        pt_is_log=True,
    )
    p4s_dict = {
        "original": p4s_from_ptetaphimass(
            x_ak_dict["original"], **p4_conversion_kwargs
        ),
        "generated": p4s_from_ptetaphimass(
            x_ak_dict["generated"], **p4_conversion_kwargs
        ),
    }
    # calculate sum of constituent p4s = jet p4s
    p4s_sum_dict = {
        "original": ak.sum(p4s_dict["original"], axis=1),
        "generated": ak.sum(p4s_dict["generated"], axis=1),
    }
    # calculate jet substructure (subjettiness ratios etc)
    substructure_calculator_original = JetSubstructure(p4s_dict["original"])
    substructure_calculator_generated = JetSubstructure(p4s_dict["generated"])

    substructure_dict = {
        "original": (substructure_calculator_original.get_substructure_as_ak_array()),
        "generated": (substructure_calculator_generated.get_substructure_as_ak_array()),
    }

    return ak.Array(
        {
            "x_ak": x_ak_dict,
            "p4s": p4s_dict,
            "p4s_sum": p4s_sum_dict,
            "substructure": substructure_dict,
            "y": ak.concatenate(y_values, axis=0),
        }
    )


def save_gen_output(gen_output: ak.Array, output_path: str | Path) -> None:
    """Save generated jet output to parquet file.

    Parameters
    ----------
    gen_output : ak.Array
        The output from generate_and_postprocess containing x_ak, substructure, y, etc.
    output_path : str | Path
        Path to save the parquet file (should end with .parquet)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(gen_output, output_path)
    print(f"Saved generated jets to {output_path}")


def load_gen_output(output_path: str | Path) -> ak.Array:
    """Load generated jet output from parquet file.

    Parameters
    ----------
    output_path : str | Path
        Path to the parquet file

    Returns
    -------
    ak.Array
        The loaded generation output
    """
    output_path = Path(output_path)
    gen_output = ak.from_parquet(output_path)
    print(f"Loaded generated jets from {output_path}")
    return gen_output


def generate_or_load(
    output_path: str | Path,
    lightning_model,
    gen_kwargs: dict,
    set_pid_to_zeros: bool = False,
    set_add_info_to_zeros: bool = False,
    force_regenerate: bool = False,
) -> ak.Array:
    """Generate jets or load from cache if already exists.

    Parameters
    ----------
    output_path : str | Path
        Path to save/load the parquet file
    lightning_model : PETLightning
        The model to use for generation (only used if file doesn't exist)
    gen_kwargs : dict
        Keyword arguments for generate_and_postprocess
    set_pid_to_zeros : bool
        Whether to zero out PID features
    set_add_info_to_zeros : bool
        Whether to zero out additional info features
    force_regenerate : bool
        If True, regenerate even if file exists

    Returns
    -------
    ak.Array
        The generation output (either loaded or freshly generated)
    """
    output_path = Path(output_path)

    if output_path.exists() and not force_regenerate:
        return load_gen_output(output_path)

    print(f"Generating jets (will save to {output_path})...")
    gen_output = generate_and_postprocess(
        lightning_model=lightning_model,
        **gen_kwargs,
        set_pid_to_zeros=set_pid_to_zeros,
        set_add_info_to_zeros=set_add_info_to_zeros,
    )
    save_gen_output(gen_output, output_path)
    return gen_output


def get_metrics_and_gen_jets_vs_epoch(
    output_path_gen_vs_epoch: str, n_jets: int, skip_epoch_zero: bool = False
):
    """
    Parameters
    ----------
    output_path_gen_vs_epoch : str
        The output path to the `generation_output` folder, which stores the generated
        output and metrics of the different epochs in the style
        generation_output
        ```
        |- epoch_1
        |---- generated_output_{n_jets}.parquet
        |---- generated_output_{n_jets}_plots
        |------- metrics.csv
        |- epoch_50
        |---- generated_output_{n_jets}.parquet
        |---- generated_output_{n_jets}_plots
        |------- metrics.csv
        ...
        ```
    """
    output_path_gen_vs_epoch = Path(output_path_gen_vs_epoch)

    df_list = []

    # loop over folders in there, and check if it contains "epoch_"
    for folder in os.listdir(output_path_gen_vs_epoch):
        if "epoch_" in folder:
            if skip_epoch_zero and folder == "epoch_0":
                continue
            try:
                # load the generated output
                gen_output = ak.from_parquet(
                    output_path_gen_vs_epoch
                    / folder
                    / f"generated_jets_{n_jets}.parquet"
                )
                # descend into folder epoch_<>/generated_jets_{n_jets}_plots
                gen_jets_plots_folder = (
                    output_path_gen_vs_epoch / folder / f"generated_jets_{n_jets}_plots"
                )
                # print(gen_jets_plots_folder)
                # check if metrics.csv exists in there
                metrics_csv_path = gen_jets_plots_folder / "metrics.csv"
                if metrics_csv_path.exists():
                    print(f"Found metrics.csv in {gen_jets_plots_folder}")
                    # read it and print the contents
                    metrics_df = pd.read_csv(metrics_csv_path)
                    # print(metrics_df)
                    df_list.append(
                        {
                            "epoch": int(folder.split("epoch_")[1]),
                            "metrics": metrics_df,
                            "gen_output": gen_output,
                        }
                    )
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")

    for df in df_list:
        print(f"Epoch: {df['epoch']}")
        print(df["metrics"])

    # sort the list of dicts by epoch
    df_list = sorted(df_list, key=lambda x: x["epoch"])

    return df_list
