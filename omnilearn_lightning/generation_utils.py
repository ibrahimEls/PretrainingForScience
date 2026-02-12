import os
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import omnilearned
import pandas as pd
import torch
import vector
from celluloid import Camera
from tqdm import tqdm

from omnilearn_lightning.array_utils import np_to_ak, p4s_from_ptetaphimass
from omnilearn_lightning.jet_substructure import JetSubstructure
from omnilearn_lightning.utils_lightweight import (
    get_numerical_label_from_name,
)

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
        |- epoch_1
        |---- generated_output_{n_jets}.parquet
        |---- generated_output_{n_jets}_plots
        |------- metrics.csv
        |- epoch_50
        |---- generated_output_{n_jets}.parquet
        |---- generated_output_{n_jets}_plots
        |------- metrics.csv
        ...
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


def animate_metrics_vs_epoch(
    df_list,
    jet_type_to_plot="top",
    dataset_name="jetnet150",
    output_path=None,
    fps=2,
    dpi=100,
    dataset_type="jetnet",
):
    """
    Create an animated plot showing KLD, W1 distance metrics and histograms evolving over epochs.

    Top panels: KLD points accumulate over epochs (dashed line + current point as disc)
    Middle panels: W1 distance points accumulate over epochs (dashed line + current point as disc)
    Bottom panels: Histograms show only the current epoch (previous ones vanish)

    Parameters
    ----------
    df_list : list of dict
        List of dicts with keys 'epoch', 'metrics', 'gen_output'
    jet_type_to_plot : str
        Jet type to plot (e.g., 'top', 'gluon', etc.)
    dataset_name : str
        Dataset name for label lookup
    output_path : str or Path, optional
        If provided, save the animation as a gif to this path
    fps : int
        Frames per second for the animation
    dpi : int
        Resolution of the output gif
    dataset_type : str
        Type of dataset for label lookup (e.g., 'jetnet', 'jetclass')

    Returns
    -------
    anim : ArtistAnimation
        The animation object
    """
    if dataset_type == "jetclass":
        jet_type_numerical_label = get_numerical_label_from_name(
            jet_type_to_plot, "jetclass"
        )
    elif "jetnet" in dataset_type:
        jet_type_numerical_label = get_numerical_label_from_name(
            jet_type_to_plot, "jetnet"
        )
    else:
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}'. Must be 'jetclass' or 'jetnet'."
        )

    bins_dict = {
        "jet_mass": np.linspace(0, 500, 80),
        "jet_pt": np.linspace(0, 1500, 80),
    }

    def binclip(x, feat_name):
        bins = bins_dict[feat_name]
        return np.clip(x, bins[0], bins[-1])

    # Sort df_list by epoch
    df_list_sorted = sorted(df_list, key=lambda x: x["epoch"])

    # Pre-extract all data
    epochs = []
    kld_jet_mass_values = []
    kld_jet_pt_values = []
    kld_jet_mass_std = []
    kld_jet_pt_std = []
    w1_jet_mass_values = []
    w1_jet_pt_values = []
    w1_jet_mass_std = []
    w1_jet_pt_std = []
    truth_kld_jet_mass_values = []
    truth_kld_jet_pt_values = []
    truth_kld_jet_mass_std = []
    truth_kld_jet_pt_std = []
    truth_w1_jet_mass_values = []
    truth_w1_jet_pt_values = []
    truth_w1_jet_mass_std = []
    truth_w1_jet_pt_std = []
    gen_jet_mass_data = []
    gen_jet_pt_data = []

    for df_i in df_list_sorted:
        epochs.append(df_i["epoch"])
        metrics_df = df_i["metrics"]
        kld_jet_mass_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "kld" and feature == "jet_mass"'
        )
        kld_jet_mass_values.append(kld_jet_mass_row["mean"].values[0])
        kld_jet_mass_std.append(kld_jet_mass_row["std"].values[0])
        kld_jet_pt_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "kld" and feature == "jet_pt"'
        )
        kld_jet_pt_values.append(kld_jet_pt_row["mean"].values[0])
        kld_jet_pt_std.append(kld_jet_pt_row["std"].values[0])
        w1_jet_mass_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "w1" and feature == "jet_mass"'
        )
        w1_jet_mass_values.append(w1_jet_mass_row["mean"].values[0])
        w1_jet_mass_std.append(w1_jet_mass_row["std"].values[0])
        w1_jet_pt_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "w1" and feature == "jet_pt"'
        )
        w1_jet_pt_values.append(w1_jet_pt_row["mean"].values[0])
        w1_jet_pt_std.append(w1_jet_pt_row["std"].values[0])
        truth_kld_jet_mass_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "truth_kld" and feature == "jet_mass"'
        )
        truth_kld_jet_mass_values.append(truth_kld_jet_mass_row["mean"].values[0])
        truth_kld_jet_mass_std.append(truth_kld_jet_mass_row["std"].values[0])
        truth_kld_jet_pt_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "truth_kld" and feature == "jet_pt"'
        )
        truth_kld_jet_pt_values.append(truth_kld_jet_pt_row["mean"].values[0])
        truth_kld_jet_pt_std.append(truth_kld_jet_pt_row["std"].values[0])
        truth_w1_jet_mass_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "truth_w1" and feature == "jet_mass"'
        )
        truth_w1_jet_mass_values.append(truth_w1_jet_mass_row["mean"].values[0])
        truth_w1_jet_mass_std.append(truth_w1_jet_mass_row["std"].values[0])
        truth_w1_jet_pt_row = metrics_df.query(
            f'jet_type == "{jet_type_to_plot}" and level == "jet" and metric == "truth_w1" and feature == "jet_pt"'
        )
        truth_w1_jet_pt_values.append(truth_w1_jet_pt_row["mean"].values[0])
        truth_w1_jet_pt_std.append(truth_w1_jet_pt_row["std"].values[0])

        jet_type_mask = df_i["gen_output"].y == jet_type_numerical_label
        gen_output_this_type = df_i["gen_output"][jet_type_mask]
        gen_jet_mass_data.append(gen_output_this_type.substructure.generated.jet_mass)
        gen_jet_pt_data.append(gen_output_this_type.substructure.generated.jet_pt)

    # Get original data for each epoch (each epoch has different original samples)
    original_jet_mass_data = []
    original_jet_pt_data = []
    for df_i in df_list_sorted:
        jet_type_mask = df_i["gen_output"].y == jet_type_numerical_label
        gen_output_this_type = df_i["gen_output"][jet_type_mask]
        original_jet_mass_data.append(
            gen_output_this_type.substructure.original.jet_mass
        )
        original_jet_pt_data.append(gen_output_this_type.substructure.original.jet_pt)

    # Create figure and camera
    fig, axarr = plt.subplots(3, 2, figsize=(12, 9))
    camera = Camera(fig)

    # Set up axes limits (these persist across frames)
    # Row 0: KLD
    axarr[0][0].set_xlim(-max(epochs) * 0.05, max(epochs) * 1.1)
    axarr[0][0].set_ylim(0, max(kld_jet_mass_values) * 1.1)
    axarr[0][1].set_xlim(-max(epochs) * 0.05, max(epochs) * 1.1)
    axarr[0][1].set_ylim(0, max(kld_jet_pt_values) * 1.1)
    # Row 1: W1
    axarr[1][0].set_xlim(-max(epochs) * 0.05, max(epochs) * 1.1)
    axarr[1][0].set_ylim(0, max(w1_jet_mass_values) * 1.1)
    axarr[1][1].set_xlim(-max(epochs) * 0.05, max(epochs) * 1.1)
    axarr[1][1].set_ylim(0, max(w1_jet_pt_values) * 1.1)

    # Convert to numpy arrays for fill_between
    epochs = np.array(epochs)
    kld_jet_mass_values = np.array(kld_jet_mass_values)
    kld_jet_pt_values = np.array(kld_jet_pt_values)
    kld_jet_mass_std = np.array(kld_jet_mass_std)
    kld_jet_pt_std = np.array(kld_jet_pt_std)
    w1_jet_mass_values = np.array(w1_jet_mass_values)
    w1_jet_pt_values = np.array(w1_jet_pt_values)
    w1_jet_mass_std = np.array(w1_jet_mass_std)
    w1_jet_pt_std = np.array(w1_jet_pt_std)
    truth_kld_jet_mass_values = np.array(truth_kld_jet_mass_values)
    truth_kld_jet_pt_values = np.array(truth_kld_jet_pt_values)
    truth_kld_jet_mass_std = np.array(truth_kld_jet_mass_std)
    truth_kld_jet_pt_std = np.array(truth_kld_jet_pt_std)
    truth_w1_jet_mass_values = np.array(truth_w1_jet_mass_values)
    truth_w1_jet_pt_values = np.array(truth_w1_jet_pt_values)
    truth_w1_jet_mass_std = np.array(truth_w1_jet_mass_std)
    truth_w1_jet_pt_std = np.array(truth_w1_jet_pt_std)

    # Animate: just loop and snap frames
    for frame in range(len(df_list_sorted)):
        # Row 0: KLD scatter with dashed line for history + disc for current point
        axarr[0][0].plot(
            epochs[: frame + 1],
            kld_jet_mass_values[: frame + 1],
            "--",
            color="C0",
            linewidth=1.5,
        )
        axarr[0][0].fill_between(
            epochs[: frame + 1],
            kld_jet_mass_values[: frame + 1] - kld_jet_mass_std[: frame + 1],
            kld_jet_mass_values[: frame + 1] + kld_jet_mass_std[: frame + 1],
            color="C0",
            alpha=0.3,
        )
        axarr[0][1].plot(
            epochs[: frame + 1],
            kld_jet_pt_values[: frame + 1],
            "--",
            color="C0",
            linewidth=1.5,
        )
        axarr[0][1].fill_between(
            epochs[: frame + 1],
            kld_jet_pt_values[: frame + 1] - kld_jet_pt_std[: frame + 1],
            kld_jet_pt_values[: frame + 1] + kld_jet_pt_std[: frame + 1],
            color="C0",
            alpha=0.3,
        )
        axarr[0][0].plot(
            epochs[frame], kld_jet_mass_values[frame], "o", color="C0", markersize=8
        )
        axarr[0][1].plot(
            epochs[frame], kld_jet_pt_values[frame], "o", color="C0", markersize=8
        )
        # Row 0: Truth KLD (grey)
        axarr[0][0].plot(
            epochs[: frame + 1],
            truth_kld_jet_mass_values[: frame + 1],
            "--",
            color="grey",
            linewidth=1.5,
        )
        axarr[0][0].fill_between(
            epochs[: frame + 1],
            truth_kld_jet_mass_values[: frame + 1]
            - truth_kld_jet_mass_std[: frame + 1],
            truth_kld_jet_mass_values[: frame + 1]
            + truth_kld_jet_mass_std[: frame + 1],
            color="grey",
            alpha=0.3,
        )
        axarr[0][1].plot(
            epochs[: frame + 1],
            truth_kld_jet_pt_values[: frame + 1],
            "--",
            color="grey",
            linewidth=1.5,
        )
        axarr[0][1].fill_between(
            epochs[: frame + 1],
            truth_kld_jet_pt_values[: frame + 1] - truth_kld_jet_pt_std[: frame + 1],
            truth_kld_jet_pt_values[: frame + 1] + truth_kld_jet_pt_std[: frame + 1],
            color="grey",
            alpha=0.3,
        )
        axarr[0][0].plot(
            epochs[frame],
            truth_kld_jet_mass_values[frame],
            "o",
            color="grey",
            markersize=8,
        )
        axarr[0][1].plot(
            epochs[frame],
            truth_kld_jet_pt_values[frame],
            "o",
            color="grey",
            markersize=8,
        )

        # Row 1: W1 scatter with dashed line for history + disc for current point
        axarr[1][0].plot(
            epochs[: frame + 1],
            w1_jet_mass_values[: frame + 1],
            "--",
            color="C0",
            linewidth=1.5,
        )
        axarr[1][0].fill_between(
            epochs[: frame + 1],
            w1_jet_mass_values[: frame + 1] - w1_jet_mass_std[: frame + 1],
            w1_jet_mass_values[: frame + 1] + w1_jet_mass_std[: frame + 1],
            color="C0",
            alpha=0.3,
        )
        axarr[1][1].plot(
            epochs[: frame + 1],
            w1_jet_pt_values[: frame + 1],
            "--",
            color="C0",
            linewidth=1.5,
        )
        axarr[1][1].fill_between(
            epochs[: frame + 1],
            w1_jet_pt_values[: frame + 1] - w1_jet_pt_std[: frame + 1],
            w1_jet_pt_values[: frame + 1] + w1_jet_pt_std[: frame + 1],
            color="C0",
            alpha=0.3,
        )
        axarr[1][0].plot(
            epochs[frame], w1_jet_mass_values[frame], "o", color="C0", markersize=8
        )
        axarr[1][1].plot(
            epochs[frame], w1_jet_pt_values[frame], "o", color="C0", markersize=8
        )
        # Row 1: Truth W1 (grey)
        axarr[1][0].plot(
            epochs[: frame + 1],
            truth_w1_jet_mass_values[: frame + 1],
            "--",
            color="grey",
            linewidth=1.5,
        )
        axarr[1][0].fill_between(
            epochs[: frame + 1],
            truth_w1_jet_mass_values[: frame + 1] - truth_w1_jet_mass_std[: frame + 1],
            truth_w1_jet_mass_values[: frame + 1] + truth_w1_jet_mass_std[: frame + 1],
            color="grey",
            alpha=0.3,
        )
        axarr[1][1].plot(
            epochs[: frame + 1],
            truth_w1_jet_pt_values[: frame + 1],
            "--",
            color="grey",
            linewidth=1.5,
        )
        axarr[1][1].fill_between(
            epochs[: frame + 1],
            truth_w1_jet_pt_values[: frame + 1] - truth_w1_jet_pt_std[: frame + 1],
            truth_w1_jet_pt_values[: frame + 1] + truth_w1_jet_pt_std[: frame + 1],
            color="grey",
            alpha=0.3,
        )
        axarr[1][0].plot(
            epochs[frame],
            truth_w1_jet_mass_values[frame],
            "o",
            color="grey",
            markersize=8,
        )
        axarr[1][1].plot(
            epochs[frame],
            truth_w1_jet_pt_values[frame],
            "o",
            color="grey",
            markersize=8,
        )

        # Row 2: Original histograms (use current epoch's original data)
        axarr[2][0].hist(
            binclip(original_jet_mass_data[frame], "jet_mass"),
            bins=bins_dict["jet_mass"],
            alpha=0.5,
            label="Original",
            color="k",
        )
        axarr[2][1].hist(
            binclip(original_jet_pt_data[frame], "jet_pt"),
            bins=bins_dict["jet_pt"],
            alpha=0.5,
            label="Original",
            color="k",
        )

        # Row 2: Generated histograms (current epoch only)
        axarr[2][0].hist(
            binclip(gen_jet_mass_data[frame], "jet_mass"),
            bins=bins_dict["jet_mass"],
            histtype="step",
            color="C0",
            linewidth=2,
            label="Generated",
        )
        axarr[2][1].hist(
            binclip(gen_jet_pt_data[frame], "jet_pt"),
            bins=bins_dict["jet_pt"],
            histtype="step",
            color="C0",
            linewidth=2,
            label="Generated",
        )

        # Labels (need to be redrawn each frame for celluloid)
        axarr[0][0].set_ylabel("KLD")
        axarr[0][0].set_xlabel("Epoch")
        axarr[0][0].set_title(f"Jet Mass KLD ({jet_type_to_plot})")
        axarr[0][1].set_ylabel("KLD")
        axarr[0][1].set_xlabel("Epoch")
        axarr[0][1].set_title(f"Jet pT KLD ({jet_type_to_plot})")
        axarr[1][0].set_ylabel("W1 distance")
        axarr[1][0].set_xlabel("Epoch")
        axarr[1][0].set_title(f"Jet Mass W1 ({jet_type_to_plot})")
        axarr[1][1].set_ylabel("W1 distance")
        axarr[1][1].set_xlabel("Epoch")
        axarr[1][1].set_title(f"Jet pT W1 ({jet_type_to_plot})")
        axarr[2][0].set_xlabel("Jet mass (GeV)")
        axarr[2][0].set_ylabel("Counts")
        axarr[2][1].set_xlabel("Jet pT (GeV)")
        axarr[2][1].set_ylabel("Counts")

        fig.tight_layout()
        camera.snap()

    # print(f"Truth KLD values: {truth_kld_jet_mass_values}")
    # print(f"Truth KLD stds: {truth_kld_jet_mass_std}")

    anim = camera.animate(interval=1000 // fps)

    if output_path is not None:
        anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        print(f"Animation saved to {output_path}")

    plt.close(fig)
    return anim
