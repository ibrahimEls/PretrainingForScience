import os

import awkward as ak

# import jetnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnilearn_lightning.generation_eval_utils import calc_metrics_for_dict
from omnilearn_lightning.plotting.feature_plotting import (
    plot_features,
    set_mpl_style,
)
from omnilearn_lightning.utils_lightweight import (
    jet_types_dict_jetclass,
    jet_types_dict_jetnet,
)


def eval_jet_generation(
    gen_output_path,
    results_output_path=None,
    dataset_type="jetnet150",
    num_eval_samples=10000,
    num_batches=10,
):
    """Evaluate generated jets with per-jet-type plotting and metrics.

    Parameters
    ----------
    gen_output_path : str
        Path to the parquet file containing generated jets.
    results_output_path : str, optional
        Path to save evaluation results (plots and metrics). If None, saves in
        the same directory as gen_output_path with "_plots" suffix.
    dataset_type : str
        Dataset type, one of "jetnet30", "jetnet150", or "jetclass".
    num_eval_samples : int
        Number of samples per batch for metric calculation.
    num_batches : int
        Number of batches for computing mean and std of metrics.
    """
    print(f"Loading generated data from output path: {gen_output_path}")
    gen_output = ak.from_parquet(gen_output_path)

    if results_output_path is None:
        output_path = gen_output_path.replace(".parquet", "_plots")
    else:
        output_path = results_output_path

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving evaluation plots to: {output_path}")

    # Select jet types based on dataset
    if dataset_type == "jetclass":
        jet_types = range(10)
        jet_types_dict = jet_types_dict_jetclass
    elif dataset_type in ["jetnet30", "jetnet150"]:
        jet_types = range(5)
        jet_types_dict = jet_types_dict_jetnet
    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            "Supported: jetnet30, jetnet150, jetclass"
        )

    feature_names_x = {
        "eta": "Particle $\\Delta\\eta$",
        "phi": "Particle $\\Delta\\phi$",
        "log_pt": "Particle $\\log(p_{T})$",
        "log_E": "Particle $\\log(E)$",
    }

    bins_dict_particle = {
        "eta": np.linspace(-1, 1, 50),
        "phi": np.linspace(-1, 1, 50),
        "log_pt": np.linspace(-3, 6.5, 50),
        "log_E": np.linspace(-3, 6.5, 50),
    }

    bins_dict_substructure = {
        "tau21": np.linspace(0, 1.2, 60),
        "tau32": np.linspace(0, 1.2, 60),
        "jet_mass": np.linspace(0, 250, 60),
        "jet_pt": np.linspace(400, 1400, 60),
        "jet_n_constituents": np.linspace(-0.5, 100.5, 102),
        "d2": np.linspace(0, 12, 50),
    }

    names_substructure = {
        "tau21": "$\\tau_{21}$",
        "tau32": "$\\tau_{32}$",
        "d2": "$D_{2}$",
        "jet_mass": "Jet mass [GeV]",
        "jet_pt": "Jet $p_{T}$ [GeV]",
        "jet_n_constituents": "Particle multiplicity",
    }

    plot_save_kwargs = dict(dpi=200, bbox_inches="tight")

    plot_kwargs_common = dict(
        ratio=True,
        ax_rows=2,
        ax_size=(2.1, 1.5),
        decorate_ax_kwargs=dict(yscale=1.0),
    )

    metrics_calc_kwargs = dict(
        return_zero_if_nan_or_inf=True,
        num_eval_samples=num_eval_samples,
        num_batches=num_batches,
    )

    # Collect all metrics for CSV output
    all_metrics = []

    set_mpl_style()

    # Loop over jet types
    for jet_type_i in jet_types:
        mask_this_type = gen_output.y == jet_type_i
        n_jets_this_type = ak.sum(mask_this_type)

        if n_jets_this_type == 0:
            print(f"No jets of type {jet_type_i}, skipping...")
            continue

        jet_type_label = jet_types_dict[jet_type_i]["label"]
        tex_label = jet_types_dict[jet_type_i]["tex_label"]
        print(
            f"\nProcessing jet type {jet_type_i} ({jet_type_label}): {n_jets_this_type} jets"
        )

        fig_legend_kwargs = dict(
            bbox_to_anchor=(0.5, 1.10),
            loc="upper center",
            fontsize=10,
            title=f"{tex_label} jets",
            ncol=2,
        )

        # Plot particle features for this jet type
        fig, axarr = plot_features(
            {
                "Target": gen_output[mask_this_type].x_ak.original,
                "Generated": gen_output[mask_this_type].x_ak.generated,
            },
            bins_dict=bins_dict_particle,
            names=feature_names_x,
            flatten=True,
            fig_legend_kwargs=fig_legend_kwargs,
            **plot_kwargs_common,
        )
        outfile = os.path.join(output_path, f"particle_features_{jet_type_label}.png")
        fig.savefig(outfile, **plot_save_kwargs)
        print(f"Saved {outfile}")
        plt.close(fig)

        # Plot substructure features for this jet type
        fig, axarr = plot_features(
            {
                "Target": gen_output[mask_this_type].substructure.original,
                "Generated": gen_output[mask_this_type].substructure.generated,
            },
            names=names_substructure,
            bins_dict=bins_dict_substructure,
            flatten=False,
            fig_legend_kwargs=fig_legend_kwargs,
            **plot_kwargs_common,
        )
        outfile = os.path.join(
            output_path, f"substructure_features_{jet_type_label}.png"
        )
        fig.savefig(outfile, **plot_save_kwargs)
        print(f"Saved {outfile}")
        plt.close(fig)

        # Calculate metrics for particle-level features
        particle_metrics = calc_metrics_for_dict(
            dict_reference=gen_output[mask_this_type].x_ak.original,
            dict_approx=gen_output[mask_this_type].x_ak.generated,
            names=list(feature_names_x.keys()),
            **metrics_calc_kwargs,
        )
        # Calculate the truth values (i.e. the KLD/W1 when sampling from the
        # same distribution) for reference
        particle_metrics_truth = calc_metrics_for_dict(
            dict_reference=gen_output[mask_this_type].x_ak.original,
            dict_approx=gen_output[mask_this_type].x_ak.original,
            names=list(feature_names_x.keys()),
            **metrics_calc_kwargs,
        )

        print(f"Particle-level metrics for {jet_type_label}:")
        for metric_name, metric_values in particle_metrics.items():
            for feature_name, (mean_val, std_val) in metric_values.items():
                print(
                    f"  {metric_name}_{feature_name}: {mean_val:.4f} +/- {std_val:.4f}"
                )
                all_metrics.append(
                    {
                        "jet_type": jet_type_label,
                        "jet_type_idx": jet_type_i,
                        "level": "particle",
                        "feature": feature_name,
                        "metric": metric_name,
                        "mean": mean_val,
                        "std": std_val,
                    }
                )

        # Add truth metrics with prefix
        for metric_name, metric_values in particle_metrics_truth.items():
            for feature_name, (mean_val, std_val) in metric_values.items():
                all_metrics.append(
                    {
                        "jet_type": jet_type_label,
                        "jet_type_idx": jet_type_i,
                        "level": "particle",
                        "feature": feature_name,
                        "metric": f"truth_{metric_name}",
                        "mean": mean_val,
                        "std": std_val,
                    }
                )

        # Calculate metrics for jet-level (substructure) features
        jet_metrics = calc_metrics_for_dict(
            dict_reference=gen_output[mask_this_type].substructure.original,
            dict_approx=gen_output[mask_this_type].substructure.generated,
            names=list(names_substructure.keys()),
            **metrics_calc_kwargs,
        )
        # Calculate truth metrics for jet-level features
        jet_metrics_truth = calc_metrics_for_dict(
            dict_reference=gen_output[mask_this_type].substructure.original,
            dict_approx=gen_output[mask_this_type].substructure.original,
            names=list(names_substructure.keys()),
            **metrics_calc_kwargs,
        )

        print(f"Jet-level metrics for {jet_type_label}:")
        for metric_name, metric_values in jet_metrics.items():
            for feature_name, (mean_val, std_val) in metric_values.items():
                print(
                    f"  {metric_name}_{feature_name}: {mean_val:.4f} +/- {std_val:.4f}"
                )
                all_metrics.append(
                    {
                        "jet_type": jet_type_label,
                        "jet_type_idx": jet_type_i,
                        "level": "jet",
                        "feature": feature_name,
                        "metric": metric_name,
                        "mean": mean_val,
                        "std": std_val,
                    }
                )
        # Add truth metrics with prefix
        for metric_name, metric_values in jet_metrics_truth.items():
            for feature_name, (mean_val, std_val) in metric_values.items():
                all_metrics.append(
                    {
                        "jet_type": jet_type_label,
                        "jet_type_idx": jet_type_i,
                        "level": "jet",
                        "feature": feature_name,
                        "metric": f"truth_{metric_name}",
                        "mean": mean_val,
                        "std": std_val,
                    }
                )

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_path, "metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    return metrics_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated jets")
    parser.add_argument(
        "--gen_output_path",
        type=str,
        # TODO: remove this default once development is done
        default="/pscratch/sd/j/jobirk/omnilearned_output/omnilearn_output/_micro_generator_jetnet150_dataset_550000_jobirk_ckpt_from_scratch/v29_micro_generator_jetnet150_dataset_550000_jobirk_ckpt_from_scratch/generation_output/best/generated_jets_10000.parquet",
        help="Path to the parquet file containing generated jets",
    )
    parser.add_argument(
        "--results_output_path",
        type=str,
        default=None,
        help="Path to save evaluation results (plots and metrics). If None,"
        " saves in the same directory as gen_output_path with '_plots' suffix.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="jetnet150",
        choices=["jetnet30", "jetnet150", "jetclass"],
        help="Dataset type (default: jetnet150)",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        # default=10000,
        help="Number of samples per batch for metric calculation",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        # default=10,
        help="Number of batches for computing mean and std",
    )

    args = parser.parse_args()

    eval_jet_generation(
        gen_output_path=args.gen_output_path,
        results_output_path=args.results_output_path,
        dataset_type=args.dataset_type,
        num_eval_samples=args.num_eval_samples,
        num_batches=args.num_batches,
    )
