import os

import awkward as ak

# import jetnet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omnilearn_lightning.generation_eval_utils import (
    calc_metrics_for_dict,
    get_jetnet_default_metrics,
)
from omnilearn_lightning.plotting.feature_plotting import (
    plot_features,
    set_mpl_style,
)
from omnilearn_lightning.utils_lightweight import (
    jet_types_dict_jetclass,
    jet_types_dict_jetnet,
)


def eta_jet_per_particle(x_ak):
    """Per-particle jet eta estimate: -delta_eta + arccosh(E / pT).

    The argument of arccosh is clipped to [1+eps, inf) so that unphysical
    generated particles with E < pT (which would give arccosh < 1) produce
    eta ~ 0 rather than NaN.
    """
    eps = 1e-6
    ratio = np.exp(x_ak["log_E"]) / np.exp(x_ak["log_pt"])
    ratio_clipped = ak.where(ratio < 1.0 + eps, 1.0 + eps, ratio)
    return -x_ak["eta"] + np.arccosh(ratio_clipped)


def compute_jet_eta_and_energy(x_ak):
    """Compute recovered jet eta and jet energy from particle-level features.

    Jet eta is the mean per-jet of eta_jet_per_particle (massless assumption).
    Jet energy is the scalar sum of per-particle energies.

    Parameters
    ----------
    x_ak : awkward array
        Particle-level features with fields: eta, phi, log_pt, log_E,
        eta_jet_per_particle.

    Returns
    -------
    jet_eta : awkward array, shape (n_jets,)
    jet_energy : awkward array, shape (n_jets,)
    """
    jet_eta = ak.mean(x_ak["eta_jet_per_particle"], axis=1)
    jet_energy = ak.sum(np.exp(x_ak["log_E"]), axis=1)
    return jet_eta, jet_energy


def eval_jet_generation(
    gen_output_path,
    results_output_path=None,
    dataset_type="jetnet150",
    num_eval_samples=10000,
    num_batches=10,
    calc_baseline=False,
    metrics_filename="metrics.csv",
    jet_types=None,
    calc_jetnet_metrics=True,
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
    calc_baseline : bool
        If True, also compute truth-vs-truth ("baseline") metrics. Default False.
    jet_types : list of int, optional
        Subset of numeric jet-type labels to evaluate. If None or empty, all jet
        types for the dataset are evaluated.
    calc_jetnet_metrics : bool
        If True (default), compute the JetNet-library default metrics
        (w1p, w1m, w1efp). Disable to skip this (relatively expensive) step.
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
        jet_types_all = range(10)
        jet_types_dict = jet_types_dict_jetclass
    elif dataset_type in ["jetnet30", "jetnet150"]:
        jet_types_all = range(5)
        jet_types_dict = jet_types_dict_jetnet
    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            "Supported: jetnet30, jetnet150, jetclass"
        )

    # Optionally restrict to a subset of (numeric) jet-type labels, preserving
    # the canonical ordering. None/empty means evaluate all jet types.
    if jet_types:
        requested = set(jet_types)
        unknown = requested - set(jet_types_all)
        if unknown:
            raise ValueError(
                f"Requested jet types {sorted(unknown)} are not valid for "
                f"dataset_type '{dataset_type}'. Valid: {list(jet_types_all)}"
            )
        jet_types = [t for t in jet_types_all if t in requested]
        print(f"Restricting evaluation to jet types: {jet_types}")
    else:
        jet_types = list(jet_types_all)

    feature_names_x = {
        "eta": "Particle $\\Delta\\eta$",
        "phi": "Particle $\\Delta\\phi$",
        "log_pt": "Particle $\\log(p_{T})$",
        "log_E": "Particle $\\log(E)$",
        "eta_jet_per_particle": "Particle $\\eta_{jet}$ estimate",
    }

    bins_dict_particle = {
        "eta": np.linspace(-1, 1, 50),
        "phi": np.linspace(-1, 1, 50),
        "log_pt": np.linspace(-3, 6.5, 50),
        "log_E": np.linspace(-3, 6.5, 50),
        "eta_jet_per_particle": np.linspace(0, 3, 50),
    }

    bins_dict_substructure = {
        "tau21": np.linspace(0, 1.2, 60),
        "tau32": np.linspace(0, 1.2, 60),
        "jet_mass": np.linspace(0, 250, 60),
        "jet_pt": np.linspace(400, 1400, 60),
        "jet_n_constituents": np.linspace(-0.5, 150.5, 152),
        "d2": np.linspace(0, 12, 50),
        "jet_eta": np.linspace(0, 3, 60),
        "jet_energy": np.linspace(400, 2000, 60),
    }

    names_substructure = {
        "tau21": "$\\tau_{21}$",
        "tau32": "$\\tau_{32}$",
        "d2": "$D_{2}$",
        "jet_mass": "Jet mass [GeV]",
        "jet_pt": "Jet $p_{T}$ [GeV]",
        "jet_n_constituents": "Particle multiplicity",
        "jet_eta": "Jet $\\eta$ (recovered)",
        "jet_energy": "Jet energy [GeV]",
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

    # Detect truth-vs-truth reference format: y is stored as a dict with
    # original/generated sub-fields (independent train/test labels).
    # In the standard format, original and generated share the same y array.
    y_original = (
        gen_output.y.original if hasattr(gen_output.y, "original") else gen_output.y
    )
    y_generated = (
        gen_output.y.generated if hasattr(gen_output.y, "generated") else gen_output.y
    )

    # -------------------------------------------------------------------------
    # Build enriched x_ak and substructure arrays (all jet types, full dataset).
    # Derived fields (eta_jet_per_particle, jet_eta, jet_energy) are computed
    # once here so they are available both inside the per-type loop and in the
    # enriched parquet written at the end.
    # -------------------------------------------------------------------------
    x_ak_orig_full = gen_output.x_ak.original
    x_ak_gen_full = gen_output.x_ak.generated

    x_ak_orig_full = ak.with_field(
        x_ak_orig_full,
        eta_jet_per_particle(x_ak_orig_full),
        "eta_jet_per_particle",
    )
    x_ak_gen_full = ak.with_field(
        x_ak_gen_full,
        eta_jet_per_particle(x_ak_gen_full),
        "eta_jet_per_particle",
    )

    jet_eta_orig_full, jet_energy_orig_full = compute_jet_eta_and_energy(x_ak_orig_full)
    jet_eta_gen_full, jet_energy_gen_full = compute_jet_eta_and_energy(x_ak_gen_full)

    substructure_orig_full = ak.with_field(
        ak.with_field(gen_output.substructure.original, jet_eta_orig_full, "jet_eta"),
        jet_energy_orig_full,
        "jet_energy",
    )
    substructure_gen_full = ak.with_field(
        ak.with_field(gen_output.substructure.generated, jet_eta_gen_full, "jet_eta"),
        jet_energy_gen_full,
        "jet_energy",
    )

    # Loop over jet types
    for jet_type_i in jet_types:
        mask_original = y_original == jet_type_i
        mask_generated = y_generated == jet_type_i
        n_jets_this_type = ak.sum(mask_original)
        # Slice from the pre-enriched full arrays (eta_jet_per_particle,
        # jet_eta, jet_energy already attached before the loop).
        x_ak_original_this_type = x_ak_orig_full[mask_original]
        x_ak_generated_this_type = x_ak_gen_full[mask_generated]
        substructure_original = substructure_orig_full[mask_original]
        substructure_generated = substructure_gen_full[mask_generated]
        p4s_original = gen_output.p4s.original[mask_original]
        p4s_generated = gen_output.p4s.generated[mask_generated]

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
                "Target": x_ak_original_this_type,
                "Generated": x_ak_generated_this_type,
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
                "Target": substructure_original,
                "Generated": substructure_generated,
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
            dict_reference=x_ak_original_this_type,
            dict_approx=x_ak_generated_this_type,
            names=list(feature_names_x.keys()),
            **metrics_calc_kwargs,
        )
        # Calculate the baseline values (i.e. the KLD/W1 when sampling from the
        # same distribution) for reference
        if calc_baseline:
            particle_metrics_baseline = calc_metrics_for_dict(
                dict_reference=x_ak_original_this_type,
                dict_approx=x_ak_original_this_type,
                names=list(feature_names_x.keys()),
                **metrics_calc_kwargs,
            )

        # Calculate the JetNet default metrics w1p, w1m, w1efp (optional)
        if calc_jetnet_metrics:
            jetnet_library_metrics = get_jetnet_default_metrics(
                p4s_ref=p4s_original,
                p4s_gen=p4s_generated,
                num_eval_samples=num_eval_samples,
                num_batches=num_batches,
                divide_pt_by_sum_pt=True,  # use pT fractions for JetNet metrics by default
            )
            jetnet_library_metrics_non_relative_pT = get_jetnet_default_metrics(
                p4s_ref=p4s_original,
                p4s_gen=p4s_generated,
                num_eval_samples=num_eval_samples,
                num_batches=num_batches,
                divide_pt_by_sum_pt=False,  # also calculate with non-relative pT for reference
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

        # Add baseline metrics with prefix
        if calc_baseline:
            for metric_name, metric_values in particle_metrics_baseline.items():
                for feature_name, (mean_val, std_val) in metric_values.items():
                    print(
                        f"  {metric_name}_{feature_name}_baseline: {mean_val:.4f} +/- {std_val:.4f}"
                    )
                    all_metrics.append(
                        {
                            "jet_type": jet_type_label,
                            "jet_type_idx": jet_type_i,
                            "level": "particle",
                            "feature": feature_name,
                            "metric": f"{metric_name}_baseline",
                            "mean": mean_val,
                            "std": std_val,
                        }
                    )
        # Add JetNet library metrics (optional)
        if calc_jetnet_metrics:
            for metric_name, (mean_val, std_val) in jetnet_library_metrics.items():
                print(f"  {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")
                all_metrics.append(
                    {
                        "jet_type": jet_type_label,
                        "jet_type_idx": jet_type_i,
                        "level": "particle",
                        "feature": metric_name,  # use metric name as feature for these
                        "metric": metric_name,
                        "mean": mean_val,
                        "std": std_val,
                    }
                )
            for metric_name, (
                mean_val,
                std_val,
            ) in jetnet_library_metrics_non_relative_pT.items():
                print(f"  {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")
                all_metrics.append(
                    {
                        "jet_type": jet_type_label,
                        "jet_type_idx": jet_type_i,
                        "level": "particle",
                        "feature": f"{metric_name}_non_relative_pT",  # use metric name as feature for these
                        "metric": f"{metric_name}_non_relative_pT",
                        "mean": mean_val,
                        "std": std_val,
                    }
                )

        # Calculate metrics for jet-level (substructure) features
        jet_metrics = calc_metrics_for_dict(
            dict_reference=substructure_original,
            dict_approx=substructure_generated,
            names=list(names_substructure.keys()),
            **metrics_calc_kwargs,
        )
        # Calculate baseline metrics for jet-level features
        if calc_baseline:
            jet_metrics_baseline = calc_metrics_for_dict(
                dict_reference=substructure_original,
                dict_approx=substructure_original,
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
        # Add baseline metrics with prefix
        if calc_baseline:
            for metric_name, metric_values in jet_metrics_baseline.items():
                for feature_name, (mean_val, std_val) in metric_values.items():
                    all_metrics.append(
                        {
                            "jet_type": jet_type_label,
                            "jet_type_idx": jet_type_i,
                            "level": "jet",
                            "feature": feature_name,
                            "metric": f"{metric_name}_baseline",
                            "mean": mean_val,
                            "std": std_val,
                        }
                    )

    # -------------------------------------------------------------------------
    # Write enriched parquet alongside the original.
    # Contains the same structure as the input but with derived fields added:
    #   x_ak:         + eta_jet_per_particle
    #   substructure: + jet_eta, jet_energy
    # -------------------------------------------------------------------------
    enriched_path = gen_output_path.replace(".parquet", "_enriched.parquet")
    # The enriched parquet always uses the truth-vs-truth struct layout
    # (original/generated sub-fields for every top-level field), regardless
    # of whether the input was in standard format (flat y array) or
    # truth-vs-truth format (y.original / y.generated).
    # load_gen_jets_for_jet_type in plotting.py handles both layouts via the
    # hasattr(gen_jets.y, "original") branch, so this is transparent to callers.
    enriched = ak.Array(
        {
            "x_ak": {
                "original": x_ak_orig_full,
                "generated": x_ak_gen_full,
            },
            "substructure": {
                "original": substructure_orig_full,
                "generated": substructure_gen_full,
            },
            "p4s": {
                "original": gen_output.p4s.original,
                "generated": gen_output.p4s.generated,
            },
            "y": {
                "original": y_original,
                "generated": y_generated,
            },
        }
    )
    ak.to_parquet(enriched, enriched_path)
    print(f"Saved enriched parquet to {enriched_path}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_path, metrics_filename)
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
    parser.add_argument(
        "--calc_baseline",
        action="store_true",
        default=False,
        help="Also compute truth-vs-truth baseline metrics (default: False)",
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default="metrics.csv",
        help="Filename for the saved metrics CSV (default: metrics.csv)",
    )
    parser.add_argument(
        "--jet_types",
        type=str,
        default="",
        help="Comma-separated numeric jet-type labels to evaluate (e.g. '0,1'). "
        "Empty (default) evaluates all jet types for the dataset.",
    )
    parser.add_argument(
        "--no_jetnet_metrics",
        dest="calc_jetnet_metrics",
        action="store_false",
        default=True,
        help="Skip the JetNet-library default metrics (w1p, w1m, w1efp).",
    )

    args = parser.parse_args()

    jet_types = (
        [int(t.strip()) for t in args.jet_types.split(",") if t.strip()]
        if args.jet_types
        else None
    )

    eval_jet_generation(
        gen_output_path=args.gen_output_path,
        results_output_path=args.results_output_path,
        dataset_type=args.dataset_type,
        num_eval_samples=args.num_eval_samples,
        num_batches=args.num_batches,
        calc_baseline=args.calc_baseline,
        metrics_filename=args.metrics_filename,
        jet_types=jet_types,
        calc_jetnet_metrics=args.calc_jetnet_metrics,
    )
