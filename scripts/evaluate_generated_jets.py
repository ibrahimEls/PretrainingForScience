import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from omnilearn_lightning.plotting.feature_plotting import (
    plot_features,
    set_mpl_style,
)


def eval_jet_generation(gen_output_path):
    print(f"Loading generated data from output path: {gen_output_path}")
    gen_output = ak.from_parquet(gen_output_path)

    output_path = gen_output_path.replace(".parquet", "_plots")
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving evaluation plots to: {output_path}")

    feature_names_x = {
        "eta": "Particle $\\Delta\\eta$",
        "phi": "Particle $\\Delta\\phi$",
        "log_pt": "Particle $\\log(p_{T})$",
        "log_E": "Particle $\\log(E)$",
    }

    plot_save_kwargs = dict(
        dpi=200,
        bbox_inches="tight",
    )

    # plot particle features
    set_mpl_style()
    fig, axarr = plot_features(
        {
            "Target": gen_output.x_ak.original,
            "Generated": gen_output.x_ak.generated,
        },
        bins_dict={
            "eta": np.linspace(-1, 1, 50),
            "phi": np.linspace(-1, 1, 50),
            "log_pt": np.linspace(-3, 6.5, 50),
            "log_E": np.linspace(-3, 6.5, 50),
        },
        ratio=True,
        names=feature_names_x,
        flatten=True,
    )
    plt.show()
    fig.savefig(os.path.join(output_path, "particle_features.png"), **plot_save_kwargs)

    # plot jet features
    set_mpl_style()
    fig, axarr = plot_features(
        {
            "Target": gen_output.p4s_sum.original,
            "Generated": gen_output.p4s_sum.generated,
        },
        bins_dict={
            "pt": np.linspace(0, 1500, 50),
            "eta": np.linspace(-2.5, 2.5, 50),
            "phi": np.linspace(-3.14, 3.14, 50),
            "mass": np.linspace(0, 200, 50),
        },
        names={
            "pt": "Jet $p_{T}$",
            "eta": "Jet $\\eta$",
            "phi": "Jet $\\phi$",
            "mass": "Jet mass",
        },
        ratio=True,
        flatten=False,
    )
    plt.show()
    fig.savefig(os.path.join(output_path, "jet_features.png"), **plot_save_kwargs)

    bins_dict_substructure = {
        "tau21": np.linspace(0, 1.2, 80),
        "tau32": np.linspace(0, 1.2, 80),
        "jet_mass": np.linspace(0, 250, 80),
        "jet_pt": np.linspace(200, 1200, 80),
        "jet_n_constituents": np.linspace(-0.5, 100.5, 102),
        "d2": np.linspace(0, 5, 50),
    }
    names_substructure = {
        "tau21": "$\\tau_{21}$",
        "tau32": "$\\tau_{32}$",
        "d2": "$D_{2}$",
        "jet_mass": "Jet mass [GeV]",
        "jet_pt": "Jet $p_{T}$ [GeV]",
        "jet_n_constituents": "Particle multiplicity",
    }

    set_mpl_style()
    fig, axarr = plot_features(
        {
            "Target": gen_output.substructure.original,
            "Generated": gen_output.substructure.generated,
        },
        names=names_substructure,
        # names=substructure_dict[0].fields,
        bins_dict=bins_dict_substructure,
        flatten=False,
        ax_rows=2,
        ax_size=(4.0, 2.5),
        legend_kwargs=dict(ncol=2, loc="upper center"),
        ratio=True,
    )
    plt.show()
    fig.savefig(
        os.path.join(output_path, "substructure_features.png"), **plot_save_kwargs
    )


if __name__ == "__main__":
    gen_output_path = "/pscratch/sd/j/jobirk/omnilearned_output/omnilearn_output/_micro_generator_jetnet150_dataset_100000_jobirk_ckpt_from_scratch/v13_micro_generator_jetnet150_dataset_100000_jobirk_ckpt_from_scratch/generation_output/best/generated_jets_200.parquet"
    eval_jet_generation(gen_output_path)
