import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from omnilearn_lightning.generation_utils import (
    generate_and_postprocess,
)
from omnilearn_lightning.plotting.feature_plotting import (
    plot_features,
    set_mpl_style,
)

from .dataloader import PETDataModule
from .model import PETLightning


def eval_top_tagging(args, ckpt_path, gpuID):
    best_model = PETLightning.load_from_checkpoint(
        ckpt_path, fine_tune=False, ckpt_loaded=ckpt_path
    )
    best_model.eval()

    device = torch.device(f"cuda:{gpuID}" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    # Define loss
    criterion = torch.nn.BCEWithLogitsLoss()

    all_labels = []
    all_scores = []
    total_loss = 0.0
    n_batches = 0

    data_module = PETDataModule(
        dataset="top",
        path=args.path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_pid=args.use_pid,
        use_add=args.use_add,
        num_samples=-1,
        load_val=True,
    )

    data_module.setup("fit")
    dataloader = data_module.val_dataloader()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val Batches"):
            inputs, labels = batch["X"], batch["y"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            model_kwargs = {
                key: (batch[key].to(device) if batch[key] is not None else None)
                for key in ["cond", "pid", "add_info"]
                if key in batch
            }
            out = best_model(inputs, labels, **model_kwargs)
            y_pred = out["y_pred"].softmax(dim=1)
            outputs = ((1 - y_pred[:, 0]) + (y_pred[:, 1])) / 2

            loss = criterion(outputs.unsqueeze(1), labels.unsqueeze(1).float())
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_scores.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / n_batches
    auc = roc_auc_score(all_labels, all_scores)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    target_eff = 0.3
    idx = np.where(tpr >= target_eff)[0][0]
    bkg_eff = fpr[idx]
    inv_bkg_eff = 1.0 / bkg_eff if bkg_eff > 0 else float("inf")

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"1/(background efficiency): {inv_bkg_eff:.4f}")

    return (avg_loss, auc, inv_bkg_eff)


def eval_quark_gluon(args, ckpt_path, gpuID):
    best_model = PETLightning.load_from_checkpoint(
        ckpt_path, fine_tune=False, ckpt_loaded=ckpt_path
    )
    best_model.eval()

    device = torch.device(f"cuda:{gpuID}" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    # Define loss
    criterion = torch.nn.BCEWithLogitsLoss()
    all_labels = []
    all_scores = []
    total_loss = 0.0
    n_batches = 0

    data_module = PETDataModule(
        dataset="qg",
        path=args.path,
        batch_size=256,
        num_workers=args.num_workers,
        use_pid=args.use_pid,
        use_add=args.use_add,
        num_samples=-1,
    )

    data_module.setup("fit")
    dataloader = data_module.val_dataloader()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val Batches"):
            inputs, labels = batch["X"], batch["y"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            model_kwargs = {
                key: (batch[key].to(device) if batch[key] is not None else None)
                for key in ["cond", "pid", "add_info"]
                if key in batch
            }
            out = best_model(inputs, labels, **model_kwargs)
            y_pred = out["y_pred"].softmax(dim=1)
            outputs = ((1 - y_pred[:, 0]) + (y_pred[:, 1])) / 2

            loss = criterion(outputs.unsqueeze(1), labels.unsqueeze(1).float())
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_scores.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / n_batches
    auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    target_eff = 0.3
    idx = np.where(tpr >= target_eff)[0][0]
    bkg_eff = fpr[idx]
    inv_bkg_eff = 1.0 / bkg_eff if bkg_eff > 0 else float("inf")

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"1/(background efficiency): {inv_bkg_eff:.4f}")

    return (avg_loss, auc, inv_bkg_eff)


def eval_jet_generation(ckpt_path, output_path):
    print(f"Loading model from checkpoint: {ckpt_path}")
    lightning_model = PETLightning.load_from_checkpoint(ckpt_path)
    print(lightning_model.model.generator)

    os.makedirs(output_path, exist_ok=True)
    # resolve output path and print
    print(f"Saving evaluation plots to: {output_path}")

    # dataset_type = "top"
    dataset_type = "jetclass"

    datamodule = PETDataModule(
        dataset=dataset_type,
        path="/pscratch/sd/j/jobirk/omnilearn_datasets_dev/",
        batch_size=200,
        num_samples=-1,
        use_pid=True if dataset_type == "jetclass" else False,
        use_add=True if dataset_type == "jetclass" else False,
        use_cond=True,
        shuffle_val_test_indices=True,
        seed_for_initial_shuffling=42,
        load_val=True,
    )

    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    feature_names_x = {
        "eta": "Particle $\\Delta\\eta$",
        "phi": "Particle $\\Delta\\phi$",
        "log_pt": "Particle $\\log(p_{T})$",
        "log_E": "Particle $\\log(E)$",
    }

    n_batches = 1
    n_steps_to_test = [0, 2, 128]
    ak_arrays_vs_steps, p4s_dict, p4s_sum_dict, substructure_dict = (
        generate_and_postprocess(
            n_batches,
            n_steps_to_test,
            dataloader,
            lightning_model,
            feature_names_x,
        )
    )

    plot_save_kwargs = dict(
        dpi=200,
        bbox_inches="tight",
    )

    # plot particle features
    set_mpl_style()
    fig, axarr = plot_features(
        {
            "Original": ak_arrays_vs_steps[0],
        }
        | {
            f"Generated ({n_steps} steps)": ak_arrays_vs_steps[n_steps]
            for n_steps in ak_arrays_vs_steps.keys()
            if n_steps != 0
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
            "Original": p4s_sum_dict[0],
        }
        | {
            f"Generated ({n_steps} steps)": p4s_sum_dict[n_steps]
            for n_steps in p4s_sum_dict.keys()
            if n_steps != 0
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
            "Original": substructure_dict[0],
        }
        | {
            f"Gen. ({n_steps} steps)": substructure_dict[n_steps]
            for n_steps in substructure_dict.keys()
            if n_steps != 0
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
