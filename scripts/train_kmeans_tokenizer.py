import argparse
import json
import os
import sys

sys.path.append("..")

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from omnilearn_lightning.array_utils import np_to_ak, p4s_from_ptetaphimass
from omnilearn_lightning.dataloader import PETDataModule
from omnilearn_lightning.plotting.feature_plotting import (
    plot_features,
    plot_features_pairplot,
)
from omnilearn_lightning.tokenizer import KMeansTokenizer
from omnilearn_lightning.utils import get_bigram

NUM_TRAINING_SAMPLES = 1_000_000
NUM_PLOTS_SAMPLES = 10_000
NUM_MAX_POINTS_PER_JET = 100
NUM_FEATURES_X = 4
NUM_FEATURES_ADD = 4
# CODEBOOK_SIZE = 8_192
CODEBOOK_SIZE = 16_384
# Default scale factors for x and add_info
DEFAULT_SCALE_FACTORS_X = torch.tensor(
    [
        0.13,  # eta
        0.13,  # phi
        0.6,  # log pt
        0.6,  # log energy
    ]
)
DEFAULT_SCALE_FACTORS_ADD_INFO = torch.tensor(
    [
        0.1,  # d0val
        1.0,  # d0err
        0.1,  # dzval
        1.0,  # dzerr
    ]
)
SAVE_FIG_KWARGS = dict(dpi=150, bbox_inches="tight")


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(
    description="Train KMeans tokenizer for JetClass or top dataset."
)
parser.add_argument(
    "--use-add",
    dest="use_add_info",
    help="Include additional features (add_info) in KMeans training and tokenization.",
    type=str2bool,
)
parser.add_argument(
    "--output-dir",
    type=str,
    help="Output directory to save the trained model and plots. ",
)
parser.add_argument(
    "--scale-factors-x",
    type=str,
    default=None,
    help=(
        "Scale factors for particle features (x). "
        "Provide as a comma-separated list of 4 floats (eta,phi,log_pt,log_E), "
        "a single float to apply to all, or 'default' to use built-ins."
    ),
)
parser.add_argument(
    "--scale-factors-add",
    type=str,
    default=None,
    help=(
        "Scale factors for add_info features. "
        "Provide as a comma-separated list of 4 floats (d0val,d0err,dzval,dzerr), "
        "a single float to apply to all, or 'default'. Ignored if --use-add is false."
    ),
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["jetclass", "top"],
    default="jetclass",
    help="Dataset to use for training the tokenizer.",
)
parser.set_defaults(use_add_info=True)
args = parser.parse_args()


def _parse_scale_factors(
    arg_value: str | None, default_tensor: torch.Tensor, expected_len: int
) -> torch.Tensor:
    """Parse a scale-factor CLI value into a torch tensor.

    Accepted formats:
    - None or 'default' (case-insensitive): return the provided default tensor.
    - Single float like '0.5': replicated to expected_len.
    - Comma-separated list like '0.1,1.0,0.1,1.0' with length == expected_len.
    """
    if arg_value is None or str(arg_value).lower() in {"default", "auto", ""}:
        return default_tensor.clone()
    text = str(arg_value).strip()
    # Try single float
    try:
        single = float(text)
        return torch.tensor([single] * expected_len, dtype=default_tensor.dtype)
    except ValueError:
        pass
    # Try CSV
    try:
        parts = [p.strip() for p in text.split(",") if p.strip() != ""]
        values = [float(p) for p in parts]
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid scale factor specification '{arg_value}': {e}"
        ) from e
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} values, got {len(values)} in '{arg_value}'."
        )
    return torch.tensor(values, dtype=default_tensor.dtype)


# Resolve scale factors from CLI
SCALE_FACTORS_X = _parse_scale_factors(
    args.scale_factors_x, DEFAULT_SCALE_FACTORS_X, NUM_FEATURES_X
)
SCALE_FACTORS_ADD_INFO = (
    _parse_scale_factors(
        args.scale_factors_add, DEFAULT_SCALE_FACTORS_ADD_INFO, NUM_FEATURES_ADD
    )
    if args.use_add_info
    else None
)

# filepath should be unique: time stamp and bigram; add visibility for add_info usage
ADD_TAG = "withAddInfo" if args.use_add_info else "noAddInfo"
UNIQUE_ID = f"kmeans_model_{get_bigram(add_timestamp=True, timestamp_first=True)}_{ADD_TAG}_{CODEBOOK_SIZE}_codes"
OUTPUT_DIR = f"{args.output_dir}/{UNIQUE_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DIR = f"{OUTPUT_DIR}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# dump the scale factors being used as json
scale_factors_dict = {
    "scale_factors_x": SCALE_FACTORS_X.tolist(),
    "scale_factors_add_info": (
        SCALE_FACTORS_ADD_INFO.tolist() if SCALE_FACTORS_ADD_INFO is not None else None
    ),
}
with open(f"{OUTPUT_DIR}/scale_factors.json", "w") as f:
    json.dump(scale_factors_dict, f, indent=4)


datamodule = PETDataModule(
    dataset=args.dataset,
    path="/pscratch/sd/j/jobirk/omnilearn_datasets/",
    batch_size=1_000,
    num_samples=NUM_TRAINING_SAMPLES,
    use_pid=args.dataset == "jetclass",  # only jetclass has pid
    use_add=args.dataset == "jetclass",  # only jetclass has add_info
    shuffle_train_indices=True,
    shuffle_val_test_indices=True,
    seed_for_initial_shuffling=42,
)
datamodule.setup("fit")

dataloader = datamodule.train_dataloader()

# Use CUDA device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

training_x = torch.zeros(
    (NUM_TRAINING_SAMPLES, NUM_MAX_POINTS_PER_JET, NUM_FEATURES_X),
    device=device,
    requires_grad=False,
)
training_add_info = torch.zeros(
    (NUM_TRAINING_SAMPLES, NUM_MAX_POINTS_PER_JET, NUM_FEATURES_ADD),
    device=device,
    requires_grad=False,
)
training_y = torch.zeros(
    (NUM_TRAINING_SAMPLES,), dtype=torch.long, device=device, requires_grad=False
)

# loop over batches to collect training data for kmeans
start_idx = 0
for batch in tqdm(dataloader):
    batch_size = batch["X"].shape[0]
    end_idx = start_idx + batch_size
    if end_idx > NUM_TRAINING_SAMPLES:
        end_idx = NUM_TRAINING_SAMPLES
        batch_size = end_idx - start_idx

    max_n_points_in_batch = batch["X"].shape[1]
    fill_until = min(max_n_points_in_batch, NUM_MAX_POINTS_PER_JET)

    training_x[start_idx:end_idx, :fill_until] = batch["X"][
        :batch_size, :fill_until
    ].to(device)
    if "add_info" in batch and args.use_add_info:
        training_add_info[start_idx:end_idx, :fill_until] = batch["add_info"][
            :batch_size, :fill_until
        ].to(device)
    training_y[start_idx:end_idx] = batch["y"][:batch_size].to(device)
    start_idx += batch_size
    if start_idx >= NUM_TRAINING_SAMPLES:
        break

# bar plot of training_y
fig, ax = plt.subplots(figsize=(3, 2))
ax.hist(training_y.cpu().numpy(), bins=np.linspace(-0.5, 9.5, 11))
ax.set_xlabel("Jet type")
ax.set_ylabel("Number of jets")
ax.set_xticks(np.arange(0, 10, 1))
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/training_y_distribution.pdf", **SAVE_FIG_KWARGS)
plt.close(fig)

mask = training_x[:, :, 0] != 0  # mask for valid points

kmeans_tokenizer = KMeansTokenizer(
    n_clusters=CODEBOOK_SIZE,
    scale_factors_x=SCALE_FACTORS_X,
    scale_factors_add_info=SCALE_FACTORS_ADD_INFO if args.use_add_info else None,
)
print("Fitting kmeans tokenizer...")
kmeans_tokenizer.fit(
    training_x,
    add_info=training_add_info if args.use_add_info else None,
    mask=mask,
)

# save and load the kmeans model
filepath = f"{OUTPUT_DIR}/{UNIQUE_ID}.pth"
print(f"Saving kmeans model to {filepath}")
# save the model on CPU to ensure compatibility
kmeans_tokenizer = kmeans_tokenizer.kmeans.to("cpu")
torch.save(kmeans_tokenizer, filepath)

# Load the model back
with open(filepath, "rb") as f:
    kmeans_tokenizer = torch.load(f, weights_only=False)
    # move back to device
    kmeans_tokenizer = kmeans_tokenizer.to(device)

feature_names = {
    "eta": "Particle $\\Delta\\eta$",
    "phi": "Particle $\\Delta\\phi$",
    "log_pt": "$\\log(p_{T})$",
    "log_E": "$\\log(E)$",
}

if args.use_add_info:
    feature_names.update(
        {
            "d0val": "$d_{0}$",
            "d0err": "$d_{0}^{err}$",
            "dzval": "$d_{z}$",
            "dzerr": "$d_{z}^{err}$",
        }
    )


def combine_features(x, add_info):
    if add_info is None:
        return x
    return torch.cat([x, add_info], dim=-1)


training_x = training_x[:NUM_PLOTS_SAMPLES]
training_add_info = training_add_info[:NUM_PLOTS_SAMPLES]
mask = mask[:NUM_PLOTS_SAMPLES]

# Original
fullres_ak = np_to_ak(
    combine_features(training_x, training_add_info).cpu().numpy(),
    names=feature_names,
    mask=mask.cpu().numpy(),
)

# Tokenize data (returns x and optional add_info parts)
tokenized_x, tokenized_add_info = kmeans_tokenizer.transform(
    training_x,
    add_info=training_add_info if args.use_add_info else None,
)
tokenized_x[~mask] = 0
if args.use_add_info:
    tokenized_add_info[~mask] = 0
tokenized = combine_features(tokenized_x, tokenized_add_info)
tokenized_ak = np_to_ak(
    tokenized.cpu().numpy(), names=feature_names, mask=mask.cpu().numpy()
)

fig, axarr = plot_features(
    {
        "JetClass particles": fullres_ak,
        "Tokenized": tokenized_ak,
    },
    names=feature_names,
    ax_rows=3 if args.use_add_info else 2,
)
fig.suptitle(
    f"Particle features ({NUM_TRAINING_SAMPLES} training jets - {NUM_PLOTS_SAMPLES} plotted jets)"
)
fig.tight_layout()
fig.savefig(
    f"{PLOT_DIR}/original_vs_tokenized_particle_features.pdf", **SAVE_FIG_KWARGS
)


helper_kwargs = dict(
    field_name_pt="log_pt",
    field_name_eta="eta",
    field_name_phi="phi",
    pt_is_log=True,
)
p4s_original = p4s_from_ptetaphimass(fullres_ak, **helper_kwargs)
p4s_original_sum = ak.sum(p4s_original, axis=1)
p4s_tokenized = p4s_from_ptetaphimass(tokenized_ak, **helper_kwargs)
p4s_tokenized_sum = ak.sum(p4s_tokenized, axis=1)

fig, axarr = plot_features(
    {
        "Original jets": p4s_original_sum,
        "Tokenized jets": p4s_tokenized_sum,
    },
    names={
        "pt": "$p_{T}$",
        "eta": "$\\eta$",
        "phi": "$\\phi$",
        "mass": "$m$",
    },
    bins_dict={
        "mass": np.linspace(0, 300, 100),
    },
    flatten=False,
)
fig.suptitle(
    f"Jet features ({NUM_TRAINING_SAMPLES} training jets - {NUM_PLOTS_SAMPLES} plotted jets)"
)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/original_vs_tokenized_jet_features.pdf", **SAVE_FIG_KWARGS)

# plot the resolution on jet features
resolution = ak.Array(
    {
        "pt_resolution": p4s_tokenized_sum.pt - p4s_original_sum.pt,
        "eta_resolution": p4s_tokenized_sum.eta - p4s_original_sum.eta,
        "phi_resolution": p4s_tokenized_sum.phi - p4s_original_sum.phi,
        "mass_resolution": p4s_tokenized_sum.mass - p4s_original_sum.mass,
    }
)
fig, axarr = plot_features(
    {"Tokenized - Original": resolution},
    names={
        "pt_resolution": "$\\Delta p_{T}$",
        "eta_resolution": "$\\Delta \\eta$",
        "phi_resolution": "$\\Delta \\phi$",
        "mass_resolution": "$\\Delta m$",
    },
    bins_dict={
        "pt_resolution": np.linspace(-50, 50, 100),
        "eta_resolution": np.linspace(-0.5, 0.5, 100),
        "phi_resolution": np.linspace(-0.5, 0.5, 100),
        "mass_resolution": np.linspace(-100, 100, 100),
    },
    flatten=False,
)
fig.suptitle(
    f"Jet feature resolutions ({NUM_TRAINING_SAMPLES} training jets - {NUM_PLOTS_SAMPLES} plotted jets)"
)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/jet_feature_resolutions.pdf", **SAVE_FIG_KWARGS)

# pairplot comparing original and tokenized features
pairplot_fig = plot_features_pairplot(
    {
        "Original": fullres_ak[:100],
        "Tokenized (KMeans centroids)": tokenized_ak[:100],
    },
    names=feature_names,
    figsize=(10, 10),
    y_labels_xcoord=-0.4,  # align y-axis labels better
)
pairplot_fig.figure.tight_layout()
pairplot_fig.figure.savefig(
    f"{PLOT_DIR}/original_vs_tokenized_features_pairplot.png", **SAVE_FIG_KWARGS
)

# pairplots for each feature comparing original and tokenized
os.makedirs(f"{PLOT_DIR}/per_feature", exist_ok=True)
for field in fullres_ak.fields:
    pairplot_fig = plot_features_pairplot(
        {
            "Original_vs_Tokenized": ak.Array(
                {
                    f"{field}_original": fullres_ak[field],
                    f"{field}_tokenized": tokenized_ak[field],
                }
            )
        },
        names={
            f"{field}_original": f"Original {feature_names[field]}",
            f"{field}_tokenized": f"Tokenized {feature_names[field]}",
        },
    )
    pairplot_fig.figure.suptitle(
        f"Jet features ({NUM_TRAINING_SAMPLES} training jets - {NUM_PLOTS_SAMPLES} plotted jets)",
        y=1.02,
    )
    pairplot_fig.figure.tight_layout()
    pairplot_fig.figure.savefig(
        f"{PLOT_DIR}/per_feature/original_vs_tokenized_{field}_pairplot.png",
        **SAVE_FIG_KWARGS,
    )
