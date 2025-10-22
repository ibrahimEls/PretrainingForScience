import sys

# update pythonpath
sys.path.append("/global/homes/j/jobirk/repositories/OmniLearnLightning_dev")
sys.path.append("/global/homes/j/jobirk/repositories/gabbro")

import os

import awkward as ak
import fast_pytorch_kmeans
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

NUM_TRAINING_SAMPLES = 10_000
NUM_MAX_POINTS_PER_JET = 100
NUM_FEATURES_X = 4
NUM_FEATURES_ADD = 4
CODEBOOK_SIZE = 20_000
# Separate scale factors for x and add_info
SCALE_FACTORS_X = torch.tensor(
    [
        0.13,  # eta
        0.13,  # phi
        1.2,  # log pt
        1.2,  # log energy
    ]
)
SCALE_FACTORS_ADD_INFO = torch.tensor(
    [
        0.1,  # d0val
        1.0,  # d0err
        0.1,  # dzval
        1.0,  # dzerr
    ]
)
SAVE_FIG_KWARGS = dict(dpi=150, bbox_inches="tight")
# filepath should be unique: time stamp and bigram
UNIQUE_ID = f"kmeans_model_{get_bigram(add_timestamp=True, timestamp_first=True)}_{CODEBOOK_SIZE}_codes"
OUTPUT_DIR = f"/pscratch/sd/j/jobirk/testing/{UNIQUE_ID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DIR = f"{OUTPUT_DIR}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

datamodule = PETDataModule(
    dataset="jetclass",
    path="/pscratch/sd/j/jobirk/omnilearn_datasets/",
    batch_size=1_000,
    num_samples=NUM_TRAINING_SAMPLES,
    train_tag="equal_class",
    use_pid=True,
    use_add=True,
)
datamodule.setup("fit")

dataloader = datamodule.train_dataloader()

training_x = torch.zeros(
    (NUM_TRAINING_SAMPLES, NUM_MAX_POINTS_PER_JET, NUM_FEATURES_X), device="cpu"
)
training_pid = torch.zeros((NUM_TRAINING_SAMPLES, NUM_MAX_POINTS_PER_JET), device="cpu")
training_add_info = torch.zeros(
    (NUM_TRAINING_SAMPLES, NUM_MAX_POINTS_PER_JET, NUM_FEATURES_ADD), device="cpu"
)
training_y = torch.zeros((NUM_TRAINING_SAMPLES,), dtype=torch.long, device="cpu")

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

    training_x[start_idx:end_idx, :fill_until] = batch["X"][:batch_size, :fill_until]
    if "pid" in batch:
        training_pid[start_idx:end_idx, :fill_until] = batch["pid"][
            :batch_size, :fill_until
        ]
    if "add_info" in batch:
        training_add_info[start_idx:end_idx, :fill_until] = batch["add_info"][
            :batch_size, :fill_until
        ]
    training_y[start_idx:end_idx] = batch["y"][:batch_size]
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
    scale_factors_add_info=SCALE_FACTORS_ADD_INFO,
)
print("Fitting kmeans tokenizer...")
kmeans_tokenizer.fit(
    training_x.cpu(), add_info=training_add_info.cpu(), mask=mask.cpu()
)

# save and load the kmeans model
filepath = f"{OUTPUT_DIR}/{UNIQUE_ID}.pth"
print(f"Saving kmeans model to {filepath}")
torch.save(kmeans_tokenizer, filepath)

with torch.serialization.safe_globals([fast_pytorch_kmeans.kmeans.KMeans, getattr]):
    with open(filepath, "rb") as f:
        kmeans_tokenizer = torch.load(f, weights_only=False)

feature_names = {
    "eta": "Particle $\\Delta\\eta$",
    "phi": "Particle $\\Delta\\phi$",
    "log_pt": "$\\log(p_{T})$",
    "log_E": "$\\log(E)$",
    "PID": "Particle ID",
    "d0val": "$d_{0}$",
    "d0err": "$d_{0}^{err}$",
    "dzval": "$d_{z}$",
    "dzerr": "$d_{z}^{err}$",
}


def combine_features(x, pid, add_info):
    return torch.cat([x, pid.unsqueeze(-1), add_info], dim=-1)


# Original
batch_ak = np_to_ak(
    combine_features(training_x, training_pid, training_add_info).cpu().numpy(),
    names=feature_names,
    mask=mask.cpu().numpy(),
)

# Tokenize data (returns x and add_info parts)
batch_tokenized_x, batch_tokenized_add_info = kmeans_tokenizer.transform(
    training_x, add_info=training_add_info
)
batch_tokenized_x[~mask] = 0
batch_tokenized_add_info[~mask] = 0
batch_tokenized = combine_features(
    batch_tokenized_x, training_pid, batch_tokenized_add_info
)
batch_tokenized_ak = np_to_ak(
    batch_tokenized.cpu().numpy(), names=feature_names, mask=mask.cpu().numpy()
)

fig, axarr = plot_features(
    {
        "JetClass particles": batch_ak,
        "Tokenized": batch_tokenized_ak,
    },
    names=feature_names,
    ax_rows=3,
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
p4s_original = p4s_from_ptetaphimass(batch_ak, **helper_kwargs)
p4s_original_sum = ak.sum(p4s_original, axis=1)
p4s_tokenized = p4s_from_ptetaphimass(batch_tokenized_ak, **helper_kwargs)
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
fig.suptitle(f"Jet features ({NUM_TRAINING_SAMPLES} training jets)")
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/original_vs_tokenized_jet_features.pdf", **SAVE_FIG_KWARGS)

# pairplot comparing original and tokenized features
pairplot_fig = plot_features_pairplot(
    {
        "Original": batch_ak[:100],
        "Tokenized (KMeans centroids)": batch_tokenized_ak[:100],
    },
    names=feature_names,
    figsize=(10, 10),
    y_labels_xcoord=-0.4,  # align y-axis labels better
)
fig.tight_layout()
fig.savefig(
    f"{PLOT_DIR}/original_vs_tokenized_features_pairplot.pdf", **SAVE_FIG_KWARGS
)

# pairplots for each feature comparing original and tokenized
os.makedirs(f"{PLOT_DIR}/per_feature", exist_ok=True)
for field in batch_ak.fields:
    pairplot_fig = plot_features_pairplot(
        {
            "Original_vs_Tokenized": ak.Array(
                {
                    f"{field}_original": batch_ak[field],
                    f"{field}_tokenized": batch_tokenized_ak[field],
                }
            )
        },
        names={
            f"{field}_original": f"Original {feature_names[field]}",
            f"{field}_tokenized": f"Tokenized {feature_names[field]}",
        },
    )
    pairplot_fig.tight_layout()
    pairplot_fig.savefig(
        f"{PLOT_DIR}/per_feature/original_vs_tokenized_{field}_pairplot.pdf",
        **SAVE_FIG_KWARGS,
    )
