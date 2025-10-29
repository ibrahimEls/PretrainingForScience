import os
import re
from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urljoin

import h5py
import numpy as np
import requests
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class PETDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        use_pid: bool = False,
        use_add: bool = False,
        num_samples=-1,
        train_tag: str = "train",
        shuffle_train_indices: bool = False,
        shuffle_val_test_indices: bool = False,
        seed_for_initial_shuffling: int = None,
        load_val=False,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_pid = use_pid
        self.use_add = use_add
        self.num_samples = num_samples
        self.train_tag = train_tag
        self.shuffle_train_indices = shuffle_train_indices
        self.shuffle_val_test_indices = shuffle_val_test_indices
        self.seed_for_initial_shuffling = seed_for_initial_shuffling
        self.load_val = load_val

    def setup(self, stage=None):
        """Called at the beginning of fit/test to set up data."""

        print(f"Setting up datamodule for stage: {stage}")

        loading_kwargs = dict(
            use_pid=self.use_pid,
            use_add=self.use_add,
            path=self.path,
            batch=self.batch_size,
            num_workers=self.num_workers,
            rank=0,  # For single-GPU or single-node usage
            size=1,
            limit_num_samples=self.num_samples,
        )
        if stage == "fit" or stage is None or stage == "validate":
            self.train_dataset = load_data(
                self.dataset,
                dataset_type=self.train_tag,
                shuffle_indices=self.shuffle_train_indices,
                seed_for_shuffling=self.seed_for_initial_shuffling,
                **loading_kwargs,
            )
            if self.load_val:
                self.val_dataset = load_data(
                    self.dataset,
                    dataset_type="val",
                    shuffle_indices=self.shuffle_val_test_indices,
                    seed_for_shuffling=self.seed_for_initial_shuffling,
                    **loading_kwargs,
                )
        elif stage == "test":
            self.test_dataset = load_data(
                self.dataset,
                dataset_type="test",
                shuffle_indices=self.shuffle_val_test_indices,
                seed_for_shuffling=self.seed_for_initial_shuffling,
                **loading_kwargs,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset


def collate_point_cloud(batch):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]

    # Stack once to avoid repeated slicing
    point_clouds = torch.stack(batch_X)  # (B, N, F)
    labels = torch.stack(batch_y)  # (B, num_classes)

    # Use validity mask based on feature index 2
    valid_mask = point_clouds[:, :, 2] != 0
    max_particles = valid_mask.sum(dim=1).max().item()

    # Truncate point clouds
    truncated_X = point_clouds[:, :max_particles, :]  # (B, M, F)
    result = {"X": truncated_X, "y": labels}

    # Handle optional fields in a loop to reduce code duplication
    optional_fields = ["cond", "pid", "add_info"]
    for field in optional_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            # Truncate if it's sequence-like (i.e., has 2 or more dims)
            if stacked.dim() >= 2 and stacked.shape[1] >= max_particles:
                stacked = stacked[:, :max_particles]
            result[field] = stacked
        else:
            result[field] = None

    return result


def get_url(dataset_name, dataset_type, base_url="https://portal.nersc.gov/cfs/m4567/"):
    url = f"{base_url}/{dataset_name}/{dataset_type}/"
    try:
        requests.head(url, allow_redirects=True, timeout=5)
        return url
    except requests.RequestException:
        return None


def download_h5_files(base_url, destination_folder):
    """
    Downloads all .h5 files from the specified directory URL.

    Args:
        base_url (str): The base URL of the directory containing the .h5 files.
        destination_folder (str): The local folder to save the downloaded files.
    """

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}")
        return

    file_links = re.findall(r'href="([^"]+\.h5)"', response.text)

    for file_name in file_links:
        file_url = urljoin(base_url, file_name)
        file_path = os.path.join(destination_folder, file_name)

        print(f"Downloading {file_url} to {file_path}")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_name}")


import h5py
import torch
from torch.utils.data import Dataset


class HEPDataset(Dataset):
    def __init__(
        self,
        file_paths,
        file_indices,
        use_pid=False,
        pid_idx=-1,
        use_add=False,
        num_add=4,
        label_shift=0,
        num_samples=-1,  # New argument to limit the number of samples
    ):
        """
        Args:
            file_paths (list): List of file paths.
            file_indices (list): List of (file_idx, sample_idx) tuples.
            use_pid (bool): Flag to select if PID information is used during training.
            use_add (bool): Flag to select if additional information besides kinematics are used.
            num_samples (int): Number of samples to include; if -1, include all samples.
            pid_idx (int): The index for the PID column in the data.
            num_add (int): Number of additional features to use if `use_add` is True.
            label_shift (int): Value to subtract from the label.
        """
        self.use_pid = use_pid
        self.use_add = use_add

        self.pid_idx = pid_idx
        self.num_add = num_add
        self.label_shift = label_shift
        self.num_samples = num_samples

        self.file_paths = file_paths
        self._file_cache = {}  # Lazy cache for open h5py.File handles
        self.file_indices = file_indices

        # Limit the number of samples if num_samples is not -1.
        if self.num_samples != -1:
            self.limit_samples(self.num_samples)

    def limit_samples(self, num_samples):
        """
        Limits the dataset to the first num_samples entries.

        Args:
            num_samples (int): The maximum number of samples to keep.
        """
        self.file_indices = self.file_indices[:num_samples]

    def __len__(self):
        return len(self.file_indices)

    def _get_file(self, file_idx):
        # Retrieve file handle from cache or open if not already opened.
        if file_idx not in self._file_cache:
            file_path = self.file_paths[file_idx]
            self._file_cache[file_idx] = h5py.File(file_path, "r")
        return self._file_cache[file_idx]

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_indices[idx]
        f = self._get_file(file_idx)

        sample = {}
        sample["X"] = torch.tensor(f["data"][sample_idx], dtype=torch.float32)
        label = f["pid"][sample_idx]
        sample["y"] = torch.tensor(label - self.label_shift, dtype=torch.int64)

        if "global" in f:
            sample["cond"] = torch.tensor(f["global"][sample_idx], dtype=torch.float32)

        if self.use_pid:
            sample["pid"] = sample["X"][:, self.pid_idx].int()
            sample["X"] = torch.cat(
                (sample["X"][:, : self.pid_idx], sample["X"][:, self.pid_idx + 1 :]),
                dim=1,
            )
        if self.use_add:
            # Assume any additional info appears last.
            sample["add_info"] = sample["X"][:, -self.num_add :]
            sample["X"] = sample["X"][:, : -self.num_add]
        return sample

    def __del__(self):
        # Clean up: close all cached file handles.
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception as e:
                print(f"Error closing file: {e}")


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    distributed=True,
    use_pid=False,
    pid_idx=4,
    use_add=False,
    num_add=4,
    num_workers=16,
    rank=0,
    size=1,
    limit_num_samples=-1,
    shuffle_indices: bool = False,
    seed_for_shuffling: int = None,
):
    supported_datasets = [
        "top",
        "qg",
        "pretrain",
        "atlas",
        "aspen",
        "jetclass",
        "jetclass2",
        "h1",
        "toy",
    ]
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}."
        )

    if dataset_name == "pretrain":
        names = ["atlas", "aspen", "jetclass", "h1"]
    else:
        names = [dataset_name]

    dataset_paths = [os.path.join(path, name, dataset_type) for name in names]

    file_list = []
    file_indices = []
    index_shift = 0
    for iname, dataset_path in enumerate(dataset_paths):
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        if not any(dataset_path.iterdir()):
            print(f"Fetching download url for dataset {names[iname]}")
            url = get_url(names[iname], dataset_type)
            if url is None:
                raise ValueError(f"No download URL found for dataset '{dataset_name}'.")
            download_h5_files(url, dataset_path)

        h5_files = list(dataset_path.glob("*.h5"))
        file_list.extend(map(str, h5_files))  # Convert to string paths

        index_file = dataset_path / "file_index.npy"
        if index_file.is_file():
            indices = np.load(index_file, mmap_mode="r")[rank::size]
            if shuffle_indices:
                # shuffle indices once here to ensure that jet types are mixed
                rng = np.random.default_rng(seed=seed_for_shuffling)
                perm = rng.permutation(len(indices))
                indices = indices[perm]
            file_indices.extend(
                (file_idx + index_shift, sample_idx) for file_idx, sample_idx in indices
            )
            index_shift += len(h5_files)
            if index_shift >= limit_num_samples and limit_num_samples != -1:
                break

        else:
            print(f"Creating index list for dataset {names[iname]}")
            file_indices = []
            # Precompute indices for efficient access
            for file_idx, path in enumerate(h5_files):
                try:
                    with h5py.File(path, "r") as f:
                        num_samples = len(f["data"])
                        file_indices.extend([(file_idx, i) for i in range(num_samples)])
                except Exception as e:
                    print(f"ERROR: File {path} is likely corrupted: {e}")
            np.save(index_file, np.array(file_indices, dtype=np.int32))

    # Shift labels if they are not used for pretrain
    label_shift = {"jetclass": 2, "aspen": 12, "jetclass2": 13}

    data = HEPDataset(
        file_list,
        file_indices,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        label_shift=label_shift.get(dataset_name, 0),
        num_samples=limit_num_samples,
    )

    loader = DataLoader(
        data,
        batch_size=batch,
        pin_memory=torch.cuda.is_available(),
        shuffle=dataset_type == "train",
        sampler=None,
        persistent_workers=True,
        # sampler=(
        #     DistributedSampler(data, shuffle=dataset_type == "train")
        #     if distributed
        #     else None
        # ),
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_point_cloud,
    )
    return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="top",
        help="Dataset name to download",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="./",
        help="Folder to save the dataset",
    )
    args = parser.parse_args()

    for tag in ["train", "test", "val"]:
        load_data(args.dataset, args.folder, dataset_type=tag, distributed=False)
