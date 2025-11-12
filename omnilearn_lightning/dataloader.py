import os
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import torch
from omnilearned.dataloader import collate_point_cloud, download_h5_files, get_url
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
        use_cond=False,
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
        self.use_cond = use_cond

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
                use_cond=self.use_cond,
                **loading_kwargs,
            )
            if self.load_val:
                self.val_dataset = load_data(
                    self.dataset,
                    dataset_type="val",
                    shuffle_indices=self.shuffle_val_test_indices,
                    seed_for_shuffling=self.seed_for_initial_shuffling,
                    use_cond=self.use_cond,
                    **loading_kwargs,
                )
        elif stage == "test":
            self.test_dataset = load_data(
                self.dataset,
                dataset_type="test",
                shuffle_indices=self.shuffle_val_test_indices,
                seed_for_shuffling=self.seed_for_initial_shuffling,
                use_cond=self.use_cond,
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
        clip_inputs=False,
        use_cond=False,
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
        self.clip_inputs = clip_inputs
        self.use_cond = use_cond

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
        if self.clip_inputs:
            # Enforce particles to be inside R=0.8 and pT > 0.5 MeV
            mask_part = (torch.hypot(sample["X"][:, 0], sample["X"][:, 1]) < 0.8) & (
                sample["X"][:, 2] > 0.0
            )
            sample["X"][:, 3] = np.clip(
                sample["X"][:, 3], a_min=sample["X"][:, 2], a_max=None
            )
            sample["X"] = sample["X"] * mask_part.unsqueeze(-1).float()

        label = f["pid"][sample_idx]
        sample["y"] = torch.tensor(label - self.label_shift, dtype=torch.int64)
        if "global" in f and self.use_cond:
            sample["cond"] = torch.tensor(f["global"][sample_idx], dtype=torch.float32)

        if self.use_pid:
            sample["pid"] = sample["X"][:, self.pid_idx].int()
            sample["X"] = torch.cat(
                (sample["X"][:, : self.pid_idx], sample["X"][:, self.pid_idx + 1 :]),
                dim=1,
            )
        if self.use_add:
            # Assume any additional info appears last
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
    use_cond=False,
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
