"""Utils that should have as few dependencies as possible, to work in lightweight environments."""

import json
from pathlib import Path

import yaml


def get_all_gen_finetuned_ckpt_paths_from_json(
    finetune_paths_json_path: str, which: str
):
    """Extract all checkpoint paths for a given 'which' (src or dest) from the finetune paths JSON."""

    if which not in ["src", "dest"]:
        raise ValueError(
            f"Invalid value for 'which': {which}. Must be 'src' or 'dest'."
        )

    with open(finetune_paths_json_path, "r") as f:
        finetune_paths = json.load(f)
    ckpt_paths = []
    for finetune_datasetsize_id in finetune_paths:
        for model_size_id in finetune_paths[finetune_datasetsize_id]:
            for pretrain_mode_id in finetune_paths[finetune_datasetsize_id][
                model_size_id
            ]:
                for pretrain_datasetsize_id in finetune_paths[finetune_datasetsize_id][
                    model_size_id
                ][pretrain_mode_id]:
                    runs = finetune_paths[finetune_datasetsize_id][model_size_id][
                        pretrain_mode_id
                    ][pretrain_datasetsize_id][0]["runs"]
                    for run in runs:
                        ckpt_path = run[f"best_ckpt_{which}"]
                        ckpt_paths.append(ckpt_path)

    return ckpt_paths


def get_pretrained_ckpts(
    pretrained_ckpts_json_path: str,
    modes: list[str],
    pretrain_datasetsize_ids: list[str],
    model_sizes: list[str],
):
    """Extract all pretrained checkpoint paths from the pretrained ckpts JSON, filtered by the given criteria."""

    with open(pretrained_ckpts_json_path, "r") as f:
        pretrained_ckpts = json.load(f)

    ckpt_paths = []
    for model_size in pretrained_ckpts:
        if model_size not in model_sizes:
            continue
        for mode in pretrained_ckpts[model_size]:
            if mode not in modes:
                continue
            for pretrain_datasetsize_id in pretrained_ckpts[model_size][mode]:
                if pretrain_datasetsize_id not in pretrain_datasetsize_ids:
                    continue
                ckpt_path = pretrained_ckpts[model_size][mode][pretrain_datasetsize_id]
                ckpt_paths.append(ckpt_path)

    return ckpt_paths


def get_ckpts_vs_epoch(yaml_file_path: str, every_n_epochs: int = 50):
    # read yaml file and extract run directories
    with open(yaml_file_path, "r") as f:
        run_dirs_list = yaml.safe_load(f)
        # print(run_dirs_list)
    ckpts_list = []
    for run_dir in run_dirs_list:
        # print(run_dir)
        run_dir = Path(run_dir)
        ckpt_dir = run_dir / "checkpoints"
        all_ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
        for ckpt_file in all_ckpt_files:
            if "epoch=" not in ckpt_file.stem:
                continue
            # extract epoch number from ckpt file name
            # epoch is encoded in str as e.g. "...-epoch=001431-
            epoch_str = ckpt_file.stem.split("-epoch=")[-1].split("-")[0]

            epoch_num = int(epoch_str)
            if epoch_num % every_n_epochs == 0:
                # print(ckpt_file)
                ckpts_list.append({epoch_num: str(ckpt_file)})
    return ckpts_list
