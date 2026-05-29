"""Utils that should have as few dependencies as possible, to work in lightweight environments."""

import json
from pathlib import Path

import yaml

# Jet type definitions
jet_types_dict_jetclass = {
    0: {"label": "QCD", "tex_label": "$q/g$", "color": "C0"},
    1: {"label": "Hbb", "tex_label": "$H\\rightarrow b\\bar{b}$", "color": "C1"},
    2: {"label": "Hcc", "tex_label": "$H\\rightarrow c\\bar{c}$", "color": "C2"},
    3: {"label": "Hgg", "tex_label": "$H\\rightarrow gg$", "color": "C3"},
    4: {"label": "H4q", "tex_label": "$H\\rightarrow 4q$", "color": "C4"},
    5: {"label": "Hlnuqq", "tex_label": "$H\\rightarrow \\ell\\nu qq'$", "color": "C5"},
    6: {"label": "Zqq", "tex_label": "$Z\\rightarrow q\\bar{q}$", "color": "C6"},
    7: {"label": "Wqq", "tex_label": "$W\\rightarrow qq'$", "color": "C7"},
    8: {"label": "Tbqq", "tex_label": "$t\\rightarrow bqq'$", "color": "C8"},
    9: {"label": "Tbl", "tex_label": "$t\\rightarrow b\\ell\\nu$", "color": "C9"},
}

jet_types_dict_jetnet = {
    0: {"label": "gluon", "tex_label": "gluon", "color": "C0"},
    1: {"label": "light_quark", "tex_label": "light quark", "color": "C1"},
    2: {"label": "top", "tex_label": "top", "color": "C2"},
    3: {"label": "W", "tex_label": "$W$", "color": "C3"},
    4: {"label": "Z", "tex_label": "$Z$", "color": "C4"},
}


def get_numerical_label_from_name(jet_type_name: str, dataset_type: str) -> int:
    if dataset_type == "jetclass":
        jet_types_dict = jet_types_dict_jetclass
    elif "jetnet" in dataset_type:
        jet_types_dict = jet_types_dict_jetnet
    else:
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}'. Must be 'jetclass' or 'jetnet'."
        )
    for label, info in jet_types_dict.items():
        if info["label"] == jet_type_name:
            return label
    raise ValueError(
        f"Jet type name '{jet_type_name}' not found in dataset '{dataset_type}'."
    )


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
    return_with_info: bool = False,
):
    """Extract all pretrained checkpoint paths from the pretrained ckpts JSON, filtered by the given criteria."""

    with open(pretrained_ckpts_json_path, "r") as f:
        pretrained_ckpts = json.load(f)

    ckpt_paths = [] if not return_with_info else {}

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
                if not ckpt_path.endswith(".ckpt"):
                    print(
                        f"Warning: Skipping invalid checkpoint path '{ckpt_path}' for model_size='{model_size}', mode='{mode}', pretrain_datasetsize_id='{pretrain_datasetsize_id}'"
                    )
                    continue
                # check if path exists
                if not Path(ckpt_path).is_file():
                    print(f"File doesn't exist: {ckpt_path}")
                    continue
                try:
                    with open(ckpt_path) as f:
                        pass
                except Exception as exc:
                    print(f"Error opening ckpt path: {ckpt_path} -- {exc}")
                if return_with_info:
                    ckpt_paths[ckpt_path] = {
                        "mode": mode,
                        "model_size": model_size,
                        "pretrain_dataset_size": pretrain_datasetsize_id,
                    }
                else:
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
