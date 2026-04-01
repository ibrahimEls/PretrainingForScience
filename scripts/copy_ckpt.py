import argparse
import json
import shutil
from pathlib import Path

import yaml

DATASET_SIZE_FOLDER_NAME = {
    "100000": "100k",
    "1000000": "1M",
    "10000000": "10M",
    "-1": "100M",
}
METHOD_PREFIX = {
    "mpm": "mpm_only",
    "mpmregress": "mpmregress_only",
    "generator": "gen_only",
    "classifier": "class_only",
    "classifier+generator": "class_gen",
    "classifier+mpm": "class_mpm",
    "classifier+mpmregress": "class_mpmregress",
    "generator+mpm": "gen_mpm",
    "generator+mpmregress": "gen_mpmregress",
    "pretrain": "pretrain",
    "pretrainregress": "pretrainregress",
    # MPM with perturbation loss (when MPM and classification are used jointly)
    "classifier+mpm_perturb": "class_mpm_perturb",
    "classifier+mpmregress_perturb": "class_mpmregress_perturb",
    "pretrain_perturb": "pretrain_perturb",
    "pretrainregress_perturb": "pretrainregress_perturb",
}
TARGET_DIR = "/global/cfs/cdirs/m3246/Omnilearned_Study/Model_Checkpoints"
SHARED_GROUP_ID = "m3246"


def main():
    parser = argparse.ArgumentParser(description="Move checkpoint files")
    parser.add_argument(
        "--target_json",
        type=Path,
        required=True,
        help="Path to the target JSON file",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the run directory containing checkpoints",
    )
    args = parser.parse_args()
    target_json = args.target_json

    ckpt_src = args.run_dir / "checkpoints"

    best_ckpt = None
    best_loss = float("inf")
    model_size = None
    method = None
    dataset_size = None
    best_step = 0
    ckpts_with_val_loss = []

    # loop over the files in the checkpoints directory
    for ckpt_file in ckpt_src.iterdir():
        if "val_loss" not in ckpt_file.name:
            continue
        # extract the val_loss value from the filename
        # filename format: _<modelsize>_<method>_<dataset_type>_dataset_<num_samples>_<username>-epoch=<epoch>-step=<step>-val_loss=<val_loss>-train_loss_steps=<train_loss_steps>.ckpt
        parts = ckpt_file.name.split("-")
        val_loss_part = [p for p in parts if p.startswith("val_loss=")][0]
        val_loss = float(val_loss_part.split("=")[1].replace(".ckpt", ""))
        ckpts_with_val_loss.append((ckpt_file.name.split("/")[-1], val_loss))
        if val_loss <= best_loss:
            if val_loss == best_loss:
                print(
                    f"Warning: Found two checkpoints with the same val_loss={val_loss}: "
                )
                for ckpt, loss in ckpts_with_val_loss:
                    if loss == best_loss:
                        print(f" - {ckpt}")
            # extract model_size, method, dataset_size from filename
            name_parts = ckpt_file.name.split("_")
            model_size = name_parts[1]
            method = name_parts[2]
            if "mpm" in method or method == "pretrain":
                # check if tokenizer_ckpt is None by looking at hparams.yaml in run_dir
                hparams_file = args.run_dir / "hparams.yaml"
                with open(hparams_file, "r") as f:
                    hparams = yaml.safe_load(f)
                if hparams.get("tokenizer_ckpt", None) is None:
                    method += "regress"
                if hparams.get("use_perturbed_loss_terms", False) and (
                    "classifier" in method or "pretrain" in method
                ):
                    method += "_perturb"

            dataset_size = name_parts[5]
            step_part = [p for p in parts if p.startswith("step=")][0]
            current_step = int(step_part.split("=")[1])
            if val_loss == best_loss and current_step < best_step:
                continue
            best_step = current_step
            best_loss = val_loss
            best_ckpt = ckpt_file

    for ckpt, loss in ckpts_with_val_loss:
        str_to_print = f"Checkpoint: {ckpt} with val_loss={loss}"
        if loss == best_loss:
            str_to_print += "  <-- best"
        print(str_to_print)

    print(f"Best checkpoint: {best_ckpt} with val_loss={best_loss}")

    # copy to target directory
    dest_path = (
        Path(TARGET_DIR)
        / model_size.capitalize()
        / DATASET_SIZE_FOLDER_NAME[dataset_size]
    )
    dest_path.mkdir(parents=True, exist_ok=True)

    dest_ckpt_path = (
        dest_path / f"{METHOD_PREFIX[method]}_val_loss={best_loss:.4f}.ckpt"
    )

    print(f"Copying to {dest_ckpt_path}")

    # ask for confirmation before copying
    response = input("Confirm with y\n")
    if response.lower() != "y":
        print("Aborting.")
        return

    # if exists, ask for confirmation to overwrite
    if dest_ckpt_path.exists():
        response = input(f"\n{dest_ckpt_path} already exists. Overwrite? (y/n): \n")
        if response.lower() != "y":
            print("Aborting copy.")
            return

    shutil.copy(best_ckpt, dest_ckpt_path)
    # set permissions to shared group
    shutil.chown(dest_ckpt_path, group=SHARED_GROUP_ID)

    # open the json file and read its contents
    # structure: {model_size: {method: {dataset_size: path}}}
    with open(target_json, "r") as f:
        json_data = json.load(f)

    # write the path to the json file
    json_data[model_size][method][dataset_size] = str(dest_ckpt_path)
    with open(target_json, "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    main()
