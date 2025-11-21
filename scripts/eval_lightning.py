#!/usr/bin/env python3
import argparse
import json
import os

from omnilearn_lightning.eval_tasks import eval_top_tagging


def load_json_or_empty(path):
    """Load JSON if it exists and is valid, otherwise return {}."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] Metrics JSON at {path} is not valid JSON; starting from empty.")
        return {}


def ensure_nested(d, keys):
    """
    Ensure nested dict structure d[keys[0]][keys[1]]...[keys[-1]] exists.
    Returns the deepest dict.
    """
    cur = d
    for k in keys:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur


def main():
    parser = argparse.ArgumentParser(description="Evaluate top-tagging ckpts from JSON")

    # ------- paths to JSONs -------
    parser.add_argument(
        "--model_paths_json",
        type=str,
        required=True,
        help="JSON file with model checkpoint paths (the big structure you showed).",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        required=True,
        help="JSON file to store evaluation metrics (same structure as model_paths_json).",
    )

    # ------- basic eval args (same as your example, for eval_top_tagging) -------
    parser.add_argument(
        "--outdir",
        type=str,
        default="/pscratch/sd/i/ibrahime/checkpoints/",
        help="Output directory",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/pscratch/sd/i/ibrahime/datasets/",
        help="Path to dataset",
    )
    parser.add_argument(
        "--dataset", type=str, default="pretrain", help="Name of the dataset to load"
    )
    parser.add_argument(
        "--dataset_size", type=int, default=-1, help="Number of datapoints"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Number of nodes for training"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Number of DataLoader workers"
    )

    # Model hyper-parameters
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=201)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_transformers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--mlp_drop", type=float, default=0.0)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--feature_drop", type=float, default=0.1)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model_size", type=str, default="micro")
    parser.add_argument("--task", type=str)

    # Training hyperparams (used inside eval_top_tagging as needed)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=10)
    parser.add_argument("--b1", type=float, default=0.95, help="Beta1 for optimizer")
    parser.add_argument("--b2", type=float, default=0.99, help="Beta2 for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_clip", action="store_true", help="Use clip loss or not")
    parser.add_argument(
        "--use_amp", action="store_true", help="Use 16-bit precision training (AMP)"
    )

    # Additional features
    parser.add_argument("--use_pid", action="store_true")
    parser.add_argument("--use_add", action="store_true")

    # GPU selection
    parser.add_argument("--gpuID", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()

    # --------- load JSONs ----------
    with open(args.model_paths_json, "r") as f:
        model_paths = json.load(f)

    metrics = load_json_or_empty(args.metrics_json)

    # --------- main loop over structure ----------
    # structure:
    #   outer:  "1M-Top", "1k-Top", "10k-Top", "100k-Top"
    #   next:   "micro" / "small" / "medium"
    #   next:   "classifier", "generator", "mpm", ...
    #   next:   "100000", "1000000", "10000000", "-1"  -> path string

    if args.task == "top_tagging":
        for setup_name, setup_dict in model_paths.items():
            for model_size, size_dict in setup_dict.items():
                # set model size in args for this branch
                args.model_size = model_size

                for combo_name, combo_dict in size_dict.items():
                    # ensure metrics subtree exists
                    metrics_branch = ensure_nested(
                        metrics, [setup_name, model_size, combo_name]
                    )

                    for dataset_size_str, ckpt_path in combo_dict.items():
                        # decide dataset size for args
                        try:
                            ds_size = int(dataset_size_str)
                        except ValueError:
                            ds_size = -1
                        args.dataset_size = ds_size

                        # If it's not a checkpoint file, just record the path and skip
                        if not ckpt_path.endswith(".ckpt"):
                            existing_entry = metrics_branch.get(dataset_size_str, {})
                            if not isinstance(existing_entry, dict):
                                existing_entry = {}
                            existing_entry.setdefault("ckpt", ckpt_path)
                            existing_entry.setdefault("status", "no_ckpt_file")
                            metrics_branch[dataset_size_str] = existing_entry
                            print(
                                f"[SKIP] {setup_name}/{model_size}/{combo_name}/{dataset_size_str} "
                                f"-> path is not a .ckpt ({ckpt_path})"
                            )
                            continue

                        existing_entry = metrics_branch.get(dataset_size_str)

                        # Check if we already evaluated this exact ckpt with metrics present
                        if (
                            isinstance(existing_entry, dict)
                            and existing_entry.get("ckpt") == ckpt_path
                            and all(
                                k in existing_entry
                                for k in ("avg_loss", "auc", "inv_bkg_eff")
                            )
                        ):
                            print(
                                f"[SKIP] {setup_name}/{model_size}/{combo_name}/{dataset_size_str} "
                                f"-> already evaluated for this ckpt."
                            )
                            continue

                        print(
                            f"[EVAL] {setup_name}/{model_size}/{combo_name}/{dataset_size_str}\n"
                            f"       ckpt: {ckpt_path}"
                        )

                        # safety: if ckpt path does not exist, warn and skip
                        if not os.path.exists(ckpt_path):
                            print(
                                f"[WARN] Checkpoint file does not exist on disk: {ckpt_path}. Skipping."
                            )
                            # still record that it was supposed to be this ckpt
                            metrics_branch[dataset_size_str] = {
                                "ckpt": ckpt_path,
                                "status": "missing_file",
                            }
                            continue

                        avg_loss, auc, inv_bkg_eff = eval_top_tagging(
                            args, ckpt_path, gpuID=args.gpuID
                        )

                        metrics_branch[dataset_size_str] = {
                            "ckpt": ckpt_path,
                            "avg_loss": float(avg_loss),
                            "auc": float(auc),
                            "inv_bkg_eff": float(inv_bkg_eff),
                        }

        # --------- write metrics JSON back (update-in-place) ----------
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

        print(f"\n[DONE] Metrics written to {args.metrics_json}")


if __name__ == "__main__":
    main()
