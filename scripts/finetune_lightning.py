#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Model-size-specific hyperparameters

MODEL_HPARAMS = {
    "micro": {
        "num_workers": 2,
        "lr": 5e-5,
        "lr_fs": 1e-4,
        "weight_decay": 0.1,
        "weight_decay_fs": 0.5,
        "batch_size": 64,
        "lr_factor": 5,
        "epoch": 30,
        "epoch_fs": 30,
        "scheduler_warmup_steps": 1028,
        "scheduler_warmup_steps_fs": 0,
    },
    "small": {
        "num_workers": 2,
        "lr": 5e-6,
        "lr_fs": 5e-4,
        "weight_decay": 0.1,
        "weight_decay_fs": 0.5,
        "batch_size": 64,
        "lr_factor": 5,
        "epoch": 30,
        "epoch_fs": 15,
        "scheduler_warmup_steps": 1028,
        "scheduler_warmup_steps_fs": 0,
    },
    "medium": {
        "num_workers": 32,
        "lr": 1e-6,
        "lr_fs": 5e-5,
        "weight_decay": 10,
        "weight_decay_fs": 0.5,
        "batch_size": 64,
        "lr_factor": 1,
        "epoch": 10,
        "epoch_fs": 30,
        "scheduler_warmup_steps": 1028,
        "scheduler_warmup_steps_fs": 0,
    },
}


# Utility helperss


def load_json(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return default
    except Exception as e:
        print(e)
        print("Loading State Config Failed.. Falling back on default")
        return default


def safe_write_json(path: str, data: Any) -> None:
    """
    Atomically write JSON so we never leave a half-written file.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def load_cmd_template(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def parse_downstream_shots(dataset_key: str) -> Optional[int]:
    """
    Parse '1k-Top', '10k-Top', '100k-Top', '1M-Top' -> integer shots.
    """
    try:
        prefix = dataset_key.split("-", 1)[0]  # e.g. "1k", "10k", "100k", "1M"
    except Exception:
        return None

    p = prefix.lower()
    try:
        if p.endswith("k"):
            return int(float(p[:-1]) * 1_000)
        if p.endswith("m"):
            return int(float(p[:-1]) * 1_000_000)
        return int(p)
    except ValueError:
        return None


def derive_downstream_short(dataset_key: str) -> str:
    """
    Convert '1M-Top' -> '1M_top' for use as dataset name if needed.
    """
    return dataset_key.replace("-", "_")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_best_ckpt_from_output(output: str) -> str:
    """
    Look for 'Best model checkpoint: /path/to/ckpt.ckpt' in the log.
    """
    marker = "Best model checkpoint:"
    best_path = None
    for line in output.splitlines():
        if marker in line:
            best_path = line.split(marker, 1)[1].strip()
    if not best_path:
        raise RuntimeError(
            "Could not find 'Best model checkpoint: <path>' in training output.\n"
            "Make sure your training script prints this exact pattern."
        )
    return best_path


def resolve_dest_base_dir(
    layout_cfg: Dict[str, Any],
    downstream_key: str,
    model_size: str,
    pre_training_mode: str,
    pretrain_label: str,
) -> Optional[str]:
    """
    Use the layout JSON to figure out the base directory for this configuration.

    For non-'from_scratch' pre_training_modes:
        layout_cfg[downstream_key][model_size][pre_training_mode][pretrain_label] -> str

    For 'from_scratch':
        layout_cfg[downstream_key][model_size]['from_scratch'] -> str or dict

    If the path ends with '.ckpt' we use dirname(path) as the base dir.
    """
    try:
        model_block = layout_cfg[downstream_key][model_size]
    except KeyError:
        print(
            f"[SKIP] No layout entry for dataset={downstream_key}, model_size={model_size}"
        )
        return None

    if pre_training_mode == "from_scratch":
        if "from_scratch" not in model_block:
            print(
                f"[SKIP] No 'from_scratch' entry for dataset={downstream_key}, model_size={model_size}"
            )
            return None
        entry = model_block["from_scratch"]
        if isinstance(entry, str):
            raw = entry
        elif isinstance(entry, dict):
            if not entry:
                print(
                    f"[SKIP] 'from_scratch' entry is an empty dict for "
                    f"dataset={downstream_key}, model_size={model_size}."
                )
                return None
            raw = next(iter(entry.values()))
        else:
            print(
                f"[SKIP] 'from_scratch' entry is neither str nor dict for "
                f"dataset={downstream_key}, model_size={model_size}"
            )
            return None
    else:
        if pre_training_mode not in model_block:
            print(
                f"[SKIP] No pre_training_mode '{pre_training_mode}' in layout for dataset={downstream_key}, model_size={model_size}"
            )
            return None
        pre_training_mode_block = model_block[pre_training_mode]
        if not isinstance(pre_training_mode_block, dict):
            print(
                f"[SKIP] Unexpected layout structure at {downstream_key}/{model_size}/{pre_training_mode}"
            )
            return None
        if pretrain_label not in pre_training_mode_block:
            print(
                f"[SKIP] No layout path for dataset={downstream_key}, model_size={model_size}, "
                f"pre_training_mode={pre_training_mode}, pretrain_label={pretrain_label}"
            )
            return None
        raw = pre_training_mode_block[pretrain_label]

    if not isinstance(raw, str) or not raw:
        print(
            f"[SKIP] Layout path for dataset={downstream_key}, model_size={model_size}, "
            f"pre_training_mode={pre_training_mode}, pretrain_label={pretrain_label} is not a non-empty string."
        )
        return None

    if raw.endswith(".ckpt"):
        base_dir = os.path.dirname(raw)
    else:
        base_dir = raw

    ensure_dir(base_dir)
    return base_dir


def get_or_init_group(
    state: Dict[str, Any],
    downstream_key: str,
    model_size: str,
    pre_training_mode: str,
    pretrain_label: str,
    pretrain_ckpt: Optional[str],
) -> Dict[str, Any]:
    """
    In state JSON we store groups per (dataset, model_size, pre_training_mode, pretrain_label),
    each group tied to a particular pretrain_ckpt path.

    state[downstream_key][model_size][pre_training_mode][pretrain_label] = [
        {
            "pretrain_ckpt": "<path or None>",
            "runs": [ {...}, ... ]
        },
        ...
    ]
    """
    ds_block = state.setdefault(downstream_key, {})
    model_block = ds_block.setdefault(model_size, {})
    pre_training_mode_block = model_block.setdefault(pre_training_mode, {})
    label_list: List[Dict[str, Any]] = pre_training_mode_block.setdefault(
        pretrain_label, []
    )

    for group in label_list:
        if group.get("pretrain_ckpt") == pretrain_ckpt:
            return group

    group = {"pretrain_ckpt": pretrain_ckpt, "runs": []}
    label_list.append(group)
    return group


def get_model_hparams(model_size: str) -> Dict[str, Any]:
    if model_size in MODEL_HPARAMS:
        return MODEL_HPARAMS[model_size]
    # Fallback conservative defaults
    return {
        "num_workers": 4,
        "num_nodes": 1,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 128,
    }


def build_cmd_snippet(
    cmd_template: str,
    model_size: str,
    pre_training_mode: str,
    pretrain_ckpt: Optional[str],
    downstream_key: str,
    downstream_short: str,
    downstream_shots: Optional[int],
    pretrain_label: str,
    run_index: int,
    seed: int,
    dataset: str,
    num_nodes: int,
    repo_dir: str,
    output_dir: str,
    dataset_dir: str,
    mode: str,
) -> str:
    """
    Format the shell template with all placeholders.

    Placeholders available:
        {model_size}
        {pre_training_mode}
        {pretrain_ckpt}
        {downstream_key}
        {downstream_short}
        {downstream_shots}
        {pretrain_label}
        {run_index}
        {seed}
        {fine_tune_flag}
        {ckpt_flag}
        {mode}
        {num_workers}, {num_nodes}, {lr}, {weight_decay}, {batch_size}
    """
    shots = downstream_shots if downstream_shots is not None else -1
    shots = -1 if shots == 1_000_000 and dataset == "top" else shots

    if pretrain_ckpt:
        fine_tune_flag = "--fine_tune y"
        ckpt_flag = f"--ckpt {pretrain_ckpt}"
    else:
        fine_tune_flag = ""
        ckpt_flag = ""

    hparams = get_model_hparams(model_size)

    fmt_kwargs = {
        "model_size": model_size,
        "pre_training_mode": pre_training_mode,
        "pretrain_ckpt": pretrain_ckpt or "",
        "downstream_key": downstream_key,
        "downstream_short": downstream_short,
        "downstream_shots": shots,
        "pretrain_label": pretrain_label,
        "run_index": run_index,
        "seed": seed,
        "fine_tune_flag": fine_tune_flag,
        "ckpt_flag": ckpt_flag,
        "mode": mode,
        "num_workers": hparams["num_workers"],
        "num_nodes": num_nodes,
        "lr": hparams["lr"] if pretrain_ckpt else hparams["lr_fs"],
        "weight_decay": hparams["weight_decay"]
        if pretrain_ckpt
        else hparams["weight_decay_fs"],
        "batch_size": hparams["batch_size"] if shots > 1000 or shots == -1 else 32,
        "dataset": dataset,
        "repo_dir": repo_dir,
        "output_dir": output_dir,
        "dataset_dir": dataset_dir,
        "lr_factor": hparams["lr_factor"] if pretrain_ckpt else 1,
        "epoch": hparams["epoch"] if pretrain_ckpt else hparams["epoch_fs"],
        "scheduler_warmup_steps": hparams["scheduler_warmup_steps"]
        if pretrain_ckpt
        else hparams["scheduler_warmup_steps_fs"],
    }

    return cmd_template.format(**fmt_kwargs)


def run_single_finetune(
    cmd_template: str,
    dest_base_dir: str,
    model_size: str,
    pre_training_mode: str,
    pretrain_ckpt: Optional[str],
    downstream_key: str,
    downstream_short: str,
    downstream_shots: Optional[int],
    pretrain_label: str,
    run_index: int,
    seed: int,
    dataset: str,
    num_nodes: int,
    repo_dir: str,
    output_dir: str,
    dataset_dir: str,
    mode: str,
    dry_run: bool = False,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Execute one finetuning run using the shell template.
    Returns (best_ckpt_src, best_ckpt_dest, cmd_snippet).
    """
    cmd_snippet = build_cmd_snippet(
        cmd_template=cmd_template,
        model_size=model_size,
        pre_training_mode=pre_training_mode,
        pretrain_ckpt=pretrain_ckpt,
        downstream_key=downstream_key,
        downstream_short=downstream_short,
        downstream_shots=downstream_shots,
        pretrain_label=pretrain_label,
        run_index=run_index,
        seed=seed,
        dataset=dataset,
        num_nodes=num_nodes,
        repo_dir=repo_dir,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        mode=mode,
    )

    print("\n" + "=" * 80)
    print(
        f"[RUN] dataset={downstream_key}, model_size={model_size}, pre_training_mode={pre_training_mode}, "
        f"pretrain_label={pretrain_label}, seed={seed}, run_index={run_index}"
    )
    print(
        f"[PRETRAIN CKPT] {pretrain_ckpt if pretrain_ckpt else 'None (from_scratch)'}"
    )
    print("[SHELL SNIPPET]")
    print(cmd_snippet)

    if dry_run:
        print("[DRY-RUN] Not executing command; returning without paths.")
        return None, None, cmd_snippet

    # Run the snippet through bash -lc so 'cd', 'source', env vars, etc. work.
    result = subprocess.run(
        ["bash", "-lc", cmd_snippet],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("[TRAIN-OUTPUT-BEGIN]")
    print(result.stdout)
    print("[TRAIN-OUTPUT-END]")

    if result.returncode != 0:
        raise RuntimeError(
            f"Training command failed with exit code {result.returncode}"
        )

    best_ckpt_src = parse_best_ckpt_from_output(result.stdout)
    print(f"[INFO] Found best checkpoint from training script: {best_ckpt_src}")

    # Per-run directory
    run_dir = os.path.join(dest_base_dir, f"run_{run_index:03d}_seed_{seed}")
    ensure_dir(run_dir)
    dest_ckpt_path = os.path.join(run_dir, os.path.basename(best_ckpt_src))

    shutil.copy2(best_ckpt_src, dest_ckpt_path)
    print(f"[INFO] Copied best checkpoint to: {dest_ckpt_path}")

    return best_ckpt_src, dest_ckpt_path, cmd_snippet


# Main orchestration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automate OmniLearned Top finetuning over many configs."
    )

    p.add_argument(
        "--pretrain-json",
        required=True,
        help="Path to Pre-Trained-Model-Paths.json (pre-training ckpt paths).",
    )
    p.add_argument(
        "--layout-json",
        required=True,
        help="Path to Top-Class-Finetuned-Model-Paths.json (destination folder structure).",
    )
    p.add_argument(
        "--state-json",
        required=True,
        help=(
            "Path to state JSON where we store lists of uncertainty runs. "
            "If it exists, it will be updated; otherwise it will be created."
        ),
    )

    p.add_argument("--dataset", required=True, help=("dataset to finetune on"))

    p.add_argument(
        "--downstream-datasets",
        default="all",
        help=(
            "Comma-separated list of downstream dataset keys from layout-json "
            "(e.g. '1M-Top,10k-Top'). Use 'all' to run over all keys."
        ),
    )
    p.add_argument(
        "--model-sizes",
        default="micro,small,medium",
        help="Comma-separated list of model sizes to consider (e.g. 'micro,small,medium').",
    )
    p.add_argument(
        "--pre_training_modes",
        default="classifier,generator,mpm,classifier+generator,classifier+mpm,generator+mpm,from_scratch",
        help=(
            "Comma-separated list of pre_training_modes to consider. "
            "Include 'from_scratch' if you want from-scratch runs."
        ),
    )

    p.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Target number of uncertainty runs per config (per pretrain_ckpt).",
    )
    p.add_argument(
        "--num_nodes",
        type=int,
        default=4,
        help="Target number of uncertainty runs per config (per pretrain_ckpt).",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=1234,
        help="Base seed; each run uses base_seed + global_run_index.",
    )

    p.add_argument("--repo_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--mode", type=str, required=True)

    p.add_argument(
        "--cmd-template-file",
        required=True,
        help=(
            "Path to a shell template file. It will be read as text and formatted with "
            "Python .format(**kwargs), then executed via 'bash -lc \"<snippet>\"'."
        ),
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not actually launch training or copy checkpoints; just print actions.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    pretrain_cfg = load_json(args.pretrain_json, default={})
    layout_cfg = load_json(args.layout_json, default={})
    state_cfg = load_json(args.state_json, default={})

    if not pretrain_cfg:
        raise RuntimeError(
            f"Pre-training JSON not found or empty: {args.pretrain_json}"
        )
    if not layout_cfg:
        raise RuntimeError(f"Layout JSON not found or empty: {args.layout_json}")

    cmd_template = load_cmd_template(args.cmd_template_file)

    all_ds_keys = list(layout_cfg.keys())
    if args.downstream_datasets.lower() == "all":
        ds_keys = all_ds_keys
    else:
        requested = [
            x.strip() for x in args.downstream_datasets.split(",") if x.strip()
        ]
        ds_keys = [k for k in requested if k in layout_cfg]
        missing_ds = [k for k in requested if k not in layout_cfg]
        if missing_ds:
            print(
                f"[WARN] Some requested downstream datasets not in layout JSON: {missing_ds}"
            )

    model_sizes = [m.strip() for m in args.model_sizes.split(",") if m.strip()]
    pre_training_modes = [
        h.strip() for h in args.pre_training_modes.split(",") if h.strip()
    ]

    # Count pretrain labels per mode for the summary
    pretrain_labels_per_mode = {}
    for mode in pre_training_modes:
        if mode == "from_scratch":
            pretrain_labels_per_mode[mode] = ["from_scratch"]
        else:
            labels = []
            for ms in model_sizes:
                if ms in pretrain_cfg and mode in pretrain_cfg[ms]:
                    labels = list(pretrain_cfg[ms][mode].keys())
                    break
            pretrain_labels_per_mode[mode] = labels if labels else ["(none found)"]

    print("=" * 80)
    print("[INFO] Starting finetuning orchestration.")
    print(f"[INFO] Downstream dataset: {args.dataset}")
    print(f"[INFO] Dry run: {args.dry_run}")
    print("=" * 80)
    print("[INFO] Parameter space:")
    print(f"       - Downstream keys ({len(ds_keys)}): {ds_keys}")
    print(f"       - Model sizes ({len(model_sizes)}): {model_sizes}")
    print(
        f"       - Pre-training modes ({len(pre_training_modes)}): {pre_training_modes}"
    )
    for mode, labels in pretrain_labels_per_mode.items():
        print(f"         - {mode}: pretrain_labels ({len(labels)}): {labels}")
    print(f"       - Runs per config: {args.n_runs}")
    print("=" * 80)
    print("[INFO] Max possible runs (before skipping already-completed):")
    total_configs = 0
    for mode in pre_training_modes:
        n_labels = len(pretrain_labels_per_mode[mode])
        configs_for_mode = len(ds_keys) * len(model_sizes) * n_labels
        total_configs += configs_for_mode
        print(
            f"       - {mode}: {len(ds_keys)} ds_keys x {len(model_sizes)} model_sizes x {n_labels} pretrain_labels = {configs_for_mode} configs"
        )
    max_runs = total_configs * args.n_runs
    print(
        f"       Total: {total_configs} configs x {args.n_runs} runs/config = {max_runs} max runs"
    )
    print("=" * 80)
    print(f"[INFO] State JSON: {args.state_json}")
    print(f"[INFO] Command template file: {args.cmd_template_file}")
    print("=" * 80)

    global_run_counter = 0
    total_runs_to_submit = 0
    total_runs_skipped = 0

    for ds_key in ds_keys:
        downstream_short = derive_downstream_short(ds_key)
        downstream_shots = parse_downstream_shots(ds_key)
        print(
            f"\n[DATASET] {ds_key} (short='{downstream_short}', shots={downstream_shots})"
        )

        for model_size in model_sizes:
            print(f"\n  [MODEL-SIZE] {model_size}")

            for pre_training_mode in pre_training_modes:
                print(f"    [PRE-TRAINING MODE] {pre_training_mode}")

                if pre_training_mode == "from_scratch":
                    pretrain_labels = ["from_scratch"]
                else:
                    if model_size not in pretrain_cfg:
                        print(
                            f"      [SKIP] model_size '{model_size}' not in pretrain JSON."
                        )
                        continue
                    if pre_training_mode not in pretrain_cfg[model_size]:
                        print(
                            f"      [SKIP] pre_training_mode '{pre_training_mode}' not in pretrain JSON for model_size '{model_size}'."
                        )
                        continue
                    # sort labels with '-1' last
                    pretrain_labels = sorted(
                        pretrain_cfg[model_size][pre_training_mode].keys(),
                        key=lambda x: (x == "-1", int(x) if x.isdigit() else 0),
                    )

                for pretrain_label in pretrain_labels:
                    print(f"      [CONFIG] pretrain_label={pretrain_label}")

                    if pre_training_mode == "from_scratch":
                        pretrain_ckpt = None
                    else:
                        pretrain_ckpt = pretrain_cfg[model_size][pre_training_mode].get(
                            pretrain_label
                        )
                        if not pretrain_ckpt:
                            print(
                                f"        [SKIP] No pretrain ckpt for model_size={model_size}, "
                                f"pre_training_mode={pre_training_mode}, pretrain_label={pretrain_label}"
                            )
                            continue
                        if not os.path.exists(pretrain_ckpt):
                            print(
                                f"        [SKIP] Pretrain ckpt path does not exist on disk: {pretrain_ckpt}"
                            )
                            continue

                    dest_base_dir = resolve_dest_base_dir(
                        layout_cfg=layout_cfg,
                        downstream_key=ds_key,
                        model_size=model_size,
                        pre_training_mode=pre_training_mode,
                        pretrain_label=pretrain_label,
                    )
                    if not dest_base_dir:
                        continue

                    state_cfg = load_json(args.state_json, default=state_cfg)
                    group = get_or_init_group(
                        state_cfg,
                        downstream_key=ds_key,
                        model_size=model_size,
                        pre_training_mode=pre_training_mode,
                        pretrain_label=pretrain_label,
                        pretrain_ckpt=pretrain_ckpt,
                    )

                    existing_runs = len(group.get("runs", []))
                    if existing_runs >= args.n_runs:
                        print(
                            f"        [SKIP] Already have {existing_runs} runs for this config "
                            f"(target={args.n_runs})."
                        )
                        total_runs_skipped += args.n_runs
                        continue

                    runs_needed = args.n_runs - existing_runs
                    total_runs_to_submit += runs_needed
                    print(
                        f"        [INFO] Have {existing_runs} runs, need {args.n_runs}. "
                        f"Launching {runs_needed} additional runs."
                    )

                    for local_run_idx in range(existing_runs, args.n_runs):
                        seed = args.base_seed + global_run_counter
                        global_run_counter += 1

                        best_src, best_dest, cmd_snippet = run_single_finetune(
                            cmd_template=cmd_template,
                            dest_base_dir=dest_base_dir,
                            model_size=model_size,
                            pre_training_mode=pre_training_mode,
                            pretrain_ckpt=pretrain_ckpt,
                            downstream_key=ds_key,
                            downstream_short=downstream_short,
                            downstream_shots=downstream_shots,
                            pretrain_label=pretrain_label,
                            run_index=local_run_idx,
                            seed=seed,
                            dry_run=args.dry_run,
                            dataset=args.dataset,
                            num_nodes=args.num_nodes,
                            repo_dir=args.repo_dir,
                            output_dir=args.output_dir,
                            dataset_dir=args.dataset_dir,
                            mode=args.mode,
                        )

                        state_cfg = load_json(args.state_json, default=state_cfg)
                        group = get_or_init_group(
                            state_cfg,
                            downstream_key=ds_key,
                            model_size=model_size,
                            pre_training_mode=pre_training_mode,
                            pretrain_label=pretrain_label,
                            pretrain_ckpt=pretrain_ckpt,
                        )

                        timestamp = datetime.now().isoformat(timespec="seconds")
                        run_record = {
                            "run_index": local_run_idx,
                            "seed": seed,
                            "timestamp": timestamp,
                            "cmd_snippet": cmd_snippet,
                            "best_ckpt_src": best_src,
                            "best_ckpt_dest": best_dest,
                        }

                        group.setdefault("runs", []).append(run_record)

                        print(f"        [SAVE] Updating state JSON: {args.state_json}")
                        if not args.dry_run:
                            safe_write_json(args.state_json, state_cfg)

    # Print summary at the end (easier to see after all the output)
    print("\n" + "=" * 80)
    print("[SUMMARY] Orchestration complete.")
    print("=" * 80)
    print("[INFO] Parameter space:")
    print(f"       - Downstream keys ({len(ds_keys)}): {ds_keys}")
    print(f"       - Model sizes ({len(model_sizes)}): {model_sizes}")
    print(
        f"       - Pre-training modes ({len(pre_training_modes)}): {pre_training_modes}"
    )
    for mode, labels in pretrain_labels_per_mode.items():
        print(f"         - {mode}: pretrain_labels ({len(labels)}): {labels}")
    print(f"       - Runs per config: {args.n_runs}")
    print("-" * 80)
    print("[INFO] Run calculation:")
    for mode in pre_training_modes:
        n_labels = len(pretrain_labels_per_mode[mode])
        configs_for_mode = len(ds_keys) * len(model_sizes) * n_labels
        print(
            f"       - {mode}: {len(ds_keys)} ds_keys x {len(model_sizes)} model_sizes x {n_labels} pretrain_labels = {configs_for_mode} configs"
        )
    print(
        f"       Total: {total_configs} configs x {args.n_runs} runs/config = {max_runs} max runs"
    )
    print("-" * 80)
    print(f"[INFO] Total runs submitted: {global_run_counter}")
    print(f"[INFO] Total runs skipped (already completed): {total_runs_skipped}")
    if args.dry_run:
        print(f"[INFO] (Dry run - {total_runs_to_submit} runs would be submitted)")
    print("-" * 80)
    print(f"[INFO] State JSON: {args.state_json}")
    print(f"[INFO] Command template: {args.cmd_template_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
