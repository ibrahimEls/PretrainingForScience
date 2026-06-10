#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Map raw pretrain mode strings to a tag substring that
# omnilearn_lightning.utils.extract_pretrained_ckpt_prefix can recognize.
MODE_TO_PREFIX = {
    "classifier": "class_only",
    "generator": "gen_only",
    "mpm": "mpm_only",
    "mpmregress": "mpmregress_only",
    "classifier+generator": "class_gen",
    "classifier+generator-preturb": "class_gen",
    "classifier+mpm": "class_mpm",
    "classifier+mpm_perturb": "class_mpm",
    "classifier+mpmregress": "class_mpmregress",
    "classifier+mpmregress_perturb": "class_mpmregress",
    "generator+mpm": "gen_mpm",
    "generator+mpmregress": "gen_mpmregress",
    "pretrain": "pretrain",
    "pretrain_perturb": "pretrain",
    "pretrainregress": "pretrainregress",
    "pretrainregress_perturb": "pretrainregress",
    "point_lejepa": "point_lejepa",
    "classifier+point_lejepa": "class_point_lejepa",
}


# Model-size-specific hyperparameters (mirrors scripts/finetune_lightning.py)

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
        "num_workers": 2,
        "lr": 1e-6,
        "lr_fs": 1e-5,
        "weight_decay": 5,
        "weight_decay_fs": 0.5,
        "batch_size": 64,
        "lr_factor": 5,
        "epoch": 30,
        "epoch_fs": 30,
        "scheduler_warmup_steps": 0,
        "scheduler_warmup_steps_fs": 0,
    },
}


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
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def load_cmd_template(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def parse_downstream_shots(dataset_key: str) -> Optional[int]:
    try:
        prefix = dataset_key.split("-", 1)[0]
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
    return dataset_key.replace("-", "_")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_best_ckpt_from_output(output: str) -> str:
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


def get_model_hparams(model_size: str) -> Dict[str, Any]:
    if model_size in MODEL_HPARAMS:
        return MODEL_HPARAMS[model_size]
    return {
        "num_workers": 4,
        "num_nodes": 1,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 128,
    }


VAL_CKPT_RE = re.compile(r"epoch=(\d+)-step=(\d+)-val_loss=([\d.]+)")


def stage_pretrain_ckpt(
    pretrain_ckpt: str,
    pre_training_mode: str,
    pretrain_step: int,
    pretrain_val_loss: float,
    staging_dir: str,
    pretrain_prefix_override: Optional[str] = None,
) -> str:
    """
    Create a symlink at ``staging_dir`` whose filename contains a known
    `_PREFIXES` tag, so train_lightning.py's
    `extract_pretrained_ckpt_prefix` succeeds.

    Returns the symlink path.
    """
    if pretrain_prefix_override:
        prefix = pretrain_prefix_override
    elif pre_training_mode in MODE_TO_PREFIX:
        prefix = MODE_TO_PREFIX[pre_training_mode]
    else:
        raise ValueError(
            f"No known prefix mapping for pre_training_mode='{pre_training_mode}'.\n"
            f"Pass --pretrain-prefix manually (one of: "
            f"class_only, gen_only, mpm_only, mpmregress_only, class_gen, "
            f"class_mpm, gen_mpm, class_mpmregress, gen_mpmregress, "
            f"pretrain, pretrainregress, point_lejepa, class_point_lejepa)."
        )

    ensure_dir(staging_dir)
    link_name = (
        f"{prefix}-step={pretrain_step:08d}-val_loss={pretrain_val_loss:.4f}.ckpt"
    )
    link_path = os.path.join(staging_dir, link_name)
    if os.path.islink(link_path) or os.path.exists(link_path):
        try:
            os.remove(link_path)
        except OSError:
            pass
    os.symlink(os.path.abspath(pretrain_ckpt), link_path)
    return link_path


def extract_val_curve_from_csv(metrics_csv: str) -> List[Dict[str, float]]:
    """
    Parse Lightning's metrics.csv and return a list of {"step", "epoch",
    "val_loss", ...} entries for rows that have a non-empty val_loss column.
    Includes any other val_* columns (e.g. val_loss_class) that are present.
    """
    if not os.path.exists(metrics_csv):
        return []

    rows: List[Dict[str, float]] = []
    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        val_cols = [c for c in reader.fieldnames if c.startswith("val_")]
        for row in reader:
            if not row.get("val_loss", "").strip():
                continue
            entry: Dict[str, float] = {}
            for k in ("step", "epoch"):
                v = row.get(k, "").strip()
                if v:
                    try:
                        entry[k] = int(float(v))
                    except ValueError:
                        pass
            for c in val_cols:
                v = row.get(c, "").strip()
                if v:
                    try:
                        entry[c] = float(v)
                    except ValueError:
                        pass
            rows.append(entry)
    return rows


def discover_val_checkpoints(ckpt_dir: str) -> List[Dict[str, Any]]:
    """
    Find all checkpoints in ckpt_dir whose filename contains 'val_loss='.
    Returns a list of dicts sorted by step ascending.  Skips last.ckpt and
    train-step-only checkpoints.
    """
    if not os.path.isdir(ckpt_dir):
        raise RuntimeError(f"Pretrain ckpt dir does not exist: {ckpt_dir}")

    found = []
    for name in os.listdir(ckpt_dir):
        if not name.endswith(".ckpt"):
            continue
        if "val_loss=" not in name:
            continue
        m = VAL_CKPT_RE.search(name)
        if not m:
            continue
        epoch = int(m.group(1))
        step = int(m.group(2))
        val_loss = float(m.group(3))
        found.append(
            {
                "pretrain_epoch": epoch,
                "pretrain_step": step,
                "pretrain_val_loss": val_loss,
                "pretrain_ckpt": os.path.join(ckpt_dir, name),
            }
        )

    found.sort(key=lambda r: r["pretrain_step"])
    return found


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
    Format the shell template with all placeholders.  Mirrors
    scripts/finetune_lightning.py:268-355.
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

    epoch = hparams["epoch"] if pretrain_ckpt else hparams["epoch_fs"]
    gpus_per_node = 4
    max_training_steps = (
        1_200_000 // (hparams["batch_size"] * num_nodes * gpus_per_node)
    ) * epoch

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
        "epoch": 99999,
        "max_training_steps": max_training_steps,
        "scheduler_warmup_steps": hparams["scheduler_warmup_steps"]
        if pretrain_ckpt
        else hparams["scheduler_warmup_steps_fs"],
    }
    return cmd_template.format(**fmt_kwargs)


def run_single_finetune(
    cmd_template: str,
    dest_step_dir: str,
    model_size: str,
    pre_training_mode: str,
    pretrain_ckpt: str,
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
    pretrain_step: int,
    dry_run: bool = False,
) -> Tuple[Optional[str], Optional[str], str, List[Dict[str, float]]]:
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
        f"[RUN] step={pretrain_step}, model_size={model_size}, "
        f"pre_training_mode={pre_training_mode}, seed={seed}"
    )
    print(f"[PRETRAIN CKPT] {pretrain_ckpt}")
    print("[SHELL SNIPPET]")
    print(cmd_snippet)

    if dry_run:
        print("[DRY-RUN] Not executing command; returning without paths.")
        return None, None, cmd_snippet, []

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

    ensure_dir(dest_step_dir)
    dest_ckpt_path = os.path.join(dest_step_dir, os.path.basename(best_ckpt_src))
    shutil.copy2(best_ckpt_src, dest_ckpt_path)
    print(f"[INFO] Copied best checkpoint to: {dest_ckpt_path}")

    run_dir = os.path.dirname(os.path.dirname(best_ckpt_src))
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    val_curve = extract_val_curve_from_csv(metrics_csv)
    if val_curve:
        print(
            f"[INFO] Extracted finetuning val curve ({len(val_curve)} entries) from {metrics_csv}"
        )
    else:
        print(f"[WARN] No val curve extracted from {metrics_csv}")

    return best_ckpt_src, dest_ckpt_path, cmd_snippet, val_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep fine-tuning over a single pre-training trajectory: one "
            "fine-tune per val-step pre-training checkpoint."
        )
    )
    p.add_argument(
        "--pretrain-ckpt-dir",
        required=True,
        help="Directory containing per-val-check pre-training checkpoints.",
    )
    p.add_argument(
        "--state-json",
        required=True,
        help="Output JSON tracking finetune runs per pre-train step.",
    )
    p.add_argument(
        "--cmd-template-file",
        required=True,
        help="Path to assets/finetune-template.sh.",
    )
    p.add_argument("--repo_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument(
        "--dest-dir",
        required=True,
        help="Where to copy best fine-tuned ckpts (a subdir per pretrain step).",
    )
    p.add_argument("--dataset", default="top")
    p.add_argument("--mode", default="classifier")
    p.add_argument(
        "--model_size",
        required=True,
        choices=["micro", "small", "medium"],
    )
    p.add_argument(
        "--pre_training_mode",
        required=True,
        help="Pre-training mode label (e.g. 'classifier+generator'). Bookkeeping only.",
    )
    p.add_argument("--num_nodes", type=int, default=4)
    p.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Take every Nth val checkpoint (1 = all).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--downstream-key",
        default="1M-Top",
        help="Downstream key (used only for short-name and template formatting).",
    )
    p.add_argument(
        "--downstream-shots",
        type=int,
        default=-1,
        help="-1 for full top dataset.",
    )
    p.add_argument(
        "--pretrain-prefix",
        default=None,
        help=(
            "Override the prefix tag used when staging the pretrain ckpt symlink "
            "(must be one of the recognized _PREFIXES like 'class_only', "
            "'class_gen', etc.). If unset, derived from --pre_training_mode."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cmd_template = load_cmd_template(args.cmd_template_file)
    state = load_json(args.state_json, default={})

    state.setdefault("model_size", args.model_size)
    state.setdefault("pre_training_mode", args.pre_training_mode)
    state.setdefault("pretrain_ckpt_dir", args.pretrain_ckpt_dir)
    state.setdefault("dest_dir", args.dest_dir)
    state.setdefault("interval", args.interval)
    state.setdefault("downstream_key", args.downstream_key)
    state.setdefault("downstream_shots", args.downstream_shots)
    state.setdefault("runs", [])

    discovered = discover_val_checkpoints(args.pretrain_ckpt_dir)
    if not discovered:
        raise RuntimeError(
            f"No val checkpoints (containing 'val_loss=') found in "
            f"{args.pretrain_ckpt_dir}"
        )

    selected = discovered[:: max(1, args.interval)]

    print("=" * 80)
    print("[INFO] Pre-training-step fine-tuning sweep")
    print(f"[INFO] Pretrain dir:    {args.pretrain_ckpt_dir}")
    print(f"[INFO] Discovered:      {len(discovered)} val ckpts")
    print(f"[INFO] Interval:        {args.interval}")
    print(f"[INFO] Selected:        {len(selected)} ckpts to fine-tune")
    print(f"[INFO] Model size:      {args.model_size}")
    print(f"[INFO] Pre-train mode:  {args.pre_training_mode}")
    print(f"[INFO] Mode (finetune): {args.mode}")
    print(f"[INFO] Dest dir:        {args.dest_dir}")
    print(f"[INFO] State JSON:      {args.state_json}")
    print(f"[INFO] Dry run:         {args.dry_run}")
    print("=" * 80)

    done_steps = {
        r.get("pretrain_step")
        for r in state["runs"]
        if r.get("best_ckpt_dest") and r.get("status", "ok") == "ok"
    }

    downstream_short = derive_downstream_short(args.downstream_key)

    n_submitted = 0
    n_skipped_already_done = 0
    n_failed = 0
    submitted_steps: List[int] = []
    skipped_steps: List[int] = []
    failed_steps: List[int] = []

    for run_index, info in enumerate(selected):
        step = info["pretrain_step"]
        if step in done_steps:
            print(f"\n[SKIP] step={step} already finetuned at {args.state_json}")
            n_skipped_already_done += 1
            skipped_steps.append(step)
            continue

        dest_step_dir = os.path.join(args.dest_dir, f"step_{step:08d}")

        staging_dir = os.path.join(args.dest_dir, ".staged")
        try:
            staged_ckpt = stage_pretrain_ckpt(
                pretrain_ckpt=info["pretrain_ckpt"],
                pre_training_mode=args.pre_training_mode,
                pretrain_step=step,
                pretrain_val_loss=info["pretrain_val_loss"],
                staging_dir=staging_dir,
                pretrain_prefix_override=args.pretrain_prefix,
            )
            print(
                f"[INFO] Staged ckpt symlink: {staged_ckpt} -> {info['pretrain_ckpt']}"
            )
        except Exception as e:
            print(f"[ERROR] step={step} could not stage symlink: {e}")
            staged_ckpt = info["pretrain_ckpt"]

        val_curve: List[Dict[str, float]] = []
        try:
            best_src, best_dest, cmd_snippet, val_curve = run_single_finetune(
                cmd_template=cmd_template,
                dest_step_dir=dest_step_dir,
                model_size=args.model_size,
                pre_training_mode=args.pre_training_mode,
                pretrain_ckpt=staged_ckpt,
                downstream_key=args.downstream_key,
                downstream_short=downstream_short,
                downstream_shots=args.downstream_shots,
                pretrain_label=str(step),
                run_index=run_index,
                seed=args.seed,
                dataset=args.dataset,
                num_nodes=args.num_nodes,
                repo_dir=args.repo_dir,
                output_dir=args.output_dir,
                dataset_dir=args.dataset_dir,
                mode=args.mode,
                pretrain_step=step,
                dry_run=args.dry_run,
            )
            status = "ok" if not args.dry_run else "dryrun"
            n_submitted += 1
            submitted_steps.append(step)
        except Exception as e:
            print(f"[ERROR] step={step} failed: {e}")
            best_src, best_dest, cmd_snippet = None, None, ""
            status = "failed"
            n_failed += 1
            failed_steps.append(step)

        run_record = {
            "pretrain_step": step,
            "pretrain_epoch": info["pretrain_epoch"],
            "pretrain_val_loss": info["pretrain_val_loss"],
            "pretrain_ckpt": info["pretrain_ckpt"],
            "staged_pretrain_ckpt": staged_ckpt,
            "best_ckpt_src": best_src,
            "best_ckpt_dest": best_dest,
            "cmd_snippet": cmd_snippet,
            "finetune_val_curve": val_curve,
            "seed": args.seed,
            "run_index": run_index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": status,
        }

        state = load_json(args.state_json, default=state)
        state.setdefault("runs", [])
        existing_idx = next(
            (i for i, r in enumerate(state["runs"]) if r.get("pretrain_step") == step),
            None,
        )
        if existing_idx is None:
            state["runs"].append(run_record)
        else:
            state["runs"][existing_idx] = run_record
        state["runs"].sort(key=lambda r: r.get("pretrain_step", 0))

        if not args.dry_run:
            ensure_dir(os.path.dirname(args.state_json) or ".")
            safe_write_json(args.state_json, state)
            print(f"[INFO] Updated state JSON: {args.state_json}")

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"  Discovered val ckpts:                {len(discovered)}")
    print(f"  Selected (after interval={args.interval}):       {len(selected)}")
    print(f"  Already-done (skipped):              {n_skipped_already_done}")
    if args.dry_run:
        print(f"  Would-be-submitted finetunings:      {n_submitted}")
    else:
        print(f"  Submitted finetunings:               {n_submitted}")
    print(f"  Failed finetunings:                  {n_failed}")
    if submitted_steps:
        print(f"  Submitted steps: {submitted_steps}")
    if skipped_steps:
        print(f"  Skipped steps:   {skipped_steps}")
    if failed_steps:
        print(f"  Failed steps:    {failed_steps}")
    print("=" * 80)
    print("[DONE] Sweep complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
