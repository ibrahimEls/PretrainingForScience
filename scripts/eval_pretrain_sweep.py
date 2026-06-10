#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional

from omnilearn_lightning.eval_tasks import eval_top_tagging

METRIC_KEYS = ("avg_loss", "auc", "inv_bkg_eff")


def load_json_or_empty(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] JSON at {path} is not valid; starting from empty.")
        return {}


def safe_write_json(path: str, data: Any) -> None:
    tmp_path = path + ".tmp"
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p or not isinstance(p, str):
        return None
    try:
        return os.path.realpath(os.path.abspath(os.path.expanduser(p)))
    except Exception:
        return p


def _has_complete_metrics(r: Dict[str, Any]) -> bool:
    return all((k in r) and (r.get(k) is not None) for k in METRIC_KEYS)


def _build_prev_index(prev_runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Map normalized best_ckpt_dest path -> previous run dict, so we can carry
    over already-computed metrics.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in prev_runs or []:
        if not isinstance(r, dict):
            continue
        key = _norm_path(r.get("best_ckpt_dest"))
        if key:
            out[key] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate top-tagging metrics for each ckpt in a pretrain-step sweep state JSON."
    )
    parser.add_argument(
        "--state-json",
        required=True,
        help="State JSON produced by finetune_pretrain_sweep.py",
    )
    parser.add_argument(
        "--eval-json",
        required=True,
        help="Output JSON (per-step metrics).  If it exists, prior metrics are reused.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be evaluated without running evals or writing output.",
    )

    parser.add_argument("--path", type=str, default="/pscratch/sd/i/ibrahime/datasets/")
    parser.add_argument(
        "--outdir", type=str, default="/pscratch/sd/i/ibrahime/checkpoints/"
    )
    parser.add_argument("--dataset", type=str, default="top")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--gpuID", type=int, default=0)
    parser.add_argument("--task", type=str, default="top_tagging")
    parser.add_argument("--use_pid", action="store_true")
    parser.add_argument("--use_add", action="store_true")

    args = parser.parse_args()

    if args.task != "top_tagging":
        raise ValueError("This script currently only supports --task top_tagging.")

    if not os.path.exists(args.state_json):
        raise RuntimeError(f"State JSON not found: {args.state_json}")
    with open(args.state_json, "r") as f:
        state = json.load(f)

    state_runs = state.get("runs", [])
    if not isinstance(state_runs, list) or not state_runs:
        raise RuntimeError(
            f"State JSON {args.state_json} has no 'runs' list or it is empty."
        )

    prev_eval = load_json_or_empty(args.eval_json)
    prev_runs_index = _build_prev_index(prev_eval.get("runs", []))

    out: Dict[str, Any] = {
        "model_size": state.get("model_size"),
        "pre_training_mode": state.get("pre_training_mode"),
        "pretrain_ckpt_dir": state.get("pretrain_ckpt_dir"),
        "dest_dir": state.get("dest_dir"),
        "interval": state.get("interval"),
        "downstream_key": state.get("downstream_key"),
        "downstream_shots": state.get("downstream_shots"),
        "runs": [],
    }

    n_skip = 0
    n_run = 0
    n_missing = 0

    for run in sorted(state_runs, key=lambda r: r.get("pretrain_step", 0)):
        if not isinstance(run, dict):
            continue

        ckpt_path = run.get("best_ckpt_dest")
        pretrain_step = run.get("pretrain_step")

        new_run = {
            "pretrain_step": pretrain_step,
            "pretrain_epoch": run.get("pretrain_epoch"),
            "pretrain_val_loss": run.get("pretrain_val_loss"),
            "pretrain_ckpt": run.get("pretrain_ckpt"),
            "staged_pretrain_ckpt": run.get("staged_pretrain_ckpt"),
            "best_ckpt_dest": ckpt_path,
            "finetune_val_curve": run.get("finetune_val_curve", []),
            "seed": run.get("seed"),
        }

        ckpt_norm = _norm_path(ckpt_path)
        prev = prev_runs_index.get(ckpt_norm) if ckpt_norm else None

        if (
            prev is not None
            and _has_complete_metrics(prev)
            and prev.get("status", "ok") == "ok"
        ):
            for k in METRIC_KEYS:
                new_run[k] = prev[k]
            new_run["status"] = "ok"
            n_skip += 1
            print(
                f"[SKIP] step={pretrain_step} -> already evaluated "
                f"(auc={prev.get('auc')})"
            )
        elif not ckpt_path or not isinstance(ckpt_path, str):
            new_run["status"] = "no_ckpt_path"
            print(f"[WARN] step={pretrain_step}: missing best_ckpt_dest")
            n_missing += 1
        elif not os.path.exists(ckpt_path):
            new_run["status"] = "missing_file"
            print(
                f"[WARN] step={pretrain_step}: ckpt does not exist on disk: {ckpt_path}"
            )
            n_missing += 1
        else:
            if args.dry_run:
                new_run["status"] = "dryrun_pending"
                print(f"[DRYRUN] step={pretrain_step} -> {ckpt_path}")
                n_run += 1
            else:
                print(f"[EVAL] step={pretrain_step} -> {ckpt_path}")
                avg_loss, auc, inv_bkg_eff = eval_top_tagging(
                    args, ckpt_path, gpuID=args.gpuID
                )
                new_run["avg_loss"] = float(avg_loss)
                new_run["auc"] = float(auc)
                new_run["inv_bkg_eff"] = float(inv_bkg_eff)
                new_run["status"] = "ok"
                n_run += 1

        out["runs"].append(new_run)

        if not args.dry_run:
            safe_write_json(args.eval_json, out)

    print("\n" + "=" * 80)
    print(
        f"[DONE] runs evaluated: {n_run}, skipped (cached): {n_skip}, missing ckpt: {n_missing}"
    )
    print(
        f"[DONE] eval JSON written to: {args.eval_json}"
        if not args.dry_run
        else "[DRYRUN] no JSON written"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
