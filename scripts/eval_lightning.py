#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from omnilearn_lightning.eval_tasks import eval_top_tagging

METRIC_KEYS = ("avg_loss", "auc", "inv_bkg_eff")


def load_json_or_empty(path):
    """Load JSON if it exists and is valid, otherwise return {}."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] JSON at {path} is not valid; starting from empty.")
        return {}


def parse_downstream_shots(dataset_key: str):
    """
    Parse something like '1k-Top', '10k-Top', '100k-Top', '1M-Top' -> integer shots.
    If parsing fails, return None and let caller decide what to do.
    """
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


def mean_and_std(vals):
    """Return (mean, std) with sample std (n-1 in denominator) for a list of floats."""
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals) / n
    if n > 1:
        var = sum((x - m) ** 2 for x in vals) / (n - 1)
    else:
        var = 0.0
    return m, math.sqrt(var)


def _norm_path(p: Optional[str]) -> Optional[str]:
    """Normalize paths for robust equality comparisons."""
    if not p or not isinstance(p, str):
        return None
    try:
        # expanduser + abspath, then resolve symlinks
        return os.path.realpath(os.path.abspath(os.path.expanduser(p)))
    except Exception:
        return p


def _group_match_key(g: Dict[str, Any], fallback_idx: int) -> Tuple[str, int]:
    """
    Stable key for matching groups across runs.
    Primary: pretrain_ckpt (stringified), fallback: index.
    """
    pre = g.get("pretrain_ckpt")
    if isinstance(pre, str) and pre.strip():
        return (pre.strip(), -1)  # -1 means "do not use index"
    return ("__no_pretrain_ckpt__", fallback_idx)


def _build_prev_group_index(
    prev_group_list: Any,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Map group_key -> prev_group dict.
    If duplicates exist, last one wins (rare; acceptable).
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not isinstance(prev_group_list, list):
        return out
    for i, g in enumerate(prev_group_list):
        if isinstance(g, dict):
            out[_group_match_key(g, i)] = g
    return out


def _find_prev_run(
    prev_runs: Any, run_index: Any, ckpt_path: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Find matching previous run.
    Priority:
      1) normalized ckpt path match
      2) run_index match
    """
    if not isinstance(prev_runs, list):
        return None

    ckpt_norm = _norm_path(ckpt_path)

    # 1) match by checkpoint path (robust)
    if ckpt_norm:
        for r_prev in prev_runs:
            if not isinstance(r_prev, dict):
                continue
            prev_ckpt_norm = _norm_path(r_prev.get("best_ckpt_dest"))
            if prev_ckpt_norm and prev_ckpt_norm == ckpt_norm:
                return r_prev

    # 2) fallback: match by run_index
    for r_prev in prev_runs:
        if not isinstance(r_prev, dict):
            continue
        if r_prev.get("run_index") == run_index:
            return r_prev

    return None


def _has_complete_metrics(r: Dict[str, Any]) -> bool:
    return all((k in r) and (r.get(k) is not None) for k in METRIC_KEYS)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate top-tagging ckpts for each run in a States JSON (multi-run structure)."
    )

    # ------- paths to JSONs -------
    parser.add_argument(
        "--states_json",
        type=str,
        required=True,
        help="States JSON file produced by the finetuning orchestrator.",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        required=True,
        help="Eval JSON to write (same structure as states_json, plus metrics).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be evaluated, but do not run evals and do not write eval_json.",
    )

    # ------- basic eval args -------
    parser.add_argument(
        "--outdir",
        type=str,
        default="/pscratch/sd/i/ibrahime/checkpoints/",
        help="Output directory (passed to eval_top_tagging)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/pscratch/sd/i/ibrahime/datasets/",
        help="Path to dataset (passed to eval_top_tagging)",
    )
    parser.add_argument(
        "--dataset", type=str, default="top", help="Name of the dataset to load"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="Number of datapoints (override if needed)",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Number of nodes (passed through)"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
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
    parser.add_argument("--task", type=str, default="top_tagging")

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

    with open(args.states_json, "r") as f:
        states = json.load(f)

    prev_eval = load_json_or_empty(args.eval_json)

    eval_out: Dict[str, Any] = {}

    if args.task != "top_tagging":
        raise ValueError("This script currently only supports --task top_tagging.")

    # dryrun stats
    n_skip = 0
    n_run = 0
    n_missing_ckpt = 0
    n_bad_ckpt_path = 0

    for setup_name, setup_dict in states.items():  # e.g. "100k-Top"
        eval_out.setdefault(setup_name, {})
        prev_setup = prev_eval.get(setup_name, {})

        inferred_ds_size = parse_downstream_shots(setup_name)
        setup_dataset_size = (
            inferred_ds_size if inferred_ds_size is not None else args.dataset_size
        )

        for model_size, size_dict in setup_dict.items():  # e.g. "micro"
            eval_out[setup_name].setdefault(model_size, {})
            prev_size = prev_setup.get(model_size, {})

            args.model_size = model_size

            if (
                model_size != "small"
            ):  # model_size == "medium" or model_size == "small":
                continue

            for head, head_dict in size_dict.items():  # e.g. "classifier"
                eval_out[setup_name][model_size].setdefault(head, {})
                prev_head = prev_size.get(head, {})

                for pretrain_label, group_list in head_dict.items():
                    if not isinstance(group_list, list):
                        print(
                            f"[WARN] Expected a list of groups at "
                            f"{setup_name}/{model_size}/{head}/{pretrain_label}, got {type(group_list)}. Skipping."
                        )
                        continue

                    prev_group_list = prev_head.get(pretrain_label, [])
                    prev_group_index = _build_prev_group_index(prev_group_list)

                    new_group_list = []

                    for g_idx, group in enumerate(group_list):
                        if not isinstance(group, dict):
                            print(
                                f"[WARN] Expected dict for group at "
                                f"{setup_name}/{model_size}/{head}/{pretrain_label}[{g_idx}], got {type(group)}. Skipping."
                            )
                            continue

                        new_group = dict(group)
                        new_group_runs: List[Dict[str, Any]] = []
                        new_group["runs"] = new_group_runs

                        pretrain_ckpt = group.get("pretrain_ckpt")
                        states_runs = group.get("runs", [])

                        group_key = _group_match_key(group, g_idx)
                        prev_group = prev_group_index.get(group_key, {})
                        prev_runs = prev_group.get("runs", [])

                        print(
                            f"[GROUP] {setup_name}/{model_size}/{head}/{pretrain_label} "
                            f"group_index={g_idx}, pretrain_ckpt={pretrain_ckpt}"
                        )

                        losses, aucs, invs = [], [], []

                        for run in states_runs:
                            if not isinstance(run, dict):
                                continue

                            ckpt_path = run.get("best_ckpt_dest")
                            run_index = run.get("run_index")

                            new_run = dict(run)
                            status = None

                            matched_prev = _find_prev_run(
                                prev_runs, run_index, ckpt_path
                            )

                            if (
                                matched_prev is not None
                                and _has_complete_metrics(matched_prev)
                                and matched_prev.get("status", "ok") == "ok"
                            ):
                                for k in METRIC_KEYS:
                                    new_run[k] = matched_prev[k]
                                status = "ok"
                                n_skip += 1
                                print(
                                    f"[SKIP EVAL] {setup_name}/{model_size}/{head}/"
                                    f"{pretrain_label}/group{g_idx}/run{run_index} "
                                    f"-> already evaluated."
                                )
                            else:
                                if not ckpt_path or not isinstance(ckpt_path, str):
                                    print(
                                        f"[WARN] Missing/invalid best_ckpt_dest for "
                                        f"{setup_name}/{model_size}/{head}/"
                                        f"{pretrain_label}/group{g_idx}/run{run_index}."
                                    )
                                    status = "no_ckpt_path"
                                    n_bad_ckpt_path += 1
                                elif not os.path.exists(ckpt_path):
                                    print(
                                        f"[WARN] Checkpoint does not exist on disk: {ckpt_path}"
                                    )
                                    status = "missing_file"
                                    n_missing_ckpt += 1
                                else:
                                    args.dataset_size = setup_dataset_size
                                    if args.dry_run:
                                        print(
                                            f"[DRYRUN EVAL] {setup_name}/{model_size}/{head}/"
                                            f"{pretrain_label}/group{g_idx}/run{run_index}\n"
                                            f"              ckpt: {ckpt_path}, dataset_size={args.dataset_size}"
                                        )
                                        status = "dryrun_pending"
                                        n_run += 1
                                    else:
                                        print(
                                            f"[EVAL] {setup_name}/{model_size}/{head}/"
                                            f"{pretrain_label}/group{g_idx}/run{run_index}\n"
                                            f"       ckpt: {ckpt_path}, dataset_size={args.dataset_size}"
                                        )
                                        avg_loss, auc, inv_bkg_eff = eval_top_tagging(
                                            args, ckpt_path, gpuID=args.gpuID
                                        )
                                        new_run["avg_loss"] = float(avg_loss)
                                        new_run["auc"] = float(auc)
                                        new_run["inv_bkg_eff"] = float(inv_bkg_eff)
                                        status = "ok"
                                        n_run += 1

                            new_run["status"] = status

                            if status == "ok" and all(
                                k in new_run for k in METRIC_KEYS
                            ):
                                losses.append(float(new_run["avg_loss"]))
                                aucs.append(float(new_run["auc"]))
                                invs.append(float(new_run["inv_bkg_eff"]))

                            new_group_runs.append(new_run)

                        summary = {
                            "num_runs": len(new_group_runs),
                            "num_evaluated": len(losses),
                        }
                        if len(losses) > 0:
                            m_loss, s_loss = mean_and_std(losses)
                            m_auc, s_auc = mean_and_std(aucs)
                            m_inv, s_inv = mean_and_std(invs)

                            summary["avg_loss"] = {"mean": m_loss, "std": s_loss}
                            summary["auc"] = {"mean": m_auc, "std": s_auc}
                            summary["inv_bkg_eff"] = {"mean": m_inv, "std": s_inv}

                        new_group["metrics_summary"] = summary
                        new_group_list.append(new_group)

                    eval_out[setup_name][model_size][head][pretrain_label] = (
                        new_group_list
                    )

                    if not args.dry_run:
                        with open(args.eval_json, "w") as f:
                            json.dump(eval_out, f, indent=2, sort_keys=True)
                        print(f"\n[DONE] Eval metrics written to {args.eval_json}")

    if args.dry_run:
        print("\n[DRYRUN SUMMARY]")
        print(f"  would_run:        {n_run}")
        print(f"  would_skip:       {n_skip}")
        print(f"  missing_ckpt:     {n_missing_ckpt}")
        print(f"  bad_ckpt_path:    {n_bad_ckpt_path}")
        print("  (dryrun does not write eval_json)")


if __name__ == "__main__":
    main()
