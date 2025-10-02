#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
from lightining_model import PETLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from dataloader import PETDataModule


def parse_case(s: str) -> Tuple[int, int]:
    """
    Parse case spec like:
      - '1g'     -> (num_nodes=1, gpus_per_node=1)
      - '4g'     -> (1,4)
      - '2n4g'   -> (2,4)
      - '3n8g'   -> (3,8)
    """
    s = s.lower().strip()
    if s.endswith("g") and "n" not in s:
        g = int(s[:-1])
        return 1, g
    if "n" in s and s.endswith("g"):
        n_str, g_str = s.split("n", 1)
        n = int(n_str)
        g = int(g_str[:-1])
        return n, g
    raise ValueError(f"Unrecognized case format: {s}")


def case_name(num_nodes: int, gpus_per_node: int) -> str:
    if num_nodes == 1:
        return f"{gpus_per_node}G"
    return f"{num_nodes}N{gpus_per_node}G"


@dataclass
class BenchResult:
    case: str
    num_nodes: int
    gpus_per_node: int
    total_gpus: int
    batch_size_per_gpu: int
    global_batch_size: int
    steps_measured: int
    warmup_steps: int
    wall_time_s: float
    steps_per_s: float
    samples_per_s: float
    speedup_vs_1gpu: float = math.nan
    efficiency_vs_linear: float = math.nan


class ThroughputCallback(Callback):
    """
    Warm up for warmup_steps, then measure timing for exactly measure_steps training steps.
    Computes steps/sec and effective samples/sec given global batch size.
    """

    def __init__(self, warmup_steps: int, measure_steps: int, global_bs: int):
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.measure_steps = int(measure_steps)
        self.global_bs = int(global_bs)

        self._start = None
        self._measured = 0
        self._done = False
        self.metrics = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gs = trainer.global_step
        if self._done:
            return

        # Start timing after warmup completed
        if gs == self.warmup_steps and self._start is None:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self._start = time.perf_counter()
            self._measured = 0

        if self._start is not None and self._measured < self.measure_steps:
            self._measured += 1
            if self._measured == self.measure_steps:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.perf_counter() - self._start
                steps_per_s = (
                    self.measure_steps / elapsed if elapsed > 0 else float("nan")
                )
                samples_per_s = steps_per_s * self.global_bs
                self.metrics = {
                    "wall_time_s": elapsed,
                    "steps_per_s": steps_per_s,
                    "samples_per_s": samples_per_s,
                }
                self._done = True
                trainer.should_stop = True

    def get_metrics(self):
        return self.metrics or {}


def build_dm(args) -> PETDataModule:
    return PETDataModule(
        dataset=args.dataset,
        path=args.path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_pid=args.use_pid,
        use_add=args.use_add,
        num_samples=args.dataset_size,
    )


def build_model(args):
    if args.model_size == "micro":
        hidden_size, num_transformers, num_heads, lr = 32, 3, 4, 1e-3
    elif args.model_size == "small":
        hidden_size, num_transformers, num_heads, lr = 128, 8, 8, 1e-3
    elif args.model_size == "medium":
        hidden_size, num_transformers, num_heads, lr = 512, 12, 16, 1e-5
    else:
        raise ValueError(f"Unknown model_size: {args.model_size}")

    model = PETLightning(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_size=hidden_size,
        num_transformers=num_transformers,
        num_heads=num_heads,
        attn_drop=args.attn_drop,
        mlp_drop=args.mlp_drop,
        mlp_ratio=args.mlp_ratio,
        feature_drop=args.feature_drop,
        num_tokens=args.num_tokens,
        K=args.K,
        radius=args.radius,
        lr=lr,
        lr_factor=args.lr_factor,
        b1=args.b1,
        b2=args.b2,
        weight_decay=args.weight_decay,
        use_clip=args.use_clip,
        mode=args.mode,
        use_amp=args.use_amp,
        use_pid=args.use_pid,
        use_add=args.use_add,
        ckpt_loaded=args.ckpt,
    )
    return model


def run_one_case(
    args, num_nodes: int, gpus_per_node: int, out_root: str
) -> BenchResult:
    total_gpus = num_nodes * gpus_per_node
    devices = gpus_per_node
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = 16 if args.use_amp else 32

    case = case_name(num_nodes, gpus_per_node)
    case_dir = os.path.join(out_root, case)
    os.makedirs(case_dir, exist_ok=True)
    logger = CSVLogger(save_dir=case_dir, name="lightning")

    dm = build_dm(args)
    model = build_model(args)

    global_bs = args.batch_size * max(1, total_gpus)
    cb = ThroughputCallback(args.warmup_steps, args.measure_steps, global_bs)

    print(f"Training with {total_gpus} GPUs with {devices} GPU per node")
    print(f"Effective Batch Size {global_bs}")

    strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)

    if gpus_per_node == 1 and num_nodes == 1:
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            max_steps=args.warmup_steps + args.measure_steps + 5,
            limit_train_batches=args.warmup_steps + args.measure_steps + 5,
            limit_val_batches=0,
            precision=precision,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[cb],
            default_root_dir=case_dir,
        )
    else:
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            strategy=strategy,
            logger=logger,
            max_steps=args.warmup_steps + args.measure_steps + 5,
            limit_train_batches=args.warmup_steps + args.measure_steps + 5,
            limit_val_batches=0,
            enable_checkpointing=False,
            precision=precision,
            callbacks=[cb],
            default_root_dir=case_dir,
            accumulate_grad_batches=2,
        )

    trainer.fit(model, dm)
    print("Finished Training!")
    m = cb.get_metrics()

    res = BenchResult(
        case=case,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        total_gpus=total_gpus,
        batch_size_per_gpu=args.batch_size,
        global_batch_size=global_bs,
        steps_measured=args.measure_steps,
        warmup_steps=args.warmup_steps,
        wall_time_s=float(m["wall_time_s"]),
        steps_per_s=float(m["steps_per_s"]),
        samples_per_s=float(m["samples_per_s"]),
    )

    with open(os.path.join(case_dir, "bench_raw.json"), "w") as f:
        json.dump(asdict(res), f, indent=2)

    return res


@rank_zero_only
def finalize(results: List[BenchResult], out_root: str, png_name: str, csv_name: str):
    import csv

    for r in results:
        r.speedup_vs_1gpu = math.nan
        r.efficiency_vs_linear = math.nan

    # Write CSV
    csv_path = os.path.join(out_root, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "case",
            "num_nodes",
            "gpus_per_node",
            "total_gpus",
            "batch_size_per_gpu",
            "global_batch_size",
            "steps_measured",
            "warmup_steps",
            "wall_time_s",
            "steps_per_s",
            "samples_per_s",
            "speedup_vs_1gpu",
            "efficiency_vs_linear_percent",
        ]
        writer.writerow(header)
        for r in results:
            writer.writerow(
                [
                    r.case,
                    r.num_nodes,
                    r.gpus_per_node,
                    r.total_gpus,
                    r.batch_size_per_gpu,
                    r.global_batch_size,
                    r.steps_measured,
                    r.warmup_steps,
                    f"{r.wall_time_s:.6f}",
                    f"{r.steps_per_s:.4f}",
                    f"{r.samples_per_s:.2f}",
                    f"{r.speedup_vs_1gpu:.3f}",
                    f"{r.efficiency_vs_linear:.1f}",
                ]
            )

    results_sorted = sorted(results, key=lambda x: x.total_gpus)

    print("\n== Scaling Results ==")
    for r in results_sorted:
        print(
            f"{r.case:>6s} | {r.total_gpus:2d} GPUs | {r.samples_per_s:10.1f} samples/s | "
            f"speedup {r.speedup_vs_1gpu:5.2f} | efficiency {r.efficiency_vs_linear:5.1f}%"
        )
    print(f"\nSaved: {csv_path}")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark scaling (1 GPU, 4 GPU, multinode) for PET model"
    )
    p.add_argument(
        "--outdir", type=str, required=True, help="Directory to store results"
    )
    p.add_argument("--cases", type=str, default="2n4g", help="Cases")
    p.add_argument(
        "--warmup_steps", type=int, default=20, help="Warmup steps before measuring"
    )
    p.add_argument(
        "--measure_steps", type=int, default=100, help="Measured steps (per case)"
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar (only for single-node cases)",
    )

    p.add_argument("--dataset", type=str, default="jetclass")
    p.add_argument("--path", type=str, default="/pscratch/sd/i/ibrahime/datasets/")
    p.add_argument("--dataset_size", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)  # per GPU
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--input_dim", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument(
        "--model_size", type=str, default="micro", choices=["micro", "small", "medium"]
    )
    p.add_argument("--attn_drop", type=float, default=0.0)
    p.add_argument("--mlp_drop", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=int, default=2)
    p.add_argument("--feature_drop", type=float, default=0.1)
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--radius", type=float, default=0.4)
    p.add_argument("--mode", type=str, default="pretrain")
    p.add_argument("--ckpt", type=str, default=None)

    p.add_argument("--lr_factor", type=float, default=0.1)
    p.add_argument("--b1", type=float, default=0.95)
    p.add_argument("--b2", type=float, default=0.99)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--use_clip", action="store_true")
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--use_pid", action="store_true")
    p.add_argument("--use_add", action="store_true")

    args = p.parse_args()
    out_dir = args.outdir + "_" + args.model_size
    os.makedirs(out_dir, exist_ok=True)

    # Parse cases
    case_specs = [c.strip() for c in args.cases.split(",") if c.strip()]
    parsed: List[Tuple[int, int]] = [parse_case(c) for c in case_specs]
    print(parsed)

    # Run ONE cases
    results: List[BenchResult] = []
    for num_nodes, gpus_per_node in parsed:
        res = run_one_case(args, num_nodes, gpus_per_node, out_dir)
        results.append(res)
        break

    finalize(
        results, out_dir, png_name="scaling_plot.png", csv_name="scaling_results.csv"
    )


if __name__ == "__main__":
    main()
