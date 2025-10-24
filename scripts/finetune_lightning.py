import argparse
import sys

sys.path.append("../")

from omnilearn_lightning.tasks.diffusion_task import diffusion_task
from omnilearn_lightning.tasks.quarkgluon_task import quark_gloun_task
from omnilearn_lightning.tasks.toptagging_task import top_tagging_task


def main():
    parser = argparse.ArgumentParser(description="Train PET model in PyTorch Lightning")

    # Basic I/O
    parser.add_argument(
        "--outdir",
        type=str,
        default="/pscratch/sd/i/ibrahime/checkpoints/",
        help="Output directory",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/pscratch/sd/i/ibrahime/checkpoints/",
        help="ckpt path",
    )
    parser.add_argument(
        "--save_tag", type=str, default="pet_model", help="Checkpoint filename tag"
    )

    # Data & training params
    parser.add_argument(
        "--dataset", type=str, default="pretrain", help="Name of the dataset to load"
    )
    parser.add_argument(
        "--dataset_size", type=int, default=-1, help="Number of datapoints"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Name of the dataset to load"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/pscratch/sd/i/ibrahime/datasets/",
        help="Path to dataset",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Number of DataLoader workers"
    )

    # Model hyper-parameters
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=201)
    parser.add_argument("--hidden_size", type=int, default=128)  # 128, 32
    parser.add_argument("--num_transformers", type=int, default=8)  # 8 ,3
    parser.add_argument("--num_heads", type=int, default=8)  # 8,4
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--mlp_drop", type=float, default=0.0)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--feature_drop", type=float, default=0.1)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument("--resume", action="store_true", help="Use clip loss or not")
    parser.add_argument("--model_size", type=str, default="micro")
    parser.add_argument("--task", type=str, default="top_tagging")

    # Training hyperparams
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
    parser.add_argument("--from_scratch", action="store_true")

    # Logging
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="omnilearned",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_tag",
        type=str,
        action="append",
        default=None,
        help="Weights & Biases tags for the run (can be specified multiple times)",
    )
    args = parser.parse_args()

    if args.model_size == "micro":
        args.hidden_size = 32
        args.num_transformers = 3
        args.num_heads = 4
        args.batch_size = 256
        args.lr = 1e-3
        tag = f"super_gen_micro_{args.task}_{args.dataset_size}"

    elif args.model_size == "small":
        args.hidden_size = 128
        args.num_transformers = 8
        args.num_heads = 8
        args.batch_size = 128
        args.lr = 1e-4
        tag = f"super_gen_small_{args.task}_{args.dataset_size}"

    elif args.model_size == "medium":
        args.hidden_size = 512
        args.num_transformers = 12
        args.num_heads = 16
        args.batch_size = 32
        args.lr = 1e-5
        tag = f"super_gen_medium_{args.task}_{args.dataset_size}"

    if args.task == "top_tagging":
        metrics = top_tagging_task(
            args.ckpt, args, tag, num_shots=args.dataset_size, gpuID=0
        )

    if args.task == "quark_gluon":
        metrics = quark_gloun_task(
            args.ckpt, args, tag, num_shots=args.dataset_size, gpuID=0
        )

    if args.task == "diffusion_task":
        metrics = diffusion_task(
            args.ckpt, args, tag, num_shots=args.dataset_size, gpuID=0
        )

    return metrics


if __name__ == "__main__":
    main()
