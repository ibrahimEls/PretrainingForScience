import argparse
import os
import sys

sys.path.append("../")

from omnilearn_lightning.tasks.toptagging_task import eval_top_tagging


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
        default="/pscratch/sd/i/ibrahime/checkpoints/FinishedCheckPoints/",
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
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
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

    args = parser.parse_args()

    if args.task == "top_tagging":
        args.ckpt = args.ckpt + "/TopTagging/"
        args.ckpt = args.ckpt + f"/{args.model_size}/"

        for model in os.listdir(args.ckpt):
            if model.endswith(".ckpt") and "2" in model:
                model_path = os.path.join(args.ckpt, model)
                print(f"Evaluating {model_path}!")
                metrics = eval_top_tagging(args, model_path, gpuID=0)

    return metrics


if __name__ == "__main__":
    main()
