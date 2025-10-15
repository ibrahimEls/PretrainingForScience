import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from omnilearn_lightning.dataloader import PETDataModule
from omnilearn_lightning.utils import (
    get_bigram,
    get_latest_checkpoint_dir,
    get_version_number,
)


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Train PET model in PyTorch Lightning")

    # Basic I/O
    parser.add_argument(
        "--outdir",
        type=str,
        default="/pscratch/sd/i/ibrahime/checkpoints/pretrained_micro_super_gen/",
        help="Output directory",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/pscratch/sd/i/ibrahime/checkpoints/lightning_logs/version_151/checkpoints/last.ckpt",
        help="Path to the model checkpoint to load",
    )

    # Data & training params
    parser.add_argument(
        "--dataset", type=str, default="jetclass", help="Name of the dataset to load"
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of Nodes")
    parser.add_argument(
        "--path",
        type=str,
        default="/pscratch/sd/i/ibrahime/datasets/",
        help="Path to dataset",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--dataset_size", type=int, default=100_000, help="dataset_size"
    )

    # Model hyper-parameters
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_size", type=str, default="micro")
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--mlp_drop", type=float, default=0.0)
    parser.add_argument("--mlp_ratio", type=int, default=2)
    parser.add_argument("--feature_drop", type=float, default=0.1)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        help="Training mode (classifier/generator/other)",
    )
    parser.add_argument("--resume", action="store_true", help="Use clip loss or not")

    # Training hyperparams
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--b1", type=float, default=0.95, help="Beta1 for optimizer")
    parser.add_argument("--b2", type=float, default=0.99, help="Beta2 for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_clip", action="store_true", help="Use clip loss or not")
    parser.add_argument(
        "--use_amp", action="store_true", help="Use 16-bit precision training (AMP)"
    )

    parser.add_argument("--pretraining_mode", type=str, default="super-gen")

    # Additional features
    parser.add_argument("--use_pid", type=str2bool, default=False)
    parser.add_argument("--use_add", type=str2bool, default=False)
    parser.add_argument("--test", type=str2bool, default=False)

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
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="Weights & Biases tags for the run",
    )

    args = parser.parse_args()

    if args.model_size == "micro":
        hidden_size = 32
        num_transformers = 3
        num_heads = 4
        batch_size = 256
        lr = 1e-3
        save_tag = f"_micro_{args.dataset_size}"

    elif args.model_size == "small":
        hidden_size = 128
        num_transformers = 8
        num_heads = 8
        batch_size = 128
        lr = 1e-4
        save_tag = f"_small_{args.dataset_size}"

    elif args.model_size == "medium":
        hidden_size = 512
        num_transformers = 12
        num_heads = 16
        batch_size = 32
        lr = 1e-5
        save_tag = f"_medium_{args.dataset_size}"

    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    if args.pretraining_mode == "super-gen":
        from omnilearn_lightning.models.model_super_gen import PETLightning

        save_tag = "super-gen" + save_tag

    elif args.pretraining_mode == "super-only":
        from omnilearn_lightning.models.model_super_only import PETLightning

        save_tag = "super-only" + save_tag

    elif args.pretraining_mode == "gen-only":
        from omnilearn_lightning.models.model_gen_only import PETLightning

        save_tag = "gen-only" + save_tag

    elif args.pretraining_mode == "self-super":
        from omnilearn_lightning.models.model_self_super import PETLightning

        save_tag = "self-super" + save_tag

    elif args.pretraining_mode == "naive-self-super":
        from omnilearn_lightning.models.model_naive_self_super import PETLightning

        save_tag = "naive-self-super" + save_tag

    else:
        raise ValueError(f"Unknown pretraining mode: {args.pretraining_mode}")

    # Create output directory only on rank 0
    if rank_zero_only.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)

    batch_size = args.batch_size

    data_module = PETDataModule(
        dataset=args.dataset,
        path=args.path,
        batch_size=batch_size,
        num_workers=args.num_workers,
        use_pid=args.use_pid,
        use_add=args.use_add,
        num_samples=args.dataset_size,
    )

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

    if args.resume:
        ckpt_path = get_latest_checkpoint_dir(
            base_dir=os.path.join(args.outdir, save_tag)
        )

    pseudo_epoch_len = int(1_000_000 / (batch_size * 4 * 10)) // 10

    out_dir_save_tag = os.path.join(args.outdir, save_tag)

    version = get_version_number(out_dir_save_tag)
    run_name = f"v{version}_{get_bigram(add_timestamp=True)}"

    run_dir = os.path.join(out_dir_save_tag, run_name)
    if rank_zero_only.rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        print(f"Output directory of this run: {run_dir}")

    # Configure loggers
    loggers = []

    # Always include CSV logger for local logging
    csv_logger = CSVLogger(save_dir=run_dir, name=None, version="")
    loggers.append(csv_logger)

    # Add wandb logger if requested
    if args.use_wandb:
        hparams = {
            k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
            for k, v in vars(args).items()
        }
        hparams.update({"run_dir": run_dir})
        # add SLURM job id if it exists
        if "SLURM_JOB_ID" in os.environ:
            hparams["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            tags=args.wandb_tags,
            save_dir=args.outdir,
        )

        wandb_logger.log_hyperparams(hparams)
        loggers.append(wandb_logger)

    checkpoint_step = ModelCheckpoint(
        filename=save_tag + "-{step:06d}-{train_loss_step:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=5,
        every_n_train_steps=pseudo_epoch_len,
        save_last=True,
    )

    ckpt_val = ModelCheckpoint(
        filename=f"{save_tag}-epoch{{epoch:06d}}",
        monitor="train_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)

    trainer = Trainer(
        max_epochs=args.epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=4 if torch.cuda.is_available() else None,
        precision=16 if args.use_amp else 32,
        callbacks=[checkpoint_step, ckpt_val, lr_monitor],
        default_root_dir=run_dir,
        logger=loggers,
        strategy=strategy,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
        num_nodes=args.num_nodes,
        enable_progress_bar=(args.num_nodes == 1),
        limit_val_batches=0,
    )

    # Training
    trainer.fit(model, data_module, ckpt_path=ckpt_path if args.resume else None)
    print("Training is complete!")
    print(f"Best model checkpoint: {checkpoint_step.best_model_path}")

    # Testing
    if args.test:
        best_model_path = checkpoint_step.best_model_path
        print(f"Using best model from {best_model_path} for testing")
        trainer.test(model=model, datamodule=data_module, ckpt_path=best_model_path)
        print("Testing is complete!")


if __name__ == "__main__":
    main()
