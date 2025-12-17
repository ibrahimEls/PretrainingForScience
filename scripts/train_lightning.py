import argparse
import os

import torch
from omnilearned.utils import get_model_parameters
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from omnilearn_lightning.dataloader import PETDataModule
from omnilearn_lightning.model import PETLightning
from omnilearn_lightning.utils import (
    extract_pretrained_ckpt_prefix,
    get_version_number,
    load_partial_checkpoint,
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
        help="Output directory",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to the model checkpoint to load",
    )

    # Data & training params
    parser.add_argument(
        "--dataset", type=str, default="jetclass", help="Name of the dataset to load"
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of Nodes")
    parser.add_argument("--path", type=str, help="Path to dataset")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--dataset_size", type=int, default=100_000, help="dataset_size"
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="Number of validation batches to use. If <1.0, this is a fraction of the total val batches.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=None,
        help="Number of training batches between each validation check. If <1.0, this is a fraction of the total training batches.",
    )
    parser.add_argument("--shuffle_val_test_indices", type=str2bool, default=False)
    parser.add_argument("--seed_for_initial_shuffling", type=int, default=None)
    parser.add_argument(
        "--scheduler_warmup_steps",
        type=int,
        default=1_000,
        help="Number of steps for learning rate warmup.",
    )
    parser.add_argument(
        "--scheduler_total_steps",
        type=int,
        default=500_000,
        help="Number of steps after which the learning rate scheduler reaches minimum LR. Will go back up afterwards.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Model hyper-parameters
    parser.add_argument("--input_dim", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_size", type=str, default="micro")
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--mlp_drop", type=float, default=0.0)
    parser.add_argument("--feature_drop", type=float, default=0.1)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        help="Training mode (classifier/generator/other)",
    )
    parser.add_argument("--resume", type=str2bool, default=False)

    # Training hyperparams
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--b1", type=float, default=0.95, help="Beta1 for optimizer")
    parser.add_argument("--b2", type=float, default=0.99, help="Beta2 for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--use_clip", action="store_true", help="Use clip loss or not")
    parser.add_argument(
        "--use_amp", action="store_true", help="Use 16-bit precision training (AMP)"
    )

    # MPM-specific
    parser.add_argument("--tokenizer_ckpt", type=str, default=None)
    parser.add_argument(
        "--pos_encoding_type",
        type=str,
        default="sort_descending_in_masked_subset",
        help=(
            "Type of positional encoding to use. Options are "
            "'sort_descending_in_masked_subset', 'sort_descending_all', or None."
        ),
    )
    parser.add_argument(
        "--masking_fraction", type=float, default=0.4, help="Masking fraction for MPM"
    )
    parser.add_argument(
        "--use_weights_for_mpm",
        type=str2bool,
        default=True,
        help="Use weights for masked continuous loss",
    )
    parser.add_argument(
        "--mpm_features",
        type=str,
        default="kin",
        help="Features to use for MPM ('kin', 'kin_pid_add')",
    )

    # Additional features
    parser.add_argument("--use_pid", type=str2bool, default=False)
    parser.add_argument("--use_add", type=str2bool, default=False)
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--fine_tune", type=str2bool, default=False)
    parser.add_argument("--use_one_cycle", type=str2bool, default=False)

    # Logging
    parser.add_argument(
        "--use_wandb", type=str2bool, default=True, help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="omnilearned",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="omnilearn-studies",
        help="Weights & Biases entity name",
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
        save_tag = f"_micro_{args.mode}_{args.dataset}_dataset_{args.dataset_size}"
        model_params = {}
        model_params["num_transformers"] = 3
        model_params["num_transformers_head"] = 2
        model_params["num_tokens"] = 4
        model_params["num_heads"] = 4
        model_params["K"] = 10
        model_params["base_dim"] = 32
        model_params["mlp_ratio"] = 2

    elif args.model_size == "small":
        model_params = get_model_parameters(args.model_size)
    elif args.model_size == "medium":
        model_params = get_model_parameters(args.model_size)
    elif args.model_size == "large":
        model_params = get_model_parameters(args.model_size)

    user = os.getenv("USER")

    save_tag = f"_{args.model_size}_{args.mode}_{args.dataset}_dataset_{args.dataset_size}_{user}"

    ckpt_tag = ""
    if args.dataset != "jetclass":
        if args.fine_tune:
            ckpt_tag = extract_pretrained_ckpt_prefix(args.ckpt)
        else:
            ckpt_tag = "from_scratch"
        save_tag = save_tag + "_ckpt_" + ckpt_tag

    # from omnilearn_lightning.callbacks.masked_prediction_callback import (
    #     MaskedPredictionCallback,
    # )
    # masked_prediction_callback = MaskedPredictionCallback()

    # Create output directory only on rank 0
    if rank_zero_only.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)

    # Set random seed for reproducibility
    if args.seed is not None:
        seed_everything(args.seed, workers=True)

    data_module = PETDataModule(
        dataset=args.dataset,
        path=args.path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Datamodule expects use_pid and use_add to be True for jetclass dataset
        # as it otherwise doesn't split into pid/add info features (then `X`
        # would have wrong input dim)
        use_pid=True if args.dataset == "jetclass" else args.use_pid,
        use_add=True if args.dataset == "jetclass" else args.use_add,
        num_samples=args.dataset_size,
        shuffle_train_indices=True,  # always shuffle indices cause then we have balanced classes no matter which dataset size
        shuffle_val_test_indices=args.shuffle_val_test_indices,
        seed_for_initial_shuffling=args.seed_for_initial_shuffling,
        load_val=True,
        use_cond=False,
    )

    print(f"Model mode: {args.mode}")

    model = PETLightning(
        num_feat=args.input_dim,
        num_classes=args.num_classes,
        attn_drop=args.attn_drop,
        mlp_drop=args.mlp_drop,
        feature_drop=args.feature_drop,
        num_tokens=args.num_tokens,
        radius=args.radius,
        lr=args.lr,
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
        tokenizer_ckpt=args.tokenizer_ckpt,
        pos_encoding_type=args.pos_encoding_type
        if args.pos_encoding_type != "None"
        else None,
        total_steps=args.scheduler_total_steps,
        warmup_steps=args.scheduler_warmup_steps,
        use_one_cycle=args.use_one_cycle,
        model_params=model_params,
        masking_fraction=args.masking_fraction,
        fine_tune=args.fine_tune,
        all_head=True
        if (
            "class" not in ckpt_tag
            and "pretrain" not in ckpt_tag
            and "scratch" not in "ckpt_tag"
            and args.fine_tune
        )
        else False,
        use_weights_for_mpm=args.use_weights_for_mpm,
        mpm_features=args.mpm_features,
    )

    if rank_zero_only.rank == 0:
        print("Model initialized.")
        print(model)

    if args.resume:
        ckpt_path = args.ckpt

    pseudo_epoch_len = int(1_000_000 / (args.batch_size * 4 * 10)) // 10

    out_dir_save_tag = os.path.join(args.outdir, save_tag)

    version = get_version_number(out_dir_save_tag)
    run_name = f"v{version}{save_tag}"  # _{get_bigram(add_timestamp=True)}"

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
        hparams.update({"run_dir": run_dir, "run_name": run_name})
        # add SLURM job id if it exists
        if "SLURM_JOB_ID" in os.environ:
            hparams["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            tags=args.wandb_tag,
            save_dir=args.outdir,
            entity=args.wandb_entity,
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
        verbose=False,
    )

    ckpt_val = ModelCheckpoint(
        filename=f"{save_tag}-{{epoch:06d}}-{{step:06d}}-{{val_loss:.4f}}-{{train_loss_step:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        # we want to save this checkpoint after each *validation* check
        save_on_train_epoch_end=False,
        every_n_epochs=None,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)

    callbacks = [checkpoint_step, ckpt_val, lr_monitor]

    # if "mpm" in args.mode or args.mode == "pretrain":
    #     print("Using MaskedPredictionCallback for self-supervised learning")
    #     callbacks.append(masked_prediction_callback)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )
    callbacks.append(early_stop_callback)

    if args.dataset != "jetclass":
        if not args.fine_tune and not args.resume:
            print("Training Downstream From Scratch")
        elif not args.fine_tune and args.resume:
            print("Resuming Downstream From Scratch")
        elif args.fine_tune and not args.resume:
            print(f"Loading Model From {args.ckpt}")
            model = load_partial_checkpoint(model, args.ckpt, task=args.dataset)
            print("Finetuning Model on Downstream")
        elif args.fine_tune and args.resume:
            print("Resuming Downstream Finetuning")

    if args.val_check_interval is not None:
        if args.val_check_interval <= 1.0:
            args.val_check_interval = float(args.val_check_interval)
        else:
            args.val_check_interval = int(args.val_check_interval)

    trainer = Trainer(
        max_epochs=args.epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=4 if torch.cuda.is_available() else None,
        precision=16 if args.use_amp else 32,
        callbacks=callbacks,
        default_root_dir=run_dir,
        logger=loggers,
        strategy=strategy,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
        num_nodes=args.num_nodes,
        enable_progress_bar=(args.num_nodes == 1),
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
    )

    # Training
    trainer.fit(model, data_module, ckpt_path=ckpt_path if args.resume else None)
    print("Training is complete!")
    print(f"Best model checkpoint: {ckpt_val.best_model_path}")

    # Testing
    if args.test:
        best_model_path = checkpoint_step.best_model_path
        print(f"Using best model from {best_model_path} for testing")
        trainer.test(model=model, datamodule=data_module, ckpt_path=best_model_path)
        print("Testing is complete!")


if __name__ == "__main__":
    main()
