import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from ..dataloader import PETDataModule
from ..models.model_super_only import PETLightning
from ..utils import get_bigram, get_version_number, load_partial_checkpoint


def top_tagging_task(ckpt_path, args, tag, num_shots=100, gpuID=0):
    print(f"Training Model for top tagging on {num_shots} shots ")
    data_module = PETDataModule(
        dataset="top",
        path=args.path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_pid=args.use_pid,
        use_add=args.use_add,
        num_samples=num_shots,
    )

    model = PETLightning(
        input_dim=4,
        num_classes=2,
        hidden_size=args.hidden_size,
        num_transformers=args.num_transformers,
        num_heads=args.num_heads,
        attn_drop=args.attn_drop,
        mlp_drop=args.mlp_drop,
        mlp_ratio=args.mlp_ratio,
        feature_drop=args.feature_drop,
        num_tokens=args.num_tokens,
        K=args.K,
        radius=args.radius,
        lr=args.lr,
        lr_factor=args.lr_factor,
        b1=0.95,
        b2=0.99,
        weight_decay=args.weight_decay,
        use_clip=False,
        mode="classifier",
        use_amp=args.use_amp,
        use_pid=args.use_pid,
        use_add=args.use_add,
        fine_tune=True,
        ckpt_loaded=ckpt_path,
    )

    patience_val = 20
    if args.resume or num_shots > 100_000:
        patience_val = 3
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience_val,
        mode="min",
        verbose=False,
    )

    tag = tag + f"_top_{num_shots}-shot"
    print(f"Training with {num_shots}! Setting Patience to: {patience_val}")

    if args.from_scratch:
        print("Training From Scratch")
    else:
        print(f"Loading Model From {ckpt_path}")
        model = load_partial_checkpoint(model, ckpt_path, task="top")
        print(f"Finetuning Model")

    loggers = []
    out_dir_save_tag = os.path.join(args.outdir, tag)
    version = get_version_number(out_dir_save_tag)
    run_name = f"v{version}_{get_bigram(add_timestamp=True)}"
    run_dir = os.path.join(out_dir_save_tag, run_name)

    csv_logger = CSVLogger(save_dir=args.outdir, name=tag, flush_logs_every_n_steps=5)
    loggers.append(csv_logger)

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

    checkpoint_callback = ModelCheckpoint(
        filename=tag + "-{epoch:06d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        verbose=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    max_epochs = 100
    strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=4 if torch.cuda.is_available() else None,
        precision=16 if args.use_amp else 32,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        default_root_dir=args.outdir,
        logger=loggers,
        strategy=strategy,
        accumulate_grad_batches=2,
        num_nodes=args.num_nodes,
        enable_progress_bar=(args.num_nodes == 1),
    )

    # Train the model
    trainer.fit(model, data_module)

    # After training, load the best model based on validation loss
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    best_model = model.__class__.load_from_checkpoint(
        best_model_path, fine_tune=False, ckpt_loaded=best_model_path
    )
    best_model.eval()

    device = torch.device(f"cuda:{gpuID}" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    # Define loss
    criterion = torch.nn.BCEWithLogitsLoss()

    all_labels = []
    all_scores = []
    total_loss = 0.0
    n_batches = 0

    data_module = PETDataModule(
        dataset="top",
        path=args.path,
        batch_size=256,
        num_workers=args.num_workers,
        use_pid=args.use_pid,
        use_add=args.use_add,
        num_samples=-1,
    )

    data_module.setup("fit")
    dataloader = data_module.val_dataloader()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val Batches"):
            inputs, labels = batch["X"], batch["y"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            model_kwargs = {
                key: (batch[key].to(device) if batch[key] is not None else None)
                for key in ["cond", "pid", "add_info"]
                if key in batch
            }
            out = best_model(inputs, labels, **model_kwargs)
            y_pred = out["y_pred"].softmax(dim=1)
            outputs = ((1 - y_pred[:, 0]) + (y_pred[:, 1])) / 2

            loss = criterion(outputs.unsqueeze(1), labels.unsqueeze(1).float())
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_scores.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / n_batches
    auc = roc_auc_score(all_labels, all_scores)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    target_eff = 0.3
    idx = np.where(tpr >= target_eff)[0][0]
    bkg_eff = fpr[idx]
    inv_bkg_eff = 1.0 / bkg_eff if bkg_eff > 0 else float("inf")

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"1/(background efficiency): {inv_bkg_eff:.4f}")

    return (avg_loss, auc, inv_bkg_eff)
