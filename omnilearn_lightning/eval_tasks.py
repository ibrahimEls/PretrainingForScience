import numpy as np
import torch
from dataloader import PETDataModule
from model import PETLightning
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def eval_top_tagging(args, ckpt_path, gpuID):
    best_model = PETLightning.load_from_checkpoint(
        ckpt_path, fine_tune=False, ckpt_loaded=ckpt_path
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
        batch_size=args.batch_size,
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


def eval_quark_gluon(args, ckpt_path, gpuID):
    best_model = PETLightning.load_from_checkpoint(
        ckpt_path, fine_tune=False, ckpt_loaded=ckpt_path
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
        dataset="qg",
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
