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


def eval_top_tagging(args, ckpt_path, gpuID):
    best_model = PETLightning.load_from_checkpoint(
        ckpt_path, fine_tune=False, ckpt_loaded=ckpt_path
    )
    best_model.eval()

    device = torch.device(f"cuda:{gpuID}" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    data_module = PETDataModule(
        dataset="top",
        path=args.path,
        batch_size=256,
        num_workers=args.num_workers,
        use_pid=args.use_pid,
        use_add=args.use_add,
        num_samples=-1,
    )
    data_module.setup("fit")  # ensures val_dataset is ready; use 'test' if preferred
    dataloader = data_module.val_dataloader()

    def compute_jet_pt_and_mass(jet_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute per-jet transverse momentum and invariant mass, returning both in one tensor.

        Args:
            jet_feats: Tensor of shape [B, P, 4], where the last dim is
                [η, φ, log(pT [MeV]), log(E [MeV])]

        Returns:
            jet_vars: Tensor of shape [B, 2], where
                jet_vars[:, 0] = jet pₜ,
                jet_vars[:, 1] = jet invariant mass.
        """
        # unpack
        eta = jet_feats[..., 0]  # [B, P]
        phi = jet_feats[..., 1]
        log_pT = jet_feats[..., 2]
        log_E = jet_feats[..., 3]

        # recover physical quantities
        pT = torch.exp(log_pT)  # [B, P]
        E = torch.exp(log_E)  # [B, P]

        # momentum components
        px = pT * torch.cos(phi)  # [B, P]
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)  # since pz = pT * sinh(η)

        # sum over constituents
        px_tot = px.sum(dim=1)  # [B]
        py_tot = py.sum(dim=1)
        pz_tot = pz.sum(dim=1)
        E_tot = E.sum(dim=1)

        # transverse momentum
        jet_pt = torch.sqrt(px_tot**2 + py_tot**2)

        # invariant mass: m^2 = E^2 - |p|^2
        mass2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
        mass2 = torch.clamp(
            mass2, min=0.0
        )  # avoid small negative due to numerical noise
        jet_mass = torch.sqrt(mass2)

        # stack into [B, 2]
        jet_vars = torch.stack([jet_pt, jet_mass], dim=1)
        return jet_vars

    # 3) histogram + normalize to probabilities
    def hist_probs(x, bins):
        h, _ = np.histogram(x, bins=bins, density=False)
        p = h.astype(np.float64) + 1e-8  # floor to avoid zeros
        return p / p.sum()

    def kl_div(p, q):
        return np.sum(p * np.log(p / q))

    kl_pt_top = []
    kl_mass_top = []

    kl_pt_not_top = []
    kl_mass_not_top = []

    plot = True
    if plot:
        gen_top = []
        gen_not_top = []
        real_top = []
        real_not_top = []

        jet_feat_gen_top = []
        jet_feat_gen_not_top = []
        jet_feat_real_top = []
        jet_feat_real_not_top = []

    with torch.no_grad():
        cnt = 0
        for batch in tqdm(dataloader, desc="Val Batches"):
            X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
            model_kwargs = {
                key: (batch[key].to(device) if batch[key] is not None else None)
                for key in ["cond", "pid", "add_info"]
                if key in batch
            }

            assert "cond" in model_kwargs, (
                "ERROR, conditioning variables not passed to model"
            )

            pred = generate_class_free(
                best_model.model,
                y,
                X.shape,
                **model_kwargs,
                nsteps=64,
                guidance_scale=3,
            )

            jet_feat_gen = compute_jet_pt_and_mass(pred)
            jet_feat_real = compute_jet_pt_and_mass(X)

            print(pred[y == 0].mean(dim=(0, 1)))
            print(X[y == 0].mean(dim=(0, 1)))
            print(pred[y == 1].mean(dim=(0, 1)))
            print(X[y == 1].mean(dim=(0, 1)))

            # # 2) define shared bins
            # bins_pt   = np.linspace(0, real_pt_top.max().max(), 100)   # adjust as needed
            # bins_mass = np.linspace(0, real_mass_top.max().max(), 100)

            # p_real_pt_top   = hist_probs(real_pt_top, bins_pt)
            # p_fake_pt_top   = hist_probs(fake_pt_top, bins_pt)
            # p_real_mass_top = hist_probs(real_mass_top, bins_mass)
            # p_fake_mass_top = hist_probs(fake_mass_top, bins_mass)

            # p_real_pt_not   = hist_probs(real_pt_not, bins_pt)
            # p_fake_pt_not   = hist_probs(fake_pt_not, bins_pt)
            # p_real_mass_not = hist_probs(real_mass_not, bins_mass)
            # p_fake_mass_not = hist_probs(fake_mass_not, bins_mass)

            # kl_pt_top.append(kl_div(p_real_pt_top,   p_fake_pt_top))
            # kl_mass_top.append(kl_div(p_real_mass_top, p_fake_mass_top))
            # kl_pt_not_top.append(kl_div(p_real_pt_not,   p_fake_pt_not))
            # kl_mass_not_top.append(kl_div(p_real_mass_not, p_fake_mass_not))

            cnt += 1

            if plot:
                gen_top.append(
                    pred[y == 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(pred[y == 1].shape[0] * pred[y == 1].shape[1], -1)
                )
                gen_not_top.append(
                    pred[y == 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(pred[y == 0].shape[0] * pred[y == 0].shape[1], -1)
                )
                real_top.append(
                    X[y == 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(pred[y == 1].shape[0] * pred[y == 1].shape[1], -1)
                )
                real_not_top.append(
                    X[y == 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(pred[y == 0].shape[0] * pred[y == 0].shape[1], -1)
                )

                jet_feat_gen_top.append(jet_feat_gen[y == 1].detach().cpu().numpy())
                jet_feat_gen_not_top.append(jet_feat_gen[y == 0].detach().cpu().numpy())
                jet_feat_real_top.append(jet_feat_real[y == 1].detach().cpu().numpy())
                jet_feat_real_not_top.append(
                    jet_feat_real[y == 0].detach().cpu().numpy()
                )

            if cnt >= 10:
                break

        print(f"KL[pT]   (top jets): {np.mean(kl_pt_top):.4f}")
        print(f"KL[mass] (top jets): {np.mean(kl_mass_top):.4f}")
        print(f"KL[pT]   (not-top):  {np.mean(kl_pt_not_top):.4f}")
        print(f"KL[mass] (not-top):  {np.mean(kl_mass_not_top):.4f}")

        if plot:
            signal = np.concatenate(gen_top, axis=0)
            background = np.concatenate(gen_not_top, axis=0)
            save_path = f"/global/homes/i/ibrahime/FoundationModelStudy/omnilearn_super_gen/plots/hist_gen_num_shot_{num_shots}_{tag}-pretrain-classifier-free3-25_mse.png"
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            plt.title("Generated Jet Feature Distribution")
            axes = axes.flatten()

            for i in range(4):
                ax = axes[i]
                ax.hist(
                    signal[:, i][signal[:, i] != 0],
                    bins=200,
                    alpha=0.5,
                    label="Signal",
                    density=True,
                    color="b",
                )
                ax.hist(
                    background[:, i][background[:, i] != 0],
                    bins=200,
                    alpha=0.5,
                    label="Background",
                    density=True,
                    color="r",
                )
                ax.set_title(f"Feature {i}")
                ax.legend()
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            save_path = f"/global/homes/i/ibrahime/FoundationModelStudy/omnilearn_super_gen/plots/hist_real_num_shot_{num_shots}_{tag}3-25.png"
            signal = np.concatenate(real_top, axis=0)
            background = np.concatenate(real_not_top, axis=0)
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            plt.title("Real Jet Feature Distribution")
            axes = axes.flatten()

            for i in range(4):
                ax = axes[i]
                ax.hist(
                    signal[:, i][signal[:, i] != 0],
                    bins=200,
                    alpha=0.5,
                    label="Signal",
                    density=True,
                    color="b",
                )
                ax.hist(
                    background[:, i][background[:, i] != 0],
                    bins=200,
                    alpha=0.5,
                    label="Background",
                    density=True,
                    color="r",
                )
                ax.set_title(f"Feature {i}")
                ax.legend()
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            signal = np.concatenate(jet_feat_gen_top, axis=0)
            background = np.concatenate(jet_feat_gen_not_top, axis=0)
            save_path = f"/global/homes/i/ibrahime/FoundationModelStudy/omnilearn_super_gen/plots/hist_jet_feat_gen_num_shot_{num_shots}_{tag}-classifier-free3_mse.png"
            fig, axes = plt.subplots(2, figsize=(10, 8))
            plt.title("Generated Jet Feature Distribution")

            for i in range(2):
                ax = axes[i]
                ax.hist(
                    signal[:, i][signal[:, i] != 0],
                    bins=100,
                    alpha=0.5,
                    label="Signal",
                    density=True,
                    color="b",
                )
                ax.hist(
                    background[:, i][background[:, i] != 0],
                    bins=100,
                    alpha=0.5,
                    label="Background",
                    density=True,
                    color="r",
                )
                ax.set_title(f"Feature {i}")
                ax.legend()
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            signal = np.concatenate(jet_feat_real_top, axis=0)
            background = np.concatenate(jet_feat_real_not_top, axis=0)
            save_path = f"/global/homes/i/ibrahime/FoundationModelStudy/omnilearn_super_gen/plots/hist_jet_feat_real_num_shot_{num_shots}_{tag}.png"
            fig, axes = plt.subplots(2, figsize=(10, 8))
            plt.title("Generated Jet Feature Distribution")

            for i in range(2):
                ax = axes[i]
                ax.hist(
                    signal[:, i][signal[:, i] != 0],
                    bins=100,
                    alpha=0.5,
                    label="Signal",
                    density=True,
                    color="b",
                )
                ax.hist(
                    background[:, i][background[:, i] != 0],
                    bins=100,
                    alpha=0.5,
                    label="Background",
                    density=True,
                    color="r",
                )
                ax.set_title(f"Feature {i}")
                ax.legend()
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return (0, 0, 0, 0)
