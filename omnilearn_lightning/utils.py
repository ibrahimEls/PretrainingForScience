import os
import re
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from sklearn import metrics
from torch.distributed import get_rank, init_process_group
from wonderwords import RandomWord


def load_partial_checkpoint(model, ckpt_path, task="top"):
    if task == "top":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_state_dict = ckpt["state_dict"]
        filtered_state = {
            key: value
            for key, value in ckpt_state_dict.items()
            if ("classifier.out" not in key)
        }

        model.load_state_dict(filtered_state, strict=False)

    elif task == "qc":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_state_dict = ckpt["state_dict"]
        filtered_state = {
            key: value
            for key, value in ckpt_state_dict.items()
            if ("body.embed.mlp.fc1" not in key)
            and ("classifier.out" not in key)
            and ("body.local_physics.mlp.fc1" not in key)
            and ("body.embed.norm.weight" not in key)
        }
        model.load_state_dict(filtered_state, strict=False)

    elif task == "gen":
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_state_dict = ckpt["state_dict"]
        filtered_state = {
            key: value
            for key, value in ckpt_state_dict.items()
            if (
                "generator.pid_embed.0.weight" not in key
                and "classifier.out" not in key
            )
        }
        model.load_state_dict(filtered_state, strict=False)

    return model


def get_latest_checkpoint_dir(base_dir: str) -> str:
    """
    Given a base directory containing subdirectories named 'version_{n}',
    finds the highest n and returns the path to its 'checkpoints' subdirectory.
    Raises a FileNotFoundError if no version_* directories are found or
    if the checkpoints subdirectory does not exist.
    """
    version_pattern = re.compile(r"^version_(\d+)$")
    max_version = -1

    # List all entries in the base directory
    for entry in os.listdir(base_dir):
        match = version_pattern.match(entry)
        if match:
            v = int(match.group(1))
            if v > max_version:
                max_version = v

    if max_version < 0:
        raise FileNotFoundError(
            f"No 'version_{{n}}' subdirectories found in {base_dir!r}"
        )

    latest_dir = os.path.join(
        base_dir, f"version_{max_version}", "checkpoints", "last.ckpt"
    )

    return latest_dir


class TrainLossEarlyStopping(Callback):
    """
    Stops training when the training loss (logged per-step) hasn’t improved
    by at least `min_delta_pct` for `patience_steps` training steps,
    checking every `check_interval` steps.
    """

    def __init__(
        self,
        monitor: str = "train_loss_step",
        min_delta_pct: float = 0.01,
        patience_steps: int = 1000,
        check_interval: int = 100,
    ):
        """
        Args:
            monitor: name of the per-step training loss metric (must be logged
                     with `on_step=True`).
            min_delta_pct: relative improvement threshold, e.g. 0.01 == 1%.
            patience_steps: how many steps with *no sufficient* improvement
                            before stopping.
            check_interval: only evaluate every N steps for efficiency.
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta_pct = min_delta_pct
        self.patience_steps = patience_steps
        self.check_interval = check_interval

        self.best_loss = float("inf")
        self.bad_steps = 0
        self.global_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 1) grab the fresh per-step tensor from outputs
        current_tensor = outputs.get("loss")
        if current_tensor is None:
            return

        self.global_step += 1
        # 2) only check every `check_interval` steps
        if self.global_step % self.check_interval != 0:
            return

        current = current_tensor.item()
        # 3) did we get at least min_delta_pct relative improvement?
        rel_improve = (self.best_loss - current) / max(self.best_loss, 1e-8)

        if rel_improve >= self.min_delta_pct or self.best_loss == float("inf"):
            self.best_loss = current
            self.bad_steps = 0
        else:
            self.bad_steps += self.check_interval

        # 4) if we’ve gone too long with no “significant” improvement, stop
        if self.bad_steps >= self.patience_steps:
            pl_module.print(
                f"[TrainLossEarlyStopping] no ≥{self.min_delta_pct * 100:.2f}% "
                f"improvement in `{self.monitor}` over {self.patience_steps} steps. "
                "Stopping."
            )
            trainer.should_stop = True


def print_metrics(y_preds_np, y_np, thresholds=[0.3, 0.5], background_class=0):
    # Compute multiclass AUC
    auc_ovo = metrics.roc_auc_score(
        y_np,
        y_preds_np if y_preds_np.shape[-1] > 2 else y_preds_np[:, -1],
        multi_class="ovo",
    )
    print(f"AUC: {auc_ovo:.4f}\n")

    num_classes = y_preds_np.shape[1]

    for signal_class in range(num_classes):
        if signal_class == background_class:
            continue

        # Create binary labels: 1 for signal_class, 0 for background_class, ignore others
        mask = (y_np == signal_class) | (y_np == background_class)
        y_bin = (y_np[mask] == signal_class).astype(int)
        scores_bin = y_preds_np[mask, signal_class] / (
            y_preds_np[mask, signal_class] + y_preds_np[mask, background_class]
        )

        # Compute ROC
        fpr, tpr, _ = metrics.roc_curve(y_bin, scores_bin)

        print(f"Signal class {signal_class} vs Background class {background_class}:")

        for threshold in thresholds:
            bineff = np.argmax(tpr > threshold)
            print(
                "Class {} effS at {} 1.0/effB = {}".format(
                    signal_class, tpr[bineff], 1.0 / fpr[bineff]
                )
            )


class CLIPLoss(nn.Module):
    # From AstroCLIP: https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/models/astroclip.py#L117
    def get_logits(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        clean_features = F.normalize(clean_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        perturbed_features = F.normalize(perturbed_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features

        logits_per_clean = logit_scale * clean_features @ perturbed_features.T
        return logits_per_clean, logits_per_clean.T

    def forward(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        weight=None,
        logit_scale: float = 2.74,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the clean and perturbed features
        logits_per_clean, logits_per_perturbed = self.get_logits(
            clean_features, perturbed_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_clean.shape[0], device=clean_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_clean, labels, reduction="none")
            + F.cross_entropy(logits_per_perturbed, labels, reduction="none")
        ) / 2
        if weight is not None:
            total_loss = torch.mean(weight * total_loss)
        else:
            total_loss = total_loss.mean()
        return {"contrastive_loss": total_loss} if output_dict else total_loss


def sum_reduce(num, device):
    r"""Sum the tensor across the devices."""
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def get_param_groups(model, wd, lr, lr_factor=1.0, fine_tune=False, all_head=False):
    no_decay, decay = [], []
    last_layer_no_decay, last_layer_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if all_head:
            print("INCREASING LR ON WHOLE HEAD")
            is_last_layer = name.startswith("classifier")
        else:
            is_last_layer = name.startswith("classifier.out")

        if any(keyword in name for keyword in model.no_weight_decay()):
            if is_last_layer:
                last_layer_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if is_last_layer:
                last_layer_decay.append(param)
            else:
                decay.append(param)

    # Base learning rate groups
    param_groups = [
        {"params": decay, "weight_decay": wd, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]

    # Adjust learning rate for last layer if fine-tuning
    last_layer_lr = lr * lr_factor if fine_tune else lr
    print(f"Setting Last Layer LR to: {last_layer_lr}")

    if last_layer_decay:
        param_groups.append(
            {"params": last_layer_decay, "weight_decay": wd, "lr": last_layer_lr}
        )
    if last_layer_no_decay:
        param_groups.append(
            {"params": last_layer_no_decay, "weight_decay": 0.0, "lr": last_layer_lr}
        )

    return param_groups


def get_checkpoint_name(tag):
    return f"best_model_{tag}.pt"


def is_master_node():
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    else:
        return True


def ddp_setup():
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2900"
        os.environ["RANK"] = "0"
        init_process_group(rank=0, world_size=1)
        rank = local_rank = 0
    else:
        init_process_group(init_method="env://")
        # overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    return local_rank, rank, dist.get_world_size()


def get_bigram(add_timestamp: bool = False, timestamp_first: bool = False) -> str:
    """Generates a random bigram consisting of an adjective and a noun.
    I.e. "Happy-Dog", "Blue-Car", etc.
    """
    rw = RandomWord()
    adjective = rw.word(include_parts_of_speech=["adjective"])
    noun = rw.word(include_parts_of_speech=["noun"])
    # remove dashes or spaces just in case
    adjective = adjective.replace("-", "").replace(" ", "")
    noun = noun.replace("-", "").replace(" ", "")
    bigram = f"{adjective.capitalize()}-{noun.capitalize()}"
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if timestamp_first:
            bigram = f"{timestamp}-{bigram}"
        else:
            bigram = f"{bigram}-{timestamp}"
    return bigram


def get_version_number(out_dir_save_tag):
    # count how many directories starting with v{number}_<remainder> exist there
    version = 0
    # run_dir = os.path.join(args.outdir, save_tag, run_name)
    for name in (
        os.listdir(out_dir_save_tag) if os.path.exists(out_dir_save_tag) else []
    ):
        if os.path.isdir(os.path.join(out_dir_save_tag, name)) and name.startswith("v"):
            try:
                ver_num = int(name.split("_")[0][1:])
                if ver_num >= version:
                    version = ver_num + 1
            except ValueError:
                pass

    return version


def extract_pretrained_ckpt_prefix(path: str) -> str:
    """
    Extracts a string like '100k_class_only', '1M_gen_only', etc. from a
    checkpoint path.

    Examples:
        "/.../100k/class_only_val_loss=.5940.ckpt" -> "100k_class_only"
        "/.../1M/gen_only_val_loss=2.6142.ckpt"   -> "1M_gen_only"
    """
    # ---- extract prefix from filename ----
    filename = os.path.basename(path)
    prefix_match = re.search(
        r"(class_only|gen_only|class_gen|mpm_only|class_mpm|gen_mpm|pretrain)",
        filename,
    )
    if not prefix_match:
        raise ValueError(f"No known prefix found in filename: {filename}")
    prefix = prefix_match.group(1)

    # ---- extract scale (parent directory: 100k, 1M, 10M, 100M) ----
    parent_dir = os.path.basename(os.path.dirname(path.rstrip("/")))
    scale_match = re.fullmatch(r"(100k|1M|10M|100M)", parent_dir)
    if not scale_match:
        raise ValueError(f"No known scale folder found above file: {parent_dir}")
    scale = scale_match.group(1)

    return f"{scale}_{prefix}"
