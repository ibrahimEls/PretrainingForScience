import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return x.unsqueeze(-1) * y.to(x.dtype)


@torch.compile
def logsnr_schedule_cosine(t, logsnr_min=-20.0, logsnr_max=20.0, shift=1.0):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2.0 * torch.log(torch.tan(a * t + b) * shift)


# @torch.compile
def get_logsnr_alpha_sigma(time, shift=1.0):
    logsnr = logsnr_schedule_cosine(time, shift=shift)[:, None, None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma

def get_ad_eps(x, mask):
    means = torch.tensor(
        [0.00027097, 0.0000285, 0.90671477, 1.29942334], device=x.device
    )
    stds = torch.tensor(
        [0.30882706, 0.30609399, 1.23984054, 1.28986600], device=x.device
    )

    # Shape: (1, 1, 4) so it can broadcast across (B, N, 4)
    means = means.view(1, 1, 4)
    stds = stds.view(1, 1, 4)

    eps = (torch.randn_like(x) * stds + means) * mask
    return eps

def network_wrapper(model, z, condition, pid, add_info, y, time):
    base_model = model.module if hasattr(model, "module") else model
    x = base_model.body(z, condition, pid, add_info, time)
    x = base_model.generator(x, y)
    return x

def perturb(x, time):
    mask = x[:, :, 3:4] != 0
    eps = torch.randn_like(x)  # eps ~ N(0, 1)
    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time)
    z = alpha * x + eps * sigma
    v = alpha * eps - sigma * x
    return z * mask, v * mask

