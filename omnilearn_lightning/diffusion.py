import torch
from omnilearned.diffusion import get_logsnr_alpha_sigma


def network_wrapper(model, z, condition, pid, add_info, y, time):
    base_model = model.module if hasattr(model, "module") else model
    x = base_model.body(z, condition, pid, add_info, time)
    x = base_model.generator(x, y)
    return x


def generate_class_free(
    model,
    y,
    shape,
    cond=None,
    pid=None,
    add_info=None,
    nsteps=64,
    device="cuda",
    guidance_scale: float = 1.0,  # <— new
) -> torch.Tensor:
    # 1) initial noise
    x = torch.randn(*shape, device=device)  # x_T ~ N(0,1)
    B = x.shape[0]

    # 2) build mask for zero-padded particles
    nparts = (100 * cond[:, -1]).int().view(B, 1).to(device)
    P = x.shape[1]
    mask = torch.arange(P, device=device).unsqueeze(0).repeat(B, 1) < nparts

    # 3) get x0 estimate at t=T
    x0 = get_logsnr_alpha_sigma(x, mask.float().unsqueeze(-1))

    def ode_wrapper(t, x_t_flat):
        x_t = x_t_flat.view_as(x0)
        time = t * torch.ones((B,), device=device)

        # unconditional pass (y_uncond=None)
        v_uncond = network_wrapper(
            model, x_t, cond, pid, add_info, y=torch.ones_like(y) * 2, time=time
        )
        # conditional pass
        v_cond = network_wrapper(model, x_t, cond, pid, add_info, y=y, time=time)
        # classifier-free guidance
        v = v_uncond + guidance_scale * (v_cond - v_uncond)

        # flatten derivative for odeint
        return v.view(B, -1)

    # flatten initial state for odeint
    y0 = x0.view(B, -1)
    t_tensor = torch.linspace(1, 0, nsteps, device=device, dtype=x0.dtype)

    # solve the ODE
    sol = odeint(
        func=ode_wrapper,
        y0=y0,
        t=t_tensor,
        method="midpoint",
        atol=1e-5,
        rtol=1e-5,
    )
    x_last = sol[-1].view_as(x0)
    return x_last
