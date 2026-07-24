"""Training loop for the CMUT multi-physics PINN."""

from __future__ import annotations

import numpy as np
import torch

from ..physics.cmut_plate import CMUTPlateResidual
from .losses import CompositeLoss
from .model import PINN, HardBCPlate


def make_cmut_pinn(device: str = "cpu") -> HardBCPlate:
    # inputs: (x, y, t_e, t_np, t_ox, t_w, omega_norm) -> (w, m)
    core = PINN(in_dim=7, out_dim=2, width=128, depth=5)
    return HardBCPlate(core).to(device)


def train_cmut_pinn(model, dataset, epochs: int = 2000, n_colloc: int = 1024,
                    lr: float = 1e-3, device: str = "cpu", log_every: int = 100,
                    length_scale: float = 33e-6, w_scale: float = 1e-8):
    """dataset: dict with tensors
        'params'  [N,4]  (t_e, t_np, t_ox, t_w) in um
        'omega'   [N,1]  normalized angular frequency
        'w_avg'   [N,1]  normalized average deflection target (FEM database)
    """
    physics = CMUTPlateResidual()
    loss_fn = CompositeLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []

    params = dataset["params"].to(device)
    omega = dataset["omega"].to(device)
    target = dataset["w_avg"].to(device)
    n_data = params.shape[0]

    for ep in range(epochs):
        opt.zero_grad()

        # ---- data loss: predicted average deflection vs FEM target ----
        idx = torch.randint(0, n_data, (min(256, n_data),))
        xy_q = torch.rand(64, 2, device=device)  # quadrature points
        d_res = []
        p_b, o_b, t_b = params[idx], omega[idx], target[idx]
        for i in range(p_b.shape[0]):
            inp = torch.cat([xy_q,
                             p_b[i].expand(64, 4),
                             o_b[i].expand(64, 1)], dim=1)
            w_avg = model(inp)[:, 0:1].mean()
            d_res.append(w_avg - t_b[i, 0])
        data_res = torch.stack(d_res)

        # ---- PDE residual on collocation points (physical units) ----
        xy = torch.rand(n_colloc, 2, device=device, requires_grad=True)
        j = torch.randint(0, n_data, (1,)).item()
        inp = torch.cat([xy, params[j].expand(n_colloc, 4),
                         omega[j].expand(n_colloc, 1)], dim=1)
        out = model(inp)
        w_phys = out[:, 0:1] * w_scale
        m_phys = out[:, 1:2] * w_scale  # moment head shares scale; D absorbs rest
        t_e = params[j, 0] * 1e-6
        t_np = params[j, 1] * 1e-6
        om = omega[j, 0] * 2 * np.pi * 10e6  # denormalize (10 MHz reference)
        # scale coords to physical length for derivatives: chain rule 1/L^2 per laplacian
        r1, r2 = physics.residual(w_phys, m_phys, xy, t_e, t_np, om)
        r1 = r1 / (length_scale ** -2 * w_scale)   # nondimensionalize residuals
        r2 = r2 / (length_scale ** -4 * w_scale)

        total, logs = loss_fn(data_res=data_res, pde_res=[r1, r2])
        total.backward()
        opt.step()
        sched.step()
        history.append(logs)
        if ep % log_every == 0:
            print(f"[{ep:5d}] total={logs['total']:.3e} data={logs.get('data', 0):.3e} "
                  f"pde={logs.get('pde', 0):.3e}")
    return history
