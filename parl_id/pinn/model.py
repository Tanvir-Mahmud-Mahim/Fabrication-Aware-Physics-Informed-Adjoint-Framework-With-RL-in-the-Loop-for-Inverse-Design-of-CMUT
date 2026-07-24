"""PINN architectures: Fourier-feature MLP with optional hard boundary encoding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding (Tancik et al.) — critical for the
    4th-order/mixed plate PDE and oscillatory Helmholtz solutions."""

    def __init__(self, in_dim: int, n_features: int = 64, sigma: float = 3.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINN(nn.Module):
    """MLP PINN. Inputs: spatial coords + design parameters (+ frequency).
    Outputs: field components (e.g. [w, m] for CMUT, [E_re, E_im] for photonic).
    """

    def __init__(self, in_dim: int, out_dim: int, width: int = 128,
                 depth: int = 5, fourier: bool = True, sigma: float = 3.0):
        super().__init__()
        self.embed = FourierFeatures(in_dim, sigma=sigma) if fourier else nn.Identity()
        d0 = self.embed.out_dim if fourier else in_dim
        layers = [nn.Linear(d0, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(x))


class HardBCPlate(nn.Module):
    """Enforces clamped BCs w = dw/dn = 0 on the unit square by construction:
    w(x, y) = phi(x, y)^2 * N(x, y),  phi = x(1-x)y(1-y)."""

    def __init__(self, core: PINN):
        super().__init__()
        self.core = core

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        xy = inputs[:, :2]
        phi = (xy[:, 0:1] * (1 - xy[:, 0:1]) * xy[:, 1:2] * (1 - xy[:, 1:2]))
        out = self.core(inputs)
        w = phi ** 2 * out[:, 0:1]        # clamped deflection
        rest = out[:, 1:]                 # moment head unconstrained
        return torch.cat([w, rest], dim=-1)
