"""Kirchhoff-Love plate electro-mechanics for the CMUT membrane (frequency domain).

Governing PDE (harmonic steady state, deflection amplitude W):

    D * biharmonic(W) - rho*h*omega^2 * W = p_es(W) + p_ext

with electrostatic parallel-plate coupling

    p_es = eps0 * V^2 / (2 * (g - W)^2)

and clamped boundary conditions W = dW/dn = 0.

A mixed (second-order) formulation is used for PINN stability:
    M = -D * laplace(W)                (moment sum)
    -laplace(M) - rho*h*omega^2 W = p  (equilibrium)
which avoids 4th-order autodiff on the biharmonic operator.
"""

from __future__ import annotations

import torch

EPS0 = 8.8541878128e-12  # F/m


class CMUTPlateResidual:
    """PDE residual of the mixed-form Kirchhoff-Love plate with electrostatic load.

    All quantities SI. Thickness parameters (t_e, t_np, t_ox, t_w) enter through
    the flexural rigidity D and areal mass rho*h of the composite membrane stack.
    """

    def __init__(self, E_nitride=250e9, nu=0.23, rho_nitride=3100.0,
                 E_gold=79e9, rho_gold=19300.0, gap=100e-9, voltage=5.0):
        self.E_n, self.nu, self.rho_n = E_nitride, nu, rho_nitride
        self.E_au, self.rho_au = E_gold, rho_gold
        self.gap, self.V = gap, voltage

    def stack_properties(self, t_e: torch.Tensor, t_np: torch.Tensor):
        """Composite flexural rigidity D and areal mass rho*h of electrode+nitride."""
        # Rule-of-mixtures composite (electrode Au layer + nitride passivation)
        h = t_e + t_np
        E_eff = (self.E_au * t_e + self.E_n * t_np) / h
        rho_h = self.rho_au * t_e + self.rho_n * t_np
        D = E_eff * h ** 3 / (12.0 * (1.0 - self.nu ** 2))
        return D, rho_h

    def electrostatic_pressure(self, w: torch.Tensor) -> torch.Tensor:
        g_eff = torch.clamp(self.gap - w, min=1e-9)
        return EPS0 * self.V ** 2 / (2.0 * g_eff ** 2)

    @staticmethod
    def _laplacian(f: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(f, xy, torch.ones_like(f), create_graph=True)[0]
        lap = 0.0
        for d in range(2):
            g2 = torch.autograd.grad(grad[:, d:d + 1], xy,
                                     torch.ones_like(grad[:, d:d + 1]),
                                     create_graph=True)[0][:, d:d + 1]
            lap = lap + g2
        return lap

    def residual(self, w: torch.Tensor, m: torch.Tensor, xy: torch.Tensor,
                 t_e: torch.Tensor, t_np: torch.Tensor, omega: torch.Tensor,
                 p_ext: float = 0.0):
        """Mixed-form residuals (r1: moment definition, r2: equilibrium).

        w, m : network outputs [N,1] (deflection, moment sum)
        xy   : collocation points [N,2] with requires_grad=True
        """
        D, rho_h = self.stack_properties(t_e, t_np)
        r1 = m + D * self._laplacian(w, xy)
        p = self.electrostatic_pressure(w) + p_ext
        r2 = -self._laplacian(m, xy) - rho_h * omega ** 2 * w - p
        return r1, r2
