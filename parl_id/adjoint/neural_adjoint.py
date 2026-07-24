"""Neural-adjoint inverse design: gradient-based optimization of design
parameters *through* a frozen, differentiable PINN surrogate.

Equivalence to the classical adjoint method:
    L = J(u, theta) + lambda^T R(u, theta)
    adjoint eq.: (dR/du)^T lambda = -(dJ/du)^T
    dJ/dtheta   = J_theta + lambda^T R_theta
When u = PINN(x; theta) satisfies R approx 0 by training, reverse-mode
autodiff of J w.r.t. theta implements the same total derivative implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class DesignSpec:
    """Box-constrained design vector with physical meaning per component."""
    names: list[str]
    lower: torch.Tensor
    upper: torch.Tensor

    def clamp(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(theta, self.upper), self.lower)

    def random(self) -> torch.Tensor:
        return self.lower + (self.upper - self.lower) * torch.rand_like(self.lower)


class NeuralAdjointOptimizer:
    """Adam/L-BFGS optimization of design parameters through a frozen surrogate.

    objective_fn(theta) -> scalar torch tensor, built from the PINN forward pass.
    """

    def __init__(self, objective_fn: Callable[[torch.Tensor], torch.Tensor],
                 spec: DesignSpec, method: str = "adam", lr: float = 1e-2):
        self.objective_fn = objective_fn
        self.spec = spec
        self.method = method
        self.lr = lr
        self.trajectory: list[dict] = []

    def run(self, theta0: torch.Tensor | None = None, iters: int = 200,
            restarts: int = 4) -> tuple[torch.Tensor, float]:
        best_theta, best_j = None, float("inf")
        for r in range(restarts):
            theta = (theta0.clone() if theta0 is not None and r == 0
                     else self.spec.random())
            theta = theta.detach().requires_grad_(True)
            if self.method == "lbfgs":
                theta, j = self._run_lbfgs(theta, iters)
            else:
                theta, j = self._run_adam(theta, iters)
            if j < best_j:
                best_theta, best_j = theta.detach().clone(), j
        return best_theta, best_j

    def _run_adam(self, theta, iters):
        opt = torch.optim.Adam([theta], lr=self.lr)
        j = float("inf")
        for i in range(iters):
            opt.zero_grad()
            J = self.objective_fn(theta)
            J.backward()
            opt.step()
            with torch.no_grad():
                theta.copy_(self.spec.clamp(theta))
            j = float(J)
            self.trajectory.append({"iter": i, "J": j,
                                    "theta": theta.detach().clone()})
        return theta, j

    def _run_lbfgs(self, theta, iters):
        opt = torch.optim.LBFGS([theta], lr=self.lr, max_iter=iters,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            J = self.objective_fn(self.spec.clamp(theta))
            J.backward()
            return J

        J = opt.step(closure)
        with torch.no_grad():
            theta.copy_(self.spec.clamp(theta))
        return theta, float(J)


def gradient_fidelity(neural_grad: torch.Tensor, exact_grad: torch.Tensor) -> float:
    """Cosine similarity between neural-adjoint and exact-adjoint gradients
    (validation figure F4 of the paper)."""
    a = neural_grad.flatten()
    b = exact_grad.flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


class SpecWarmStartBank:
    """Specification-to-specification warm starting of the inverse engine.

    Inspired by Soda-PTA's transfer of adjoint-tuned settings to unseen
    circuits (Sun et al., 2025) --- but transferring *solved designs* rather
    than optimizer hyperparameters: the optimum found at one target
    specification (e.g., operating frequency) seeds the gradient refinement at
    the nearest unsolved specification, replacing most of the random-probe
    budget. Store with add(spec, theta); query with suggest(spec).
    """

    def __init__(self):
        self._bank: list[tuple[torch.Tensor, torch.Tensor]] = []

    def add(self, spec, theta: torch.Tensor) -> None:
        s = torch.atleast_1d(torch.as_tensor(spec, dtype=torch.float32))
        self._bank.append((s, theta.detach().clone()))

    def suggest(self, spec) -> torch.Tensor | None:
        """Nearest-specification solved design, or None if the bank is empty."""
        if not self._bank:
            return None
        s = torch.atleast_1d(torch.as_tensor(spec, dtype=torch.float32))
        dists = [float((bs - s).norm()) for bs, _ in self._bank]
        return self._bank[int(np.argmin(dists))][1].clone()
