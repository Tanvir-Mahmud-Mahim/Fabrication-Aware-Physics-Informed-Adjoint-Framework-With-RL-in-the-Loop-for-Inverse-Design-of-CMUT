"""Composite PINN loss: data + PDE residual + BC, with self-adaptive weights.

Includes an optional rank-aware Pearson-correlation term (inspired by PearSAN,
Bezick et al., Adv. Optical Materials 2026): inverse design needs the surrogate
to *rank* candidate designs correctly more than it needs absolute accuracy, so
maximizing the correlation between predicted and true objectives is a directly
optimization-relevant training signal. Unlike PearSAN's physics-blind
surrogate, here the correlation term coexists with the PDE residuals.
"""

from __future__ import annotations

import torch


def pearson_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_corr = 1 - rho(pred, target); differentiable, scale-invariant."""
    p = pred.flatten() - pred.mean()
    t = target.flatten() - target.mean()
    rho = (p * t).sum() / (p.norm() * t.norm() + 1e-12)
    return 1.0 - rho


class CompositeLoss:
    def __init__(self, w_data: float = 1.0, w_pde: float = 0.1, w_bc: float = 1.0,
                 w_corr: float = 0.0, adaptive: bool = True):
        self.w = {"data": w_data, "pde": w_pde, "bc": w_bc, "corr": w_corr}
        self.adaptive = adaptive
        self.ema = {k: 1.0 for k in self.w}

    def __call__(self, data_res: torch.Tensor | None = None,
                 pde_res: list[torch.Tensor] | None = None,
                 bc_res: torch.Tensor | None = None,
                 pred_target: tuple[torch.Tensor, torch.Tensor] | None = None
                 ) -> tuple[torch.Tensor, dict]:
        terms, logs = [], {}
        if data_res is not None:
            l = (data_res ** 2).mean()
            terms.append(self.w["data"] * self._scale("data", l))
            logs["data"] = float(l)
        if pred_target is not None and self.w["corr"] > 0:
            l = pearson_correlation_loss(*pred_target)
            terms.append(self.w["corr"] * self._scale("corr", l))
            logs["corr"] = float(l)
        if pde_res:
            l = sum((r ** 2).mean() for r in pde_res) / len(pde_res)
            terms.append(self.w["pde"] * self._scale("pde", l))
            logs["pde"] = float(l)
        if bc_res is not None:
            l = (bc_res ** 2).mean()
            terms.append(self.w["bc"] * self._scale("bc", l))
            logs["bc"] = float(l)
        total = sum(terms)
        logs["total"] = float(total)
        return total, logs

    def _scale(self, key: str, loss: torch.Tensor) -> torch.Tensor:
        """Gradient-magnitude style balancing via loss-EMA normalization."""
        if not self.adaptive:
            return loss
        with torch.no_grad():
            self.ema[key] = 0.99 * self.ema[key] + 0.01 * float(loss)
        return loss / max(self.ema[key], 1e-12)
