"""Fabrication-feasibility projections applied during adjoint optimization."""

from __future__ import annotations

import torch


def project_min_feature(theta: torch.Tensor, min_step: float) -> torch.Tensor:
    """Snap design parameters to the nearest fabricable increment (e.g. the
    deposition tool's thickness resolution)."""
    return torch.round(theta / min_step) * min_step


def erosion_dilation(pattern: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary morphological erosion/dilation of a 2D density pattern —
    the standard photonic-foundry robustness triple (eroded/nominal/dilated)."""
    k = 2 * radius + 1
    pad = radius
    pooled_max = torch.nn.functional.max_pool2d(
        pattern[None, None], k, stride=1, padding=pad)[0, 0]
    pooled_min = -torch.nn.functional.max_pool2d(
        -pattern[None, None], k, stride=1, padding=pad)[0, 0]
    return pooled_min, pooled_max  # eroded, dilated
