"""Wrapper around google/ceviche-challenges (pip install ceviche-challenges).

Provides: (a) exact forward solves + exact adjoint gradients for the photonic
testbench, and (b) field datasets for PINN supervision. Falls back to the
bundled minimal FDFD (parl_id.physics.helmholtz) if the package is missing.
"""

from __future__ import annotations

import numpy as np

try:
    import ceviche_challenges as cc
    HAS_CEVICHE = True
except ImportError:
    HAS_CEVICHE = False

from ..physics.helmholtz import fdfd_solve


def make_mode_converter():
    """Returns the standard lightweight mode-converter challenge model."""
    if not HAS_CEVICHE:
        raise ImportError("pip install ceviche-challenges")
    spec = cc.mode_converter.prefabs.mode_converter_spec_12()
    params = cc.mode_converter.prefabs.mode_converter_sim_params()
    return cc.mode_converter.model.ModeConverterModel(params, spec)


def simulate(model, design: np.ndarray):
    """S-parameters and fields for a density design in [0,1]^(HxW)."""
    s_params, fields = model.simulate(design)
    return s_params, fields


def fallback_waveguide_dataset(nx: int = 120, ny: int = 80, dl: float = 40e-9,
                               wavelength: float = 1.55e-6, n_samples: int = 8,
                               seed: int = 0):
    """Small FDFD field dataset (random waveguide widths) for PINN supervision
    when ceviche-challenges is unavailable."""
    rng = np.random.default_rng(seed)
    eps_list, ez_list = [], []
    for _ in range(n_samples):
        eps = np.ones((ny, nx))
        wgw = rng.integers(8, 16)
        eps[ny // 2 - wgw // 2: ny // 2 + wgw // 2, :] = 12.25  # Si
        src = np.zeros((ny, nx))
        src[ny // 2 - wgw // 2: ny // 2 + wgw // 2, 15] = 1.0
        ez = fdfd_solve(eps, dl, wavelength, src)
        eps_list.append(eps)
        ez_list.append(ez)
    return np.array(eps_list), np.array(ez_list)
