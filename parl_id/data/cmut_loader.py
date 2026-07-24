"""Loader for the CMUT FEM database (your COMSOL parametric-sweep export).

Expected CSV columns: t_e, t_np, t_ox, t_w, frequency_MHz, displacement_um
(the same 6-column layout as matrix D in the previous paper).

Also provides a physics-flavored synthetic generator so the full pipeline is
runnable before the real CSV is dropped in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

# Parametric-sweep bounds from the previous paper (um)
BOUNDS = {
    "t_e": (0.44, 0.84),
    "t_np": (1.3, 2.5),
    "t_ox": (1.3, 2.5),
    "t_w": (2.9, 3.5),
}


def load_csv(path: str, f_ref_mhz: float = 10.0, w_ref_um: float = 1e-2) -> dict:
    df = pd.read_csv(path)
    params = torch.tensor(df[["t_e", "t_np", "t_ox", "t_w"]].values,
                          dtype=torch.float32)
    omega = torch.tensor(df[["frequency_MHz"]].values / f_ref_mhz,
                         dtype=torch.float32)
    w_avg = torch.tensor(df[["displacement_um"]].values / w_ref_um,
                         dtype=torch.float32)
    return {"params": params, "omega": omega, "w_avg": w_avg}


def synthetic_database(n_combos: int = 200, n_freq: int = 25,
                       seed: int = 0) -> dict:
    """Plate-physics-flavored surrogate database: resonance ~ sqrt(D/rho h)/L^2,
    Lorentzian displacement response. Replace with load_csv() for real runs."""
    rng = np.random.default_rng(seed)
    rows_p, rows_o, rows_w = [], [], []
    for _ in range(n_combos):
        p = np.array([rng.uniform(*BOUNDS[k]) for k in
                      ("t_e", "t_np", "t_ox", "t_w")])
        t_e, t_np_, t_ox, t_w = p
        h = t_e + t_np_
        f0 = 2.0 + 3.0 * h / (t_w ** 0.5)          # MHz, monotone in stiffness
        q = 30.0 + 20.0 * t_ox
        freqs = np.linspace(0.5, 8.0, n_freq)       # MHz
        amp = 1.0 / np.sqrt((1 - (freqs / f0) ** 2) ** 2
                            + (freqs / (f0 * q)) ** 2)
        disp = 1e-3 * amp / amp.max() * (2.5 / h)   # um, softer -> larger
        for f, d in zip(freqs, disp):
            rows_p.append(p)
            rows_o.append([f / 10.0])
            rows_w.append([d / 1e-2])
    return {"params": torch.tensor(np.array(rows_p), dtype=torch.float32),
            "omega": torch.tensor(np.array(rows_o), dtype=torch.float32),
            "w_avg": torch.tensor(np.array(rows_w), dtype=torch.float32)}
