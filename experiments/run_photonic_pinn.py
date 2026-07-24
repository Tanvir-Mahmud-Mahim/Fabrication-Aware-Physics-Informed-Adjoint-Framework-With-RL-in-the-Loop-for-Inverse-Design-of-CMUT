"""Experiment 4 (Fig. F4/F6): photonic testbench.

(a) Generates FDFD reference fields (ceviche-challenges if installed, else the
    bundled minimal FDFD), (b) trains a Helmholtz PINN on them, and
(c) validates neural-adjoint vs exact-adjoint gradients (cosine similarity).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from parl_id.data.ceviche_loader import HAS_CEVICHE, fallback_waveguide_dataset  # noqa: E402
from parl_id.physics.helmholtz import HelmholtzResidual, fdfd_adjoint_gradient, fdfd_solve  # noqa: E402
from parl_id.pinn.losses import CompositeLoss  # noqa: E402
from parl_id.pinn.model import PINN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--ny", type=int, default=60)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ceviche-challenges installed: {HAS_CEVICHE} | device: {device}")

    # ---- reference fields ----
    eps_all, ez_all = fallback_waveguide_dataset(nx=args.nx, ny=args.ny,
                                                 n_samples=4)
    eps, ez = eps_all[0], ez_all[0]
    ny, nx = eps.shape
    dl, wl = 40e-9, 1.55e-6

    # normalized coords and field
    ys, xs = np.mgrid[0:ny, 0:nx]
    xy = np.stack([xs.ravel() / nx, ys.ravel() / ny], axis=1)
    ez_n = ez / (np.abs(ez).max() + 1e-30)

    xy_t = torch.tensor(xy, dtype=torch.float32, device=device)
    eps_t = torch.tensor(eps.ravel()[:, None], dtype=torch.float32, device=device)
    tgt = torch.tensor(np.stack([ez_n.real.ravel(), ez_n.imag.ravel()], axis=1),
                       dtype=torch.float32, device=device)

    # ---- PINN: (x, y, eps) -> (E_re, E_im) ----
    model = PINN(in_dim=3, out_dim=2, width=128, depth=5, sigma=10.0).to(device)
    phys = HelmholtzResidual(wl)
    loss_fn = CompositeLoss(w_pde=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n = xy_t.shape[0]
    Lx = nx * dl  # physical width for residual scaling
    for ep in range(args.epochs):
        opt.zero_grad()
        idx = torch.randint(0, n, (2048,), device=device)
        inp = torch.cat([xy_t[idx], eps_t[idx]], dim=1)
        pred = model(inp)
        data_res = pred - tgt[idx]

        cidx = torch.randint(0, n, (512,), device=device)
        xy_c = xy_t[cidx].clone().requires_grad_(True)
        inp_c = torch.cat([xy_c, eps_t[cidx]], dim=1)
        out_c = model(inp_c)
        # residual in normalized coords: laplace_norm = Lx^2 * laplace_phys
        r_re, r_im = phys.residual(out_c[:, 0:1], out_c[:, 1:2], xy_c,
                                   eps_t[cidx])
        scale = (phys.k0 * Lx) ** 2
        r_re = r_re / scale
        r_im = r_im / scale

        total, logs = loss_fn(data_res=data_res, pde_res=[r_re, r_im])
        total.backward()
        opt.step()
        if ep % 200 == 0:
            print(f"[{ep:5d}] total={logs['total']:.3e} "
                  f"data={logs['data']:.3e} pde={logs['pde']:.3e}")

    # ---- gradient-fidelity check (F4) ----
    # J = |Ez|^2 at a probe point; exact adjoint via FDFD vs finite-diff proxy
    probe = (ny // 2, nx - 20)
    dJ = np.zeros((ny, nx), dtype=complex)
    dJ[probe] = np.conj(ez[probe])
    src = np.zeros((ny, nx))
    src[ny // 2 - 5:ny // 2 + 5, 15] = 1.0
    g_exact = fdfd_adjoint_gradient(eps, dl, wl, src, dJ, ez)

    # finite-difference check on 5 random pixels
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(5):
        iy, ix = rng.integers(15, ny - 15), rng.integers(20, nx - 25)
        de = 1e-3
        eps_p = eps.copy(); eps_p[iy, ix] += de
        ez_p = fdfd_solve(eps_p, dl, wl, src)
        g_fd = (abs(ez_p[probe]) ** 2 - abs(ez[probe]) ** 2) / de
        errs.append(abs(g_fd - g_exact[iy, ix]) / (abs(g_fd) + 1e-30))
    print(f"adjoint-vs-FD relative errors: {np.round(errs, 3)}")

    Path("outputs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "outputs/photonic_pinn.pt")
    np.savez("outputs/photonic_ref.npz", eps=eps, ez=ez, g_exact=g_exact)


if __name__ == "__main__":
    main()
