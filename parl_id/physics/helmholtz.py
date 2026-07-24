"""2D scalar Helmholtz physics for the photonic testbench.

PINN residual:   laplace(Ez) + k0^2 * eps(x, y) * Ez + i*omega*mu0*Jz = 0

Also provides a small sparse FDFD solver used as the exact reference
(ground-truth fields and exact adjoint gradients) when ceviche-challenges
is not installed.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

C0 = 299792458.0


class HelmholtzResidual:
    """Autodiff Helmholtz residual for a complex field represented as (Re, Im)."""

    def __init__(self, wavelength: float = 1.55e-6):
        self.k0 = 2.0 * np.pi / wavelength

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

    def residual(self, e_re: torch.Tensor, e_im: torch.Tensor,
                 xy: torch.Tensor, eps: torch.Tensor,
                 src_re: torch.Tensor | float = 0.0,
                 src_im: torch.Tensor | float = 0.0):
        k2e = self.k0 ** 2 * eps
        r_re = self._laplacian(e_re, xy) + k2e * e_re + src_re
        r_im = self._laplacian(e_im, xy) + k2e * e_im + src_im
        return r_re, r_im


def _s_profile(num: int, npml: int) -> np.ndarray:
    """Complex coordinate-stretching profile for absorbing boundary layers."""
    s = np.ones(num, dtype=complex)
    for i in range(npml):
        sigma = (3.0 * (npml - i) / npml) ** 3
        s[i] = 1 + 1j * sigma
        s[num - 1 - i] = 1 + 1j * sigma
    return s


def _system_matrix(eps_grid: np.ndarray, dl: float, wavelength: float,
                   npml: int) -> sp.csc_matrix:
    ny, nx = eps_grid.shape
    k0 = 2 * np.pi / wavelength
    sx, sy = _s_profile(nx, npml), _s_profile(ny, npml)

    def d1(num, s):
        e = np.ones(num)
        D = sp.diags([-e, e], [0, 1], shape=(num, num), format="csr") / dl
        return sp.diags(1.0 / s) @ D

    Dxf, Dyf = d1(nx, sx), d1(ny, sy)
    Lap = (sp.kron(sp.eye(ny), (-Dxf.T.conj()) @ Dxf)
           + sp.kron((-Dyf.T.conj()) @ Dyf, sp.eye(nx)))
    return (Lap + k0 ** 2 * sp.diags(eps_grid.ravel())).tocsc()


def fdfd_solve(eps_grid: np.ndarray, dl: float, wavelength: float,
               source: np.ndarray, npml: int = 10) -> np.ndarray:
    """Minimal 2D FDFD (Ez polarization) with absorbing (SC-PML-like) layers.

    Returns the complex Ez field on the grid.
    """
    ny, nx = eps_grid.shape
    A = _system_matrix(eps_grid, dl, wavelength, npml)
    b = 1j * source.ravel().astype(complex)
    ez = spla.spsolve(A, b)
    return ez.reshape(ny, nx)


def fdfd_adjoint_gradient(eps_grid: np.ndarray, dl: float, wavelength: float,
                          source: np.ndarray, dJ_dEz: np.ndarray,
                          ez: np.ndarray, npml: int = 10) -> np.ndarray:
    """Exact discrete adjoint gradient dJ/deps for a real objective J(Ez).

    dJ_dEz: complex array, the Wirtinger derivative dJ/dEz on the grid.
    Derivation: A x = b, dA/deps_i = k0^2 e_i e_i^T,
    lambda = -A^{-T} (dJ/dEz),  dJ/deps_i = 2 Re(k0^2 * lambda_i * x_i).
    """
    ny, nx = eps_grid.shape
    k0 = 2 * np.pi / wavelength
    A = _system_matrix(eps_grid, dl, wavelength, npml)
    lam = spla.spsolve(A.T, -dJ_dEz.ravel().astype(complex))
    # factor 2: real objective J(Ez), dJ = 2 Re[(dJ/dEz) dEz]
    grad = 2.0 * np.real(k0 ** 2 * lam * ez.ravel())
    return grad.reshape(ny, nx)
