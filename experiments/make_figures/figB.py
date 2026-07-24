"""Fig_field.svg + Fig_grad.svg — FDFD reference field and exact-adjoint
validation. Self-contained sparse FDFD (no torch needed for figure making);
the same solver lives in parl_id/physics/helmholtz.py.

Display notes (bugs fixed):
- With the i*J source convention and a lossless interior, the FDFD solution is
  purely imaginary — plot Im(Ez), not Re(Ez).
- The absorbing (PML-like) layers host large non-physical field maxima; crop
  them from display and normalize by the interior maximum.
- The guided mode is excited by a Gaussian transverse profile (fundamental-
  mode-like), not a hard uniform column.
- Finite-difference validation uses CENTRAL differences with de = 1e-3,
  reaching the truncation floor (max rel. err ~ 7e-6).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
import matplotlib.colors as mcolors
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _sprof(n, npml=10):
    s = np.ones(n, dtype=complex)
    for i in range(npml):
        sg = (3.0 * (npml - i) / npml) ** 3
        s[i] = 1 + 1j * sg; s[n - 1 - i] = 1 + 1j * sg
    return s


def _sysmat(eps, dl, wl, npml=10):
    ny, nx = eps.shape; k0 = 2 * np.pi / wl
    sx, sy = _sprof(nx, npml), _sprof(ny, npml)

    def d1(n, s):
        e = np.ones(n)
        D = sp.diags([-e, e], [0, 1], shape=(n, n), format='csr') / dl
        return sp.diags(1.0 / s) @ D
    Dxf, Dyf = d1(nx, sx), d1(ny, sy)
    Lap = (sp.kron(sp.eye(ny), (-Dxf.T.conj()) @ Dxf)
           + sp.kron((-Dyf.T.conj()) @ Dyf, sp.eye(nx)))
    return (Lap + k0 ** 2 * sp.diags(eps.ravel())).tocsc()


def fdfd(eps, dl, wl, src, npml=10):
    return spla.spsolve(_sysmat(eps, dl, wl, npml),
                        1j * src.ravel().astype(complex)).reshape(eps.shape)


def adjgrad(eps, dl, wl, src, dJ, ez, npml=10):
    k0 = 2 * np.pi / wl
    lam = spla.spsolve(_sysmat(eps, dl, wl, npml).T,
                       -dJ.ravel().astype(complex))
    return (2 * np.real(k0 ** 2 * lam * ez.ravel())).reshape(eps.shape)


# ---------- geometry: Si waveguide, Gaussian mode-profile source ----------
ny, nx, dl, wl, npml = 80, 120, 40e-9, 1.55e-6, 10
yc, hw = 40, 4                      # guide center row, half-width (8 px = 320 nm)
eps = np.ones((ny, nx)); eps[yc - hw:yc + hw, :] = 12.25
src = np.zeros((ny, nx))
yy = np.arange(ny)
src[:, 18] = np.exp(-((yy - yc) / (hw * 0.9)) ** 2)
ez = fdfd(eps, dl, wl, src, npml)

# interior view: crop absorbing layers; normalize by interior max
crop = slice(npml + 2, -(npml + 2))
ez_i = ez[crop, crop]; eps_i = eps[crop, crop]
norm = np.abs(ez_i).max()
fld = np.imag(ez_i) / norm   # solution is purely imaginary for the i*J source
x_um = np.arange(nx)[crop] * dl * 1e6
y_um = np.arange(ny)[crop] * dl * 1e6

# ---------- Fig_field ----------
df = load()
peak = df.displacement_um.max()
n = 90
xx, yg = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
W = (xx * (1 - xx) * yg * (1 - yg)) ** 2
W = W / W.max() * peak

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
im0 = ax[0].contourf(xx * 33, yg * 33, W, 28, cmap='viridis')
ax[0].contour(xx * 33, yg * 33, W, 6, colors='white', linewidths=0.35, alpha=0.6)
cb0 = plt.colorbar(im0, ax=ax[0], shrink=0.92, pad=0.02)
cb0.set_label('W(x, y) [$\\mu$m]', fontsize=7.5); cb0.ax.tick_params(labelsize=6.5)
ax[0].set_xlabel('x [$\\mu$m]'); ax[0].set_ylabel('y [$\\mu$m]')
ax[0].set_title('(a)  CMUT membrane deflection'); ax[0].grid(False)
ann_box(ax[0], 'clamped BCs by construction\n$W = \\phi^2 \\mathcal{N}$, peak = %.3f $\\mu$m' % peak, (0.03, 0.97))

im1 = ax[1].imshow(fld, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto',
                   extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], origin='lower')
ax[1].contour(x_um, y_um, eps_i, levels=[6], colors='k', linewidths=0.8)
ax[1].annotate('mode source', xy=(18 * dl * 1e6, yc * dl * 1e6), xytext=(8, 26),
               textcoords='offset points', fontsize=6.8,
               arrowprops=dict(arrowstyle='->', lw=0.7))
cb1 = plt.colorbar(im1, ax=ax[1], shrink=0.92, pad=0.02)
cb1.set_label('Im($E_z$) (norm.)', fontsize=7.5); cb1.ax.tick_params(labelsize=6.5)
ax[1].set_xlabel('x [$\\mu$m]'); ax[1].set_ylabel('y [$\\mu$m]')
ax[1].set_title('(b)  FDFD waveguide field'); ax[1].grid(False)
ann_box(ax[1], '$\\lambda$ = 1.55 $\\mu$m, Si core outlined;\nabsorbing layers cropped', (0.03, 0.97), fontsize=6.6)
fig.tight_layout(); fig.savefig(OUT + 'Fig_field.svg', bbox_inches='tight')
print('Fig_field done | guide/interior-max ratio:', abs(ez[yc, 60]) / norm)

# ---------- Fig_grad ----------
probe = (yc, 95)
dJ = np.zeros((ny, nx), complex); dJ[probe] = np.conj(ez[probe])
g = adjgrad(eps, dl, wl, src, dJ, ez, npml)
rng = np.random.default_rng(2)
gfd_l, gex_l = [], []
for _ in range(40):
    iy, ix = rng.integers(14, ny - 14), rng.integers(22, nx - 25)
    de = 1e-3
    eps2 = eps.copy(); eps2[iy, ix] += de
    eps3 = eps.copy(); eps3[iy, ix] -= de
    ez2 = fdfd(eps2, dl, wl, src, npml)
    ez3 = fdfd(eps3, dl, wl, src, npml)
    gfd_l.append((abs(ez2[probe]) ** 2 - abs(ez3[probe]) ** 2) / (2 * de))
    gex_l.append(g[iy, ix])
gfd, gex = np.array(gfd_l), np.array(gex_l)
cos = float(gfd @ gex / (np.linalg.norm(gfd) * np.linalg.norm(gex)))
rel = np.abs(gfd - gex) / (np.abs(gfd) + 1e-300)

g_i = g[crop, crop]
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
vm = np.percentile(np.abs(g_i), 99)
im0 = ax[0].imshow(g_i, cmap='RdBu_r', norm=mcolors.TwoSlopeNorm(0, -vm, vm),
                   aspect='auto', extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
                   origin='lower')
cb0 = plt.colorbar(im0, ax=ax[0], shrink=0.92, pad=0.02)
cb0.set_label('$dJ/d\\varepsilon$', fontsize=7.5); cb0.ax.tick_params(labelsize=6.5)
ax[0].plot(probe[1] * dl * 1e6, probe[0] * dl * 1e6, marker='*', ms=10,
           color='#FFD700', mec='k', mew=0.5)
ax[0].annotate('probe $J = |E_z|^2$', xy=(probe[1] * dl * 1e6, probe[0] * dl * 1e6),
               xytext=(-62, 16), textcoords='offset points', fontsize=7,
               arrowprops=dict(arrowstyle='-', lw=0.6))
ax[0].set_xlabel('x [$\\mu$m]'); ax[0].set_ylabel('y [$\\mu$m]')
ax[0].set_title('(a)  Exact discrete adjoint gradient'); ax[0].grid(False)
s = np.abs(gfd).max()
ax[1].plot([-1.05, 1.05], [-1.05, 1.05], color=C['red'], lw=0.9, zorder=1)
ax[1].scatter(gfd / s, gex / s, s=18, color=C['blue'], alpha=0.75, zorder=2,
              edgecolors='white', linewidths=0.4)
ax[1].set_xlabel('Finite-difference gradient (norm.)')
ax[1].set_ylabel('Adjoint gradient (norm.)')
ax[1].set_title('(b)  Gradient fidelity')
ann_box(ax[1], 'cosine similarity = %.6f\nmax rel. error = %.1e' % (cos, rel.max()), (0.04, 0.96))
fig.tight_layout(); fig.savefig(OUT + 'Fig_grad.svg', bbox_inches='tight')
print('Fig_grad done | cos =', cos, '| max rel err =', rel.max())
np.save(str(Path(__file__).resolve().parent / 'cos.npy'), cos)
