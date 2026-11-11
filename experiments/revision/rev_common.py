"""Shared utilities for the revision experiments (NumPy/SciPy + autograd).

Ground-truth data: only the 123 FEM-simulated base designs of the released
database are used.  The released file also contains two augmented copies of
each design (inputs perturbed by up to 12 %, outputs unchanged, the
augmentation of the precursor study); these copies are excluded because their
labels are not independent FEM results.
"""
import json, time
from pathlib import Path
import numpy as onp
import autograd.numpy as np
from autograd import grad, value_and_grad
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'; RES.mkdir(exist_ok=True)
CSV = HERE.parent / 'augmented_reshaped_dataset.csv'
PCOLS = ['t_e', 't_np', 't_ox', 't_w']

# nominal bulk material values (enter the physics prior only through ratios;
# the overall load scale and effective span are calibrated)
E_AU, NU_AU, RHO_AU = 70e9, 0.44, 19300.0
E_SN, NU_SN, RHO_SN = 250e9, 0.23, 3100.0


def load_base():
    """Return dataframe of the 123 FEM base designs and design table."""
    df = pd.read_csv(CSV)
    X = df[PCOLS].values
    rnd = onp.all(onp.abs(X * 100 - onp.round(X * 100)) < 1e-6, axis=1)
    b = df[rnd].copy()
    b['design'] = b.groupby(PCOLS).ngroup()
    designs = b.groupby('design')[PCOLS].first().values
    return b, designs, df


def group_folds(n_designs, k=5, seed=0):
    r = onp.random.default_rng(seed)
    idx = r.permutation(n_designs)
    return [onp.sort(idx[i::k]) for i in range(k)]


# ---------------- physics prior: Kirchhoff-Love bilayer plate ----------------
def rigidity(te, tnp, mode='laminate'):
    """Flexural rigidity [N m] of the Au/Si3N4 bilayer (thicknesses in um)."""
    t1, t2 = te * 1e-6, tnp * 1e-6
    e1, e2 = E_AU / (1 - NU_AU ** 2), E_SN / (1 - NU_SN ** 2)
    if mode == 'mixture':  # rule of mixtures (original manuscript)
        h = t1 + t2
        eeff = (E_AU * t1 + E_SN * t2) / h
        nu = (NU_AU * t1 + NU_SN * t2) / h
        return eeff * h ** 3 / (12 * (1 - nu ** 2))
    z1, z2 = t1, t1 + t2
    zn = (e1 * t1 * (t1 / 2) + e2 * t2 * (t1 + t2 / 2)) / (e1 * t1 + e2 * t2)
    return (e1 * ((z1 - zn) ** 3 - (0 - zn) ** 3) + e2 * ((z2 - zn) ** 3 - (z1 - zn) ** 3)) / 3


def areal_mass(te, tnp):
    return RHO_AU * te * 1e-6 + RHO_SN * tnp * 1e-6


def modal_integrals(bc='clamped', n=401):
    """I0=int psi^2, I1=int psi, I4=int (lap psi)^2 on the unit square for the
    single admissible trial function psi (clamped: phi^2, simply supported-like: phi)."""
    s = onp.linspace(0, 1, n); x, y = onp.meshgrid(s, s)
    phi = x * (1 - x) * y * (1 - y)
    psi = phi ** 2 if bc == 'clamped' else phi
    h = s[1] - s[0]
    lap = onp.gradient(onp.gradient(psi, h, axis=1), h, axis=1) + onp.gradient(onp.gradient(psi, h, axis=0), h, axis=0)
    w = onp.ones(n); w[0] = w[-1] = 0.5
    W2 = onp.outer(w, w) * h * h
    return float((psi ** 2 * W2).sum()), float((psi * W2).sum()), float((lap ** 2 * W2)[2:-2, 2:-2].sum())


# ---------------- networks ----------------
def init_mlp(sizes, r):
    return [(r.normal(0, onp.sqrt(1.0 / m), (m, n)), onp.zeros(n)) for m, n in zip(sizes[:-1], sizes[1:])]


def mlp(params, x):
    for W, b in params[:-1]:
        x = np.tanh(np.dot(x, W) + b)
    W, b = params[-1]
    return np.dot(x, W) + b


def fourier(x, B):
    if B is None:
        return x
    z = 2 * onp.pi * np.dot(x, B)
    return np.concatenate([x, np.sin(z), np.cos(z)], axis=1)


def adam(params_flat_grad, theta, steps, lr, callback=None, b1=0.9, b2=0.999, eps=1e-8):
    """Minimal Adam over a list-of-arrays parameter container (flattened)."""
    from autograd.misc.flatten import flatten
    flat, unflat = flatten(theta)
    g_fun = lambda f: flatten(params_flat_grad(unflat(f)))[0]
    m = onp.zeros_like(flat); v = onp.zeros_like(flat)
    for t in range(1, steps + 1):
        g = g_fun(flat)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        flat = flat - lr * (m / (1 - b1 ** t)) / (onp.sqrt(v / (1 - b2 ** t)) + eps)
        if callback is not None and t % 100 == 0:
            callback(t, unflat(flat))
    return unflat(flat)


def r2(y, p):
    return float(1 - onp.sum((y - p) ** 2) / onp.sum((y - y.mean()) ** 2))


def save_json(name, obj):
    with open(RES / name, 'w') as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    with open(RES / name) as f:
        return json.load(f)
