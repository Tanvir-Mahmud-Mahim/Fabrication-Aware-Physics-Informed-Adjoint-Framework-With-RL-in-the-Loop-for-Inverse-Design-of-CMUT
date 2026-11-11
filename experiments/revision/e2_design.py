"""E2 (Reviewer 1, point 2): inverse design and fabrication robustness on the
physics-informed surrogate (PINN, lam = 0.1, 5-member ensemble trained on all
123 FEM designs).

Specification: target band-averaged displacement W_tgt (two-sided sensitivity
specification); deviation d = |W/W_tgt - 1|.  Process corruption (kappa = 1):
t_e, t_np x N(1, 0.03^2); t_ox x N(1, 0.05^2); t_w + N(0, 0.08^2) um.
Nominal designs are confined to the FEM box shrunk by two process standard
deviations, so corrupted designs stay inside the sampled range (no clipping).
Stages: (a) surrogate ensemble, (b) inverse-engine comparison at equal query
budget, (c) robust correction (mean-variance vs CVaR, Rockafellar-Uryasev
sample-average form) with common random numbers, (d) estimator-noise study."""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import *
import e1_surrogate as E1
from autograd.misc.flatten import flatten
from scipy.optimize import minimize

CFG = E1.CONFIGS['pinn_lam0.1']
lo, hi = E1.lo, E1.hi
SIG = onp.array([0.03, 0.03, 0.05, 0.08])        # relative, relative, relative, absolute [um]
REL = onp.array([1, 1, 1, 0])
ILO = onp.where(REL == 1, lo * (1 + 2 * SIG), lo + 2 * SIG)
IHI = onp.where(REL == 1, hi * (1 - 2 * SIG), hi - 2 * SIG)
ENS_FILE = RES / 'e2_ensemble.pkl'


def train_ensemble(seeds):
    th = E1.designs; ly = onp.log(E1.Wbar)
    try:
        ens = pickle.load(open(ENS_FILE, 'rb'))
    except FileNotFoundError:
        ens = {}
    for s in seeds:
        if s not in ens:
            ens[s] = E1.fit(CFG, th, ly, seed=100 + s)
            pickle.dump(ens, open(ENS_FILE, 'wb'))
    return ens


def load_ens():
    if not ENS_FILE.exists():
        train_ensemble(range(5))
    ens = pickle.load(open(ENS_FILE, 'rb'))
    return [ens[k] for k in sorted(ens)]


ENS = None
def W(th):
    """Ensemble-mean band-averaged displacement [um]; th: (n,4)."""
    global ENS
    if ENS is None:
        ENS = load_ens()
    acc = 0.0
    for p, B in ENS:
        lp, _ = E1.predict(p, B, th, CFG)
        acc = acc + np.exp(lp)
    return acc / len(ENS)


def corrupt(th, xi, kappa=1.0):
    s = kappa * SIG
    return np.stack([th[0] * (1 + s[0] * xi[:, 0]), th[1] * (1 + s[1] * xi[:, 1]),
                     th[2] * (1 + s[2] * xi[:, 2]), th[3] + s[3] * xi[:, 3]], axis=1)


def dev(th, xi, wt, kappa=1.0):
    return np.abs(W(corrupt(th, xi, kappa)) / wt - 1)


def cvar_upper(dv, alpha):
    k = max(1, int(onp.ceil(alpha * len(dv))))
    return float(onp.mean(onp.sort(dv)[-k:]))


def stats(th, wt, xi, kappa=1.0):
    dv = onp.asarray(dev(onp.asarray(th), xi, wt, kappa))
    nom = float(W(onp.asarray(th)[None])[0] / wt - 1)
    return dict(nominal_dev=nom, mean_dev=float(dv.mean()), std_dev=float(dv.std()),
                p95_dev=float(onp.percentile(dv, 95)), cvar5_dev=cvar_upper(dv, 0.05),
                yield_2pct=float(onp.mean(dv <= 0.02)), yield_5pct=float(onp.mean(dv <= 0.05)))


# ---------------- Stage 2: inverse engine ----------------
def J_spec(th, wt):
    return (W(th[None])[0] / wt - 1) ** 2

gJ = grad(J_spec)

def to_box(u):
    return ILO + (IHI - ILO) * onp.clip(u, 0, 1)


def gradient_refine(u0, wt, iters, lr=0.05, tol=5e-3):
    u = u0.copy(); q = 0; hist = []
    for t in range(iters):
        th = to_box(u); val = float(J_spec(th, wt)); q += 1; hist.append(onp.sqrt(val))
        if onp.sqrt(val) < tol:
            return th, q, True, hist
        g = onp.asarray(gJ(th, wt)) * (IHI - ILO)
        u = onp.clip(u - lr * g / (onp.linalg.norm(g) + 1e-12) * min(1.0, 20 * onp.sqrt(val)), 0, 1)
    th = to_box(u); return th, q, onp.sqrt(float(J_spec(th, wt))) < tol, hist


def engine(method, wt, budget, rng, tol=5e-3):
    """Returns (theta, queries_used, success, best-so-far curve of |dev|)."""
    curve = []; best = (onp.inf, None)
    def rec(th, v):
        nonlocal best
        if v < best[0]:
            best = (v, th)
        curve.append(best[0])
    if method == 'random':
        for k in range(budget):
            th = to_box(rng.random(4)); v = float(abs(W(th[None])[0] / wt - 1)); rec(th, v)
            if v < tol:
                return th, k + 1, True, curve
        return best[1], budget, False, curve
    if method == 'gradient':   # unseeded: restarts from random points
        used = 0
        while used < budget:
            th, q, ok, h = gradient_refine(rng.random(4), wt, min(25, budget - used), tol=tol)
            for v in h: rec(th, v)
            used += q
            if ok:
                return th, used, True, curve
        return best[1], budget, False, curve
    if method == 'seeded':     # probe phase then gradient refinement of the best probes
        n_probe = budget // 2
        U = rng.random((n_probe, 4)); TH = to_box(U)
        v = onp.abs(onp.asarray(W(TH)) / wt - 1)
        for k in range(n_probe):
            rec(TH[k], float(v[k]))
            if v[k] < tol:
                return TH[k], k + 1, True, curve
        used = n_probe
        for i in onp.argsort(v)[:4]:
            th, q, ok, h = gradient_refine(U[i], wt, min(25, budget - used), tol=tol)
            for vv in h: rec(th, vv)
            used += q
            if ok:
                return th, used, True, curve
            if used >= budget:
                break
        return best[1], used, False, curve


# ---------------- Stage 3 baseline: robust correction by SAA ----------------
def robust_opt(th0, wt, kind, xi, kappa=1.0, alpha=0.05, beta=1.0, iters=150):
    K = len(xi)
    if kind == 'mv':
        def obj(z):
            dv = dev(z[:4], xi, wt, kappa); mu = np.mean(dv)
            return mu + beta * np.sqrt(np.mean((dv - mu) ** 2) + 1e-12)
        z0 = onp.array(th0, float)
        bnds = list(zip(ILO, IHI))
    else:   # Rockafellar-Uryasev: CVaR = min_t t + E[(d - t)^+]/alpha
        def obj(z):
            dv = dev(z[:4], xi, wt, kappa)
            return z[4] + np.mean(np.maximum(dv - z[4], 0)) / alpha
        d0 = onp.asarray(dev(onp.array(th0, float), xi, wt, kappa))
        z0 = onp.r_[th0, onp.percentile(d0, 100 * (1 - alpha))]
        bnds = list(zip(ILO, IHI)) + [(0, 1)]
    vg = value_and_grad(obj)
    o = minimize(lambda z: (lambda v, g: (float(v), onp.asarray(g)))(*vg(z)), z0, jac=True,
                 method='L-BFGS-B', bounds=bnds, options=dict(maxiter=iters))
    return o.x[:4], int(o.nfev)
