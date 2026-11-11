"""E1 (Reviewer 1, point 1): physics-informed surrogate versus data-only
baseline under held-out-DESIGN evaluation (5-fold grouped CV over the FEM
designs), with ablations of the residual weight, boundary encoding, rigidity
model and Fourier-feature bandwidth.

Target: band-averaged displacement of each FEM design (the frequency profiles
are flat to a median coefficient of variation of 0.1 %, see e0_data_audit.py).
Physics-informed model (single-mode variational form of Eq. (1)):
    W(x,y;theta) = psi(x,y) q(theta),  psi = phi^2 (clamped, hard BC)
    Galerkin residual  R = q K(theta) - F,  K = D(theta) I4 / a^4
    log q = log(F/K(theta)) + g_NN(theta)          (F, a calibrated)
    loss  = MSE(log q, log W_FEM) + lam * mean_c (exp(g_c) - 1)^2
The residual term is the normalized Galerkin residual R/F at 600 unlabeled
collocation designs; lam = 0 removes it.  Resumable (results cached)."""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import *
from autograd.misc.flatten import flatten
from scipy.optimize import minimize

b, designs, _ = load_base()
Wbar = b.groupby('design')['displacement_um'].mean().values
lo, hi = designs.min(0), designs.max(0)
SUBSETS = {'all123': onp.arange(len(designs)),
           'clean54': onp.where(designs[:, 0] >= 0.44)[0]}
I_CL = modal_integrals('clamped'); I_SS = modal_integrals('ss')


def xin(th):
    return (th - lo) / (hi - lo) * 2 - 1


def log_prior(th, cfg):
    I0, I1, I4 = cfg['I']
    D = rigidity(th[:, 0], th[:, 1], cfg['rig'])
    return np.log(I1 ** 2 / I4) - np.log(D)     # log(F a^4 / (D I4)) up to calibrated const


def build(cfg, seed):
    r = onp.random.default_rng(seed)
    B = None if cfg['sigma'] is None else r.normal(0, cfg['sigma'], (4, 16))
    din = 4 if B is None else 36
    return {'net': init_mlp([din, cfg['width'], cfg['width'], 1], r), 'c': onp.zeros(1)}, B


def predict(p, B, th, cfg):
    g = mlp(p['net'], fourier(xin(th), B))[:, 0]
    base = log_prior(th, cfg) if cfg['phys'] else 0.0
    return p['c'][0] + base + g, g


def fit(cfg, th, ly, seed=0):
    p, B = build(cfg, seed)
    base = log_prior(th, cfg) if cfg['phys'] else 0.0
    p['c'][0] = float(onp.mean(ly - base))
    r = onp.random.default_rng(seed + 11)
    thc = lo + (hi - lo) * r.random((600, 4))
    def loss(p):
        lp, _ = predict(p, B, th, cfg)
        L = np.mean((lp - ly) ** 2) + cfg['l2'] * sum(np.sum(W ** 2) for W, _ in p['net'])
        if cfg['phys'] and cfg['lam'] > 0:
            _, gc = predict(p, B, thc, cfg)
            L = L + cfg['lam'] * np.mean((np.exp(gc) - 1) ** 2)
        return L
    fl, un = flatten(p)
    vg = value_and_grad(lambda q: loss(un(q)))
    o = minimize(lambda q: (lambda v, g: (float(v), onp.asarray(g)))(*vg(q)), fl, jac=True,
                 method='L-BFGS-B', options=dict(maxiter=cfg['iters']))
    return un(o.x), B


BASE = dict(phys=True, lam=0.1, sigma=None, I=I_CL, rig='laminate', width=16, l2=1e-3, iters=300)
CONFIGS = {
    'data_only':        dict(BASE, phys=False),
    'data_only_ff1':    dict(BASE, phys=False, sigma=1.0),
    'pinn_lam0':        dict(BASE, lam=0.0),
    'pinn_lam0.01':     dict(BASE, lam=0.01),
    'pinn_lam0.1':      dict(BASE, lam=0.1),
    'pinn_lam1':        dict(BASE, lam=1.0),
    'pinn_bc_ss':       dict(BASE, I=I_SS),
    'pinn_rig_mixture': dict(BASE, rig='mixture'),
    'pinn_ff1':         dict(BASE, sigma=1.0),
    'pinn_ff3':         dict(BASE, sigma=3.0),
}


def cv_run(name, subset, k):
    cfg = CONFIGS[name]; idx = SUBSETS[subset]
    te = group_folds(len(idx), 5, 0)[k]; tr = onp.setdiff1d(onp.arange(len(idx)), te)
    p, B = fit(cfg, designs[idx][tr], onp.log(Wbar[idx][tr]), seed=k)
    lp, _ = predict(p, B, designs[idx][te], cfg)
    return [float(v) for v in onp.exp(lp)], [float(v) for v in Wbar[idx][te]]


def summarize():
    res = load_json('e1_surrogate.json'); out = {}
    for subset in SUBSETS:
        for name in CONFIGS:
            ks = [res.get(f'{subset}|{name}|{k}') for k in range(5)]
            if any(v is None for v in ks):
                continue
            yp = onp.concatenate([v[0] for v in ks]); yt = onp.concatenate([v[1] for v in ks])
            fold_r2 = [r2(onp.array(v[1]), onp.array(v[0])) for v in ks]
            out[f'{subset}|{name}'] = dict(R2=r2(yt, yp), MAE_nm=float(onp.abs(yt - yp).mean() * 1e3),
                                          R2_fold_sd=float(onp.std(fold_r2)))
    save_json('e1_summary.json', out)
    return out


if __name__ == '__main__':
    budget = float(sys.argv[1]); part = int(sys.argv[2]); nparts = int(sys.argv[3])
    t0 = time.time()
    jobs = [(s, n, k) for s in SUBSETS for n in CONFIGS for k in range(5)][part::nparts]
    fn = f'e1_part{part}.json'
    try:
        res = load_json(fn)
    except FileNotFoundError:
        res = {}
    for s, n, k in jobs:
        key = f'{s}|{n}|{k}'
        if key in res:
            continue
        if time.time() - t0 > budget:
            break
        res[key] = cv_run(n, s, k); save_json(fn, res)
    print('part', part, len(res), '/', len(jobs), flush=True)
    merged = {}
    for f in RES.glob('e1_part*.json'):
        merged.update(load_json(f.name))
    if len(merged) == len(SUBSETS) * len(CONFIGS) * 5:
        save_json('e1_surrogate.json', merged); summarize(); print('summary written')
