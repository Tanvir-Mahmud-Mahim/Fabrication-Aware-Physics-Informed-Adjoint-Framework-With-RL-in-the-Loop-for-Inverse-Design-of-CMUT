"""E2e (Reviewer 1, point 2, CVaR noise): sampling noise of the reward
estimators.  For 30 candidate designs on the W_tgt = 0.030 um iso-performance
set, each estimator is computed 100 (K = 120) or 40 (K > 120) times at sample size K; reported are the
mean relative standard error and the Spearman rank correlation with a
K = 20000 reference (ranking fidelity is what the optimizer needs)."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_design as E2
from rev_common import *
from scipy.stats import spearmanr
wt = 0.030; r = onp.random.default_rng(3)
CF = RES / 'e2e_cands.npz'
if CF.exists():
    cands = list(onp.load(CF)['c'])
else:
    cands = []
    while len(cands) < 30:
        th, q, ok, _ = E2.engine('gradient', wt, 200, r)
        if ok: cands.append(th)
    onp.savez(CF, c=onp.array(cands))
def mv(dv): return dv.mean() + dv.std()
def cv5(dv): return E2.cvar_upper(dv, 0.05)
def cv10(dv): return E2.cvar_upper(dv, 0.10)
ests = {'mean-variance': mv, 'CVaR_5%': cv5, 'CVaR_10%': cv10}
RF = RES / 'e2e_ref.npz'
if RF.exists():
    ref = dict(onp.load(RF))
else:
    xr = onp.random.default_rng(99).normal(size=(20000, 4))
    dr = [onp.asarray(E2.dev(c, xr, wt)) for c in cands]
    ref = {n: onp.array([f(dv) for dv in dr]) for n, f in ests.items()}
    onp.savez(RF, **ref)
try:
    out = load_json('e2e_noise.json')
except FileNotFoundError:
    out = {}
for K in [int(a) for a in sys.argv[1:]]:
    NR = 100 if K <= 120 else 40
    vals = {n: onp.zeros((NR, 30)) for n in ests}
    for rep in range(NR):
        xi = onp.random.default_rng(1000 + rep).normal(size=(K, 4))
        dvs = [onp.asarray(E2.dev(c, xi, wt)) for c in cands]
        for n, f in ests.items():
            vals[n][rep] = [f(dv) for dv in dvs]
    for n in ests:
        rse = float(onp.mean(vals[n].std(0) / vals[n].mean(0)))
        rho = float(onp.mean([spearmanr(vals[n][i], ref[n]).correlation for i in range(NR)]))
        out[f'K{K}|{n}'] = dict(rel_std_err=rse, spearman_vs_ref=rho, tail_samples=int(onp.ceil({'mean-variance': 1, 'CVaR_5%': .05, 'CVaR_10%': .1}[n] * K)))
        print(K, n, round(rse, 4), round(rho, 3), flush=True)
    save_json('e2e_noise.json', out)
