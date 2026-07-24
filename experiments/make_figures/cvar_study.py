"""CVaR tail-risk correction study (paper Section VI-D, Table FOM panel C).

Optimizes CVaR_5% of the corrupted-performance distribution around the
seeded-gradient optimum (state_c1.npz, produced by figC1.py) and evaluates
nominal / mean-variance-corrected / CVaR-corrected designs on a common set
of 2,000 Monte-Carlo process corruptions. Writes cvar_state.npz, which
figC1.py picks up to draw the third CDF curve in Fig_yield.
Run order: figC1.py -> cvar_study.py -> figC1.py (refresh figure) -> figC2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy.interpolate import RBFInterpolator
import numpy as np

df = load()
TH, J43 = combo_objective(df, 4.3)
lo, hi = TH.min(0), TH.max(0)
rbf = RBFInterpolator((TH-lo)/(hi-lo), J43, kernel='thin_plate_spline',
                      smoothing=1e-6)
f = lambda t: float(rbf(np.clip(t, 0, 1)[None])[0])

_here = Path(__file__).resolve().parent
st = np.load(_here / 'state_c1.npz')
theta_star, theta_rob_ms = st['theta_star'], st['robust']


def corrupt(t_um, r):
    xi = t_um.copy()
    xi[:2] *= r.normal(1.0, 0.03, 2)
    xi[3] += r.normal(0.0, 0.08)
    xi[2] *= r.normal(1.0, 0.05)
    return (np.clip(xi, lo, hi) - lo) / (hi - lo)


def perf_samples(t_um, K, seed):
    r = np.random.default_rng(seed)
    return np.array([f(corrupt(t_um, r)) for _ in range(K)])


def cvar(p, alpha=0.05):
    k = max(1, int(alpha * len(p)))
    return float(np.sort(p)[:k].mean())


# --- CVaR-corrected design: local search with common random numbers ---
best_ct, seed0 = theta_star.copy(), 1234
best_c = cvar(perf_samples(best_ct, 200, seed0))
r = np.random.default_rng(5)
span = hi - lo
for it in range(30):
    cand = np.clip(best_ct + r.normal(0, 0.05, 4) * span, lo, hi)
    c = cvar(perf_samples(cand, 200, seed0))
    if c > best_c:
        best_c, best_ct = c, cand

# --- common-seed evaluation of all designs (K = 2000, independent seed) ---
K = 2000
th_local = np.array([0.227, 1.648, 2.711, 3.162])  # unseeded local optimum
designs = [('shallow local optimum', th_local), ('nominal', theta_star),
           ('mean-variance corrected', theta_rob_ms),
           ('CVaR-corrected', best_ct)]
for name, th in designs:
    p = perf_samples(th, K, 777)
    print(f"{name:26s} mu={p.mean():.5f} sd={p.std():.2e} "
          f"sd/mu={p.std()/p.mean()*100:.1f}% P5={np.percentile(p, 5):.5f} "
          f"CVaR5={cvar(p):.5f} mu-sd={p.mean()-p.std():.5f}")
print('CVaR-corrected theta [um]:', np.round(best_ct, 3))
np.savez(_here / 'cvar_state.npz', theta_cvar=best_ct)
