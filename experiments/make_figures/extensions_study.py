"""Extensions study: adopt-and-strengthen mechanisms from contemporary SOTA.

Quantifies, on the released FEM database (numpy/scipy only):
  (1) Spec-to-spec warm starting (Soda-PTA-inspired, design transfer instead
      of hyperparameter transfer): query savings when the optimum at one
      frequency seeds the gradient engine at neighboring frequencies, vs. the
      full 200-probe cold start.
  (2) Bandwidth-aware objective (TSMC-grating-coupler-inspired): peak-only vs.
      band-mean optimum, and each design's performance under both criteria.
  (3) Rank quality of the objective surrogate (PearSAN-motivated): Pearson and
      Spearman correlation between RBF-surrogate scores and FEM values on
      held-out combos --- the quantity a rank-aware surrogate loss improves.
Writes extensions_state.npz. Run AFTER figC1.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy.interpolate import RBFInterpolator
from scipy.stats import pearsonr, spearmanr
import numpy as np

_here = Path(__file__).resolve().parent
df = load()
TH, _ = combo_objective(df, 4.3)
lo, hi = TH.min(0), TH.max(0)


def make_perf(f_tgt, band=False):
    fn = combo_objective_band if band else combo_objective
    TH2, J2 = fn(df, f_tgt)
    r = RBFInterpolator((TH2 - lo) / (hi - lo), J2,
                        kernel='thin_plate_spline', smoothing=1e-6)
    return lambda t: float(r(np.clip(t, 0, 1)[None])[0])


def grad_fd(f, t, h=1e-3):
    g = np.zeros_like(t)
    for i in range(len(t)):
        tp, tm = t.copy(), t.copy(); tp[i] += h; tm[i] -= h
        g[i] = (f(np.clip(tp, 0, 1)) - f(np.clip(tm, 0, 1))) / (2 * h)
    return g


def polish(f, t0, iters=40, lr0=0.03):
    t = t0.copy(); v = np.zeros(4)
    for it in range(iters):
        v = 0.7 * v + lr0 * (0.95 ** it) * grad_fd(f, t)
        t = np.clip(t + v, 0, 1)
    return t, f(t)


def cold_start(f, rng, n_probe=200, n_refine=4):
    probes = rng.random((n_probe, 4))
    vals = np.array([f(t) for t in probes])
    best, bt = -np.inf, None
    for t0 in probes[np.argsort(vals)[-n_refine:]]:
        t, J = polish(f, t0)
        if J > best: best, bt = J, t
    queries = n_probe + n_refine * 40
    return bt, best, queries


# ---------- (1) spec-to-spec warm start ----------
rng = np.random.default_rng(7)
st = np.load(_here / 'state_c1.npz')
theta_43 = (st['theta_star'] - lo) / (hi - lo)   # solved design at 4.3 MHz
print('--- (1) spec-to-spec warm start (seed: 4.3 MHz optimum) ---')
ws_rows = []
for f_t in [4.0, 4.5, 5.0]:
    f = make_perf(f_t)
    t_c, J_c, q_c = cold_start(f, np.random.default_rng(int(f_t * 10)))
    t_w, J_w = polish(f, theta_43)               # warm: 40 iters only
    q_w = 40
    ws_rows.append([f_t, J_c, q_c, J_w, q_w])
    print(f'  f={f_t} MHz  cold: J={J_c:.4f} ({q_c} queries) | '
          f'warm: J={J_w:.4f} ({q_w} queries) | '
          f'J retained {J_w/J_c*100:.1f}% at {q_w/q_c*100:.0f}% budget')

# ---------- (2) bandwidth-aware objective ----------
print('--- (2) peak-only vs. bandwidth-aware objective (4.3 +/- 0.5 MHz) ---')
f_peak = make_perf(4.3, band=False)
f_band = make_perf(4.3, band=True)
t_p, J_p, _ = cold_start(f_peak, np.random.default_rng(1))
t_b, J_b, _ = cold_start(f_band, np.random.default_rng(1))
print(f'  peak-optimal design:  peak={f_peak(t_p):.4f}  band-mean={f_band(t_p):.4f}')
print(f'  band-optimal design:  peak={f_peak(t_b):.4f}  band-mean={f_band(t_b):.4f}')
print(f'  band-mean gain of band-aware design: '
      f'{(f_band(t_b)-f_band(t_p))/f_band(t_p)*100:+.1f}% '
      f'at peak cost {(f_peak(t_b)-f_peak(t_p))/f_peak(t_p)*100:+.1f}%')

# ---------- (3) surrogate rank quality on held-out combos ----------
print('--- (3) objective-surrogate rank quality (5-fold over combos) ---')
TH3, J3 = combo_objective(df, 4.3)
THn = (TH3 - lo) / (hi - lo)
rngk = np.random.default_rng(0)
idx = rngk.permutation(len(J3))
pear, spear = [], []
for k in range(5):
    te = idx[k::5]; tr = np.setdiff1d(idx, te)
    r = RBFInterpolator(THn[tr], J3[tr], kernel='thin_plate_spline', smoothing=1e-6)
    pred = r(THn[te])
    pear.append(pearsonr(pred, J3[te])[0])
    spear.append(spearmanr(pred, J3[te])[0])
print(f'  Pearson rho = {np.mean(pear):.3f} +/- {np.std(pear):.3f} | '
      f'Spearman rho = {np.mean(spear):.3f} +/- {np.std(spear):.3f}')

np.savez(_here / 'extensions_state.npz',
         warmstart=np.array(ws_rows),
         theta_peak=lo + t_p * (hi - lo), theta_band=lo + t_b * (hi - lo),
         peak_of_peak=f_peak(t_p), band_of_peak=f_band(t_p),
         peak_of_band=f_peak(t_b), band_of_band=f_band(t_b),
         pearson=np.mean(pear), spearman=np.mean(spear))
print('saved extensions_state.npz')
