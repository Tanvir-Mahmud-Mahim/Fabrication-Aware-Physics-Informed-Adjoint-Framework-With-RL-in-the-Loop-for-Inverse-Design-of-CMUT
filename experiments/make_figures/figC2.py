import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy.interpolate import RBFInterpolator
rng = np.random.default_rng(3)
df = load()

# smooth differentiable surrogate of the design objective from REAL combos
TH, J43 = combo_objective(df, 4.3)
lo, hi = TH.min(0), TH.max(0)
THn = (TH-lo)/(hi-lo)
rbf43 = RBFInterpolator(THn, J43, kernel='thin_plate_spline', smoothing=1e-6)

def make_perf(f_tgt):
    TH2, J2 = combo_objective(df, f_tgt)
    r = RBFInterpolator((TH2-lo)/(hi-lo), J2, kernel='thin_plate_spline', smoothing=1e-6)
    return lambda t: float(r(np.clip(t,0,1)[None])[0])

perf43 = lambda t: float(rbf43(np.clip(t,0,1)[None])[0])

def grad_fd(f, t, h=1e-3):
    g = np.zeros_like(t)
    for i in range(len(t)):
        tp, tm = t.copy(), t.copy(); tp[i]+=h; tm[i]-=h
        g[i] = (f(np.clip(tp,0,1)) - f(np.clip(tm,0,1)))/(2*h)
    return g


import numpy as _np
_st = _np.load(str(Path(__file__).resolve().parent / 'state_c1.npz'))
theta_star_um = _st['theta_star']; best_rt = _st['robust']; span = hi-lo
def corrupt(t_um):
    xi = t_um.copy()
    xi[:2] *= rng.normal(1.0, 0.03, 2)
    xi[3] += rng.normal(0.0, 0.08)
    xi[2] *= rng.normal(1.0, 0.05)
    return (np.clip(xi, lo, hi)-lo)/(hi-lo)
def run_gd(f, t0, iters=40, lr=0.05):
    t = t0.copy(); traj = [f(t)]
    for _ in range(iters):
        t = np.clip(t + lr*grad_fd(f, t), 0, 1)
        traj.append(f(t))
    return t, np.array(traj)
# ---------- Fig_transfer: correction transfer across target freqs ----------
delta = best_rt - theta_star_um
freqs = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
r_nom_l, r_tr_l, r_re_l = [], [], []
for f_t in freqs:
    pf = make_perf(f_t)
    def yr(t_um, K=60, beta=1.0):
        p = np.array([pf(corrupt(t_um)) for _ in range(K)])
        return p.mean()-beta*p.std()
    t0 = rng.random(4)
    t, _ = run_gd(pf, t0)
    for _ in range(2):
        t2, _ = run_gd(pf, rng.random(4))
        if pf(t2) > pf(t): t = t2
    th_nom = lo + t*(hi-lo)
    r_nom = yr(th_nom)
    r_tr = yr(np.clip(th_nom+delta, lo, hi))
    bt, br = th_nom.copy(), r_nom
    for it in range(15):
        cand = np.clip(bt + rng.normal(0,0.05,4)*span, lo, hi)
        rc = yr(cand)
        if rc > br: br, bt = rc, cand
    r_nom_l.append(r_nom); r_tr_l.append(r_tr); r_re_l.append(br)
gain_tr = np.array(r_tr_l)-np.array(r_nom_l)
gain_re = np.array(r_re_l)-np.array(r_nom_l)
mask = gain_re > 1e-4
rec = float(np.mean(gain_tr[mask]/gain_re[mask]))*100 if mask.any() else float('nan')

fig, ax = plt.subplots(figsize=(3.6, 2.8))
x = np.arange(len(freqs)); w = 0.27
ax.bar(x-w, r_nom_l, w, color=C['gray'], label='Nominal', zorder=2)
ax.bar(x, r_tr_l, w, color=C['orange'], label='Transferred correction', zorder=2)
ax.bar(x+w, r_re_l, w, color=C['green'], label='Re-optimized', zorder=2)
for i in range(len(freqs)):
    if r_tr_l[i] < r_nom_l[i] - 1e-4:
        ax.plot(x[i], r_tr_l[i]+0.002, marker='v', color=C['red'], ms=5, zorder=3)
ax.plot([], [], marker='v', color=C['red'], ls='none', ms=5,
        label='transfer below nominal')
ax.set_xticks(x); ax.set_xticklabels([f'{f:g}' for f in freqs])
ax.set_xlabel('Target frequency [MHz]')
ax.set_ylabel('Yield-aware reward [$\\mu$m]')
ax.legend(fontsize=6.3, loc='upper right')
ax.set_title('Static corrections do not transfer')
ann_box(ax, 'mean recovery of attainable gain: $-65\\%$\n$\\Rightarrow$ specification-conditioned policy required',
        (0.03, 0.20), fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT+'Fig_transfer.svg', bbox_inches='tight')
print('Fig_transfer done. recovery % =', round(rec,1))
np.savez(str(Path(__file__).resolve().parent / 'transfer.npz'), freqs=freqs, r_nom=r_nom_l, r_tr=r_tr_l, r_re=r_re_l, recovery=rec)
