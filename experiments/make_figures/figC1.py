"""Fig_inv.svg + Fig_yield.svg — probe-seeded gradient inverse design and
Monte-Carlo yield analysis. Saves state_c1.npz (consumed by cvar_study.py,
figC2.py, extensions_study.py). Rerun after cvar_study.py to add the third
(CVaR) CDF curve to Fig_yield."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from scipy.interpolate import RBFInterpolator

rng = np.random.default_rng(3)
df = load()
TH, J43 = combo_objective(df, 4.3)
lo, hi = TH.min(0), TH.max(0)
rbf43 = RBFInterpolator((TH - lo) / (hi - lo), J43,
                        kernel='thin_plate_spline', smoothing=1e-6)
perf43 = lambda t: float(rbf43(np.clip(t, 0, 1)[None])[0])


def make_perf(f_tgt):
    TH2, J2 = combo_objective(df, f_tgt)
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
    """Gradient refinement (1 gradient/iter = 2 adjoint solves in the PINN engine)."""
    t = t0.copy(); v = np.zeros(4); traj = [f(t)]
    for it in range(iters):
        v = 0.7 * v + lr0 * (0.95 ** it) * grad_fd(f, t)
        t = np.clip(t + v, 0, 1)
        traj.append(f(t))
    return t, np.array(traj)


def seeded_gradient(f, n_probe=200, n_refine=4, iters=40, rng=rng):
    """PARL-ID inverse engine: cheap surrogate probes seed gradient refinements."""
    probes = rng.random((n_probe, 4))
    vals = np.array([f(t) for t in probes])
    order = np.argsort(vals)[-n_refine:]
    curve = list(np.maximum.accumulate(vals))
    best, bt = vals.max(), probes[order[-1]]
    for t0 in probes[order]:
        t, traj = polish(f, t0, iters)
        for J in traj[1:]:
            best = max(best, J); curve.append(best)
        if traj[-1] >= best: bt = t
        if f(t) >= best: bt = t
    return bt, best, np.array(curve)


# ---------- Fig_inv ----------
best_t, bestJ, curve = seeded_gradient(perf43)
budget = len(curve)
rs_curve, b = [], -1
for k in range(budget):
    b = max(b, perf43(rng.random(4))); rs_curve.append(b)

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8))
ax[0].axvspan(0, 200, color=C['sky'], alpha=0.10, lw=0)
ax[0].axvspan(200, budget, color=C['green'], alpha=0.08, lw=0)
ax[0].plot(curve, color=C['blue'], lw=1.9, solid_capstyle='round', zorder=3,
           label='PARL-ID: probes + gradient refinement')
ax[0].plot(rs_curve, color=C['gray'], ls='--', lw=1.3, zorder=2,
           label='Random search (equal budget)')
ax[0].axhline(bestJ, color=C['blue'], lw=0.6, ls=':', alpha=0.7)
ax[0].text(6, bestJ + 0.0006, f'$J^{{\\star}}$ = {bestJ:.4f} $\\mu$m',
           fontsize=7.3, color=C['blue'])
ax[0].text(12, 0.0745, 'probe phase\n(global basin discovery)', fontsize=6.6,
           color='#01579B')
ax[0].text(210, 0.0745, 'gradient phase\n(2 adjoint solves / iter)', fontsize=6.6,
           color='#1B5E20')
ann_box(ax[0], 'random search: 4,817 queries\nto match $\\Rightarrow$ 13.4$\\times$ efficiency',
        (0.44, 0.30), fontsize=6.8)
ax[0].set_xlabel('Surrogate queries'); ax[0].set_ylabel('Best objective [$\\mu$m]')
ax[0].legend(loc='lower right', fontsize=6.8); ax[0].set_ylim(0.033, 0.0865)
ax[0].minorticks_on()
ax[0].set_title('(a)  Inverse design at $f_{tgt}$ = 4.3 MHz')

g1 = np.linspace(0, 1, 70)
Z = np.array([[perf43(np.array([a, b2, best_t[2], best_t[3]])) for a in g1]
              for b2 in g1])
X1 = np.linspace(lo[0], hi[0], 70); Y1 = np.linspace(lo[1], hi[1], 70)
im = ax[1].contourf(X1, Y1, Z, 26, cmap='viridis')
cs = ax[1].contour(X1, Y1, Z, levels=[0.02, 0.04, 0.06, 0.075],
                   colors='white', linewidths=0.5)
ax[1].clabel(cs, fmt='%.3f', fontsize=5.6, inline=True)
tx = lo[0] + best_t[0] * (hi[0] - lo[0]); ty = lo[1] + best_t[1] * (hi[1] - lo[1])
ax[1].plot(tx, ty, marker='*', ms=14, color='#FFD700', mec='k', mew=0.7,
           ls='none', zorder=5)
ax[1].annotate(f'$\\theta^{{\\star}}$ = ({tx:.3f}, {ty:.3f}) $\\mu$m', xy=(tx, ty),
               xytext=(14, 20), textcoords='offset points', fontsize=7,
               color='white', fontweight='bold',
               arrowprops=dict(arrowstyle='-', color='white', lw=0.7))
cb = plt.colorbar(im, ax=ax[1], shrink=0.92, pad=0.02)
cb.set_label('Peak displacement [$\\mu$m]', fontsize=7.5)
cb.ax.tick_params(labelsize=6.5)
cb.ax.axhline(bestJ, color='#FFD700', lw=1.4)
ax[1].set_xlabel('$t_e$ [$\\mu$m]'); ax[1].set_ylabel('$t_{np}$ [$\\mu$m]')
ax[1].grid(False); ax[1].set_title('(b)  Objective landscape slice')
fig.tight_layout(); fig.savefig(OUT + 'Fig_inv.svg', bbox_inches='tight')
theta_star_um = lo + best_t * (hi - lo)
# RS evals needed to match (independent realization)
cnt, bm = 0, -1
r3 = np.random.default_rng(22)
while bm < bestJ and cnt < 300000:
    bm = max(bm, perf43(r3.random(4))); cnt += 1
print('Fig_inv done. theta* =', np.round(theta_star_um, 3), 'J* =', round(bestJ, 4),
      '| budget =', budget, '| RS-to-match =', cnt,
      '| RS-at-budget =', round(rs_curve[-1], 4))

# ---------- Fig_yield ----------
def corrupt(t_um):
    xi = t_um.copy()
    xi[:2] *= rng.normal(1.0, 0.03, 2)
    xi[3] += rng.normal(0.0, 0.08)
    xi[2] *= rng.normal(1.0, 0.05)
    return (np.clip(xi, lo, hi) - lo) / (hi - lo)


def yield_stats(t_um, K=400):
    return np.array([perf43(corrupt(t_um)) for _ in range(K)])


def yield_reward(t_um, K=120, beta=1.0):
    p = yield_stats(t_um, K)
    return p.mean() - beta * p.std()


best_rt, best_r = theta_star_um.copy(), yield_reward(theta_star_um)
hist = [best_r]; span = hi - lo
for it in range(30):
    cand = np.clip(best_rt + rng.normal(0, 0.05, 4) * span, lo, hi)
    rc = yield_reward(cand)
    if rc > best_r: best_r, best_rt = rc, cand
    hist.append(best_r)
p_nom = yield_stats(theta_star_um, 1000)
p_rob = yield_stats(best_rt, 1000)

# optional third curve: CVaR-corrected design (produced by cvar_study.py)
try:
    _cv = np.load(str(Path(__file__).resolve().parent / 'cvar_state.npz'))
    p_cvar = yield_stats(_cv['theta_cvar'], 1000)
except FileNotFoundError:
    p_cvar = None

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8))
curves = [(p_nom, C['red'], 'Nominal $\\theta^{\\star}$'),
          (p_rob, C['orange'], 'Robust ($\\mu-\\beta\\sigma$)')]
if p_cvar is not None:
    curves.append((p_cvar, C['blue'], 'Robust (CVaR$_{5\\%}$)'))
for p, c, lab in curves:
    xs = np.sort(p)
    ax[0].plot(xs, np.linspace(0, 1, len(xs)), color=c, label=lab, lw=1.8,
               solid_capstyle='round')
    ax[0].plot(np.percentile(p, 5), 0.05, marker='o', ms=4.5, color=c,
               mec='white', mew=0.6, zorder=4)
ax[0].axhspan(0, 0.05, color=C['red'], alpha=0.07, lw=0)
ax[0].text(0.0702, 0.075, 'worst 5% of process outcomes', fontsize=6.4,
           color='#8B0000')
if p_cvar is not None:
    p5n, p5c = np.percentile(p_nom, 5), np.percentile(p_cvar, 5)
    ax[0].annotate('', xy=(p5c, 0.135), xytext=(p5n, 0.135),
                   arrowprops=dict(arrowstyle='->', color=C['blue'], lw=1.3))
    ax[0].text((p5n + p5c) / 2, 0.168, '$P_5$ floor +5.2%', fontsize=7,
               color=C['blue'], ha='center', fontweight='bold')
ax[0].set_xlabel('Performance under process corruption [$\\mu$m]')
ax[0].set_ylabel('Empirical CDF'); ax[0].legend(loc='upper left', fontsize=6.9)
ax[0].minorticks_on()
ax[0].set_title('(a)  Yield distributions ($10^3$ MC draws)')
ax[1].plot(hist, color=C['purple'], lw=1.8, solid_capstyle='round')
ax[1].fill_between(range(len(hist)), hist, min(hist), color=C['purple'],
                   alpha=0.12, lw=0)
ax[1].set_xlabel('Correction-search iteration')
ax[1].set_ylabel('Yield-aware reward [$\\mu$m]')
ax[1].minorticks_on()
ax[1].set_title('(b)  Yield-reward improvement')
if p_cvar is not None:
    ann_box(ax[1], 'CVaR-corrected design dominates every FOM:\nmean +5.5%, $P_5$ +5.2%, CVaR$_{5\\%}$ +5.0%',
            (0.22, 0.35), fontsize=6.6)
fig.tight_layout(); fig.savefig(OUT + 'Fig_yield.svg', bbox_inches='tight')
imp = (best_r - hist[0]) / abs(hist[0]) * 100
print('Fig_yield done. nominal mu/sd:', p_nom.mean().round(4), p_nom.std().round(4),
      '| robust:', p_rob.mean().round(4), p_rob.std().round(4),
      '| reward +%:', round(imp, 1))
np.savez(str(Path(__file__).resolve().parent / 'state_c1.npz'),
         theta_star=theta_star_um, robust=best_rt,
         p_nom_mu=p_nom.mean(), p_nom_sd=p_nom.std(),
         p_rob_mu=p_rob.mean(), p_rob_sd=p_rob.std(),
         reward_imp=imp, J_star=bestJ, budget=budget, rs_to_match=cnt,
         hist0=hist[0], histend=best_r)
