"""Regenerates every revision figure (SVG, into latex/) from results/*.json."""
import sys, json, pickle, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'make_figures')); sys.path.insert(0, str(HERE))
from common import plt, C, ann_box
import numpy as onp
R = HERE / 'results'; OUT = str(HERE.parent.parent.parent / 'latex') + '/'
J = lambda n: json.load(open(R / n))

# ---------- Fig_surrogate: held-out-design accuracy + ablations ----------
s = J('e1_summary.json')
names = [('data_only', 'Data-only network'), ('pinn_lam0', 'Physics prior, no residual'), ('pinn_lam0.01', 'Physics, residual weight 0.01'),
         ('pinn_lam0.1', 'Physics, residual weight 0.1'), ('pinn_lam1', 'Physics, residual weight 1'), ('pinn_rig_mixture', 'Physics, rule-of-mixtures stiffness'),
         ('pinn_bc_ss', 'Physics, edges not clamped'), ('pinn_ff1', 'Physics, Fourier features'), ('data_only_ff1', 'Data-only, Fourier features')]
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
y = onp.arange(len(names)); h = 0.38
for j, (sub, col, lab) in enumerate([('all123', C['blue'], 'all 123 FEM designs'), ('clean54', C['orange'], '54-design subset ($t_e\\geq$0.44 $\\mu$m)')]):
    v = [max(s[f'{sub}|{n}']['R2'], -0.5) for n, _ in names]
    ax[0].barh(y + (j - 0.5) * h, v, h, color=col, label=lab, zorder=2)
    for yy, vv, n in zip(y, v, names):
        raw = s[f'{sub}|{n[0]}']['R2']
        ax[0].text(max(vv, 0) + 0.02, yy + (j - 0.5) * h, f'{raw:.2f}', va='center', fontsize=7.5)
ax[0].set_yticks(y); ax[0].set_yticklabels([n for _, n in names], fontsize=7.5); ax[0].invert_yaxis()
ax[0].axvline(0, color='k', lw=0.6); ax[0].set_xlim(-0.5, 1.12); ax[0].set_xlabel('$R^2$ on held-out designs (5-fold)')
ax[0].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=2); ax[0].set_title('(a) Prediction accuracy on unseen designs')
a = J('e0_audit.json')
from rev_common import load_base
b, d, _ = load_base(); m = b.groupby('design')['displacement_um'].mean().values
o = onp.lexsort((d[:, 3], d[:, 2], d[:, 1], d[:, 0]))
ax[1].plot(onp.arange(69), m[o][:69], '.-', color=C['red'], ms=3, lw=0.7, label='$t_e$ = 0.24 $\\mu$m block (5 $t_w$ levels)')
ax[1].plot(onp.arange(69, 123), m[o][69:], '.-', color=C['blue'], ms=3, lw=0.7, label='$t_e$ = 0.44/0.64 $\\mu$m blocks (3 $t_w$ levels)')
for jp in a['jump_positions_te024'][:-1]: ax[1].axvline(jp + 0.5, color=C['red'], lw=0.3, alpha=0.5)
ax[1].set_xlabel('FEM design index (sorted $t_e, t_{np}, t_{ox}, t_w$)'); ax[1].set_ylabel('Band-averaged $W$ [$\\mu$m]')
ax[1].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.22)); ax[1].set_title('(b) Database check: where the values jump')
fig.tight_layout(); fig.savefig(OUT + 'Fig_surrogate.svg', bbox_inches='tight'); plt.close(fig)

# ---------- Fig_engine: inverse engines ----------
e = J('e2b_engine.json'); mx = J('e2c_max.json')
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.6))
for k, col, lab in [('random', C['gray'], 'Random search'), ('gradient', C['blue'], 'Gradient descent (restarts)'), ('seeded', C['orange'], 'Random probes, then gradient')]:
    c = onp.array(e[k]['curve_median']) * 100; q1 = onp.array(e[k]['curve_q25']) * 100; q3 = onp.array(e[k]['curve_q75']) * 100
    ax[0].semilogy(onp.arange(1, 201), onp.maximum(c, 1e-3), color=col, lw=1.6, label=f"{lab} ({e[k]['success_rate'] * 100:.0f}%)")
    ax[0].fill_between(onp.arange(1, 201), onp.maximum(q1, 1e-3), onp.maximum(q3, 1e-3), color=col, alpha=0.15, lw=0)
ax[0].axhline(0.5, color='k', ls=':', lw=0.8); ax[0].set_xlabel('Number of model evaluations'); ax[0].set_ylabel('Best error from target [%]')
ax[0].legend(fontsize=7.5, loc='upper right', title='success rate', title_fontsize=7.5); ax[0].set_title('(a) Hitting a target displacement')
ks = ['random', 'gradient', 'seeded']; labs = ['Random\nsearch', 'Gradient\n(restarts)', 'Random probes,\nthen gradient']
ax[1].bar(range(3), [mx[k]['mean'] for k in ks], yerr=[mx[k]['sd'] for k in ks], color=[C['gray'], C['blue'], C['orange']], capsize=3, zorder=2)
ax[1].axhline(mx['fem_best'], color=C['red'], ls='--', lw=1.0, label=f"best FEM design ({mx['fem_best']:.4f} $\\mu$m)")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labs, fontsize=7.5); ax[1].set_ylim(0.06, 0.095)
ax[1].set_ylabel('Best $W$ found [$\\mu$m]'); ax[1].legend(fontsize=7.5, loc='upper left'); ax[1].set_title('(b) Maximizing displacement')
fig.tight_layout(); fig.savefig(OUT + 'Fig_engine.svg', bbox_inches='tight'); plt.close(fig)

# ---------- Fig_robust: yield and tail risk per specification ----------
rb = J('e2d_robust.json'); wts = sorted(rb, key=float)
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.6)); x = onp.arange(len(wts)); w = 0.27
for j, (k, col, lab) in enumerate([('nominal', C['gray'], 'Nominal $\\theta^\\star$'), ('mv', C['orange'], 'Corrected: mean + spread'), ('cvar', C['blue'], 'Corrected: worst 5% (CVaR)')]):
    ax[0].bar(x + (j - 1) * w, [rb[t][k]['yield_5pct'] * 100 for t in wts], w, color=col, label=lab, zorder=2)
    ax[1].bar(x + (j - 1) * w, [rb[t][k]['cvar5_dev'] * 100 for t in wts], w, color=col, label=lab, zorder=2)
for a_, yl, tl in [(ax[0], 'Devices within $\\pm$5% [%]', '(a) Share of devices within 5% of target'), (ax[1], 'Worst-5% error [%]', '(b) Average error of the worst 5% (lower is better)')]:
    a_.set_xticks(x); a_.set_xticklabels([f'{float(t) * 1e3:.0f}' for t in wts]); a_.set_xlabel('Target $W_{tgt}$ [nm]'); a_.set_ylabel(yl); a_.set_title(tl)
ax[0].set_ylim(0, 105)
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=7.5, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(); fig.savefig(OUT + 'Fig_robust.svg', bbox_inches='tight'); plt.close(fig)

# ---------- Fig_policy: learned policies ----------
ev = J('e3c_eval.json')
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
cmap = {'sac_gcn_per': (C['blue'], 'GCN-SAC + PER'), 'sac_gcn_uniform': (C['sky'], 'GCN-SAC, uniform replay'),
        'sac_mlp_per': (C['purple'], 'MLP-SAC + PER'), 'ppo_gcn': (C['green'], 'GCN-PPO')}
seen = set()
for k, log in ev['rl_curves'].items():
    ag = k.rsplit('_s', 1)[0]; col, lab = cmap[ag]
    st = [p[0] for p in log]; g = [p[1] for p in log]
    ax[0].plot(st, g, '-o', color=col, ms=2.5, lw=1.1, label=lab if ag not in seen else None); seen.add(ag)
orc = onp.mean([ev['nominal']['per_spec'][s]['cvar5_dev'] - ev['oracle_saa']['per_spec'][s]['cvar5_dev'] for s in ev['nominal']['per_spec']])
ax[0].axhline(orc, color=C['red'], ls='--', lw=1.0, label='re-optimized for each target'); ax[0].axhline(0, color='k', lw=0.6)
ax[0].set_ylim(-0.45, 0.12); ax[0].set_xlabel('Environment steps'); ax[0].set_ylabel('Gain on unseen targets')
ax[0].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2); ax[0].set_title('(a) Reinforcement-learning training')
rows = [('nominal', 'Nominal'), ('static', 'Static\nvector'), ('adjoint_policy', 'Gradient-\ntrained\npolicy'),
        ('sac_gcn_per', 'GCN-SAC\n+PER'), ('ppo_gcn', 'GCN-\nPPO'), ('oracle_saa', 'Re-optimized\nper target')]
vals, errs = [], []
for k, _ in rows:
    ks = [kk for kk in ev if kk == k or kk.startswith(k + '_s')]
    v = [ev[kk]['mean_cvar5'] * 100 for kk in ks]; vals.append(onp.mean(v)); errs.append(onp.std(v))
ax[1].bar(range(len(rows)), vals, yerr=errs, capsize=3, color=[C['gray'], C['orange'], C['purple'], C['blue'], C['green'], C['red']], zorder=2)
ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels([r.replace('-\n', '-').replace('\n', ' ') for _, r in rows], fontsize=7.5, rotation=35, ha='right')
ax[1].set_ylabel('Worst-5% error [%]'); ax[1].set_title('(b) Result on unseen targets')
fig.tight_layout(); fig.savefig(OUT + 'Fig_policy.svg', bbox_inches='tight'); plt.close(fig)

# ---------- Fig_photonic ----------
fa = J('e4a_fidelity.json'); fy = J('e4b_yield.json')
fig, ax = plt.subplots(1, 3, figsize=(7.1, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15, 1.1]))
ns = ['100', '200', '430']
ax[0].boxplot([fa[n]['cos_all'] for n in ns], widths=0.5, medianprops=dict(color=C['red']))
ax[0].set_xticklabels([f'N={n}\n$R^2$={fa[n]["R2_test"]:.2f}' for n in ns], fontsize=7.5); ax[0].axhline(1, color=C['green'], ls='--', lw=0.8)
ax[0].axhline(0, color='k', lw=0.5); ax[0].set_ylim(-1.05, 1.1); ax[0].set_ylabel('Cosine with exact gradient'); ax[0].set_title('(a) Surrogate gradient')
for k, col, lab in [('nominal', C['gray'], 'Nominal'), ('triple', C['orange'], 'Worst of three'), ('cvar', C['blue'], 'Worst 2 of 6 (CVaR)')]:
    T = onp.sort(fy[k]['T']); ax[1].plot(T, onp.linspace(0, 1, len(T)), color=col, lw=1.6, label=f"{lab}: Y$_{{0.9}}$={fy[k]['yield_090'] * 100:.0f}%")
ax[1].axvline(0.9, color='k', ls=':', lw=0.8); ax[1].set_xlabel('Transmission $|S_{21}|^2$'); ax[1].set_ylabel('Empirical CDF'); ax[1].set_xlim(0.4, 1.0)
ax[1].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.3)); ax[1].set_title('(b) Bend, fabrication errors')
sys.argv = ['x']; import e4_photonic as P
xs = [pickle.load(open(R / f'e4_opt_{k}.pkl', 'rb'))['x'] for k in ['nominal', 'cvar']]
img = onp.concatenate([P.density(xs[0], 0.5).T, onp.ones((40, 3)), P.density(xs[1], 0.5).T], 1)
ax[2].imshow(img, origin='lower', cmap='Greys', vmin=0, vmax=1); ax[2].set_xticks([20, 63]); ax[2].set_xticklabels(['Nominal', 'CVaR'], fontsize=7.5)
ax[2].set_yticks([]); ax[2].grid(False); ax[2].set_title('(c) Designs ($\\eta$ = 0.5)')
fig.tight_layout(); fig.savefig(OUT + 'Fig_photonic.svg', bbox_inches='tight'); plt.close(fig)
print('figures written')
