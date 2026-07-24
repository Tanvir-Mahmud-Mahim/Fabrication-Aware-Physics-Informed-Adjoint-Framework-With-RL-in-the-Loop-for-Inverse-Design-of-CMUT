"""Fig_ablation.svg — component ablation of PARL-ID (paper Sec. VI, ablation).

(a) Inverse-engine ablation at a fixed 360-query budget.
(b) Fabrication-objective ablation on the three yield FOMs (2,000 common
    Monte-Carlo corruptions). Numbers produced by figC1.py / cvar_study.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

# ---------- (a) inverse-engine ablation ----------
engines = ['Random\nsearch', 'Probes only\n(no gradient)',
           'Gradient only\n(no probes)', 'PARL-ID\n(composition)']
J = [0.0705, 0.0618, 0.0731, 0.0813]
cols = [C['gray'], C['sky'], C['orange'], C['blue']]

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8))
bars = ax[0].bar(engines, J, color=cols, width=0.62, zorder=3,
                 edgecolor='white', linewidth=0.6)
for b, v in zip(bars, J):
    ax[0].text(b.get_x() + b.get_width() / 2, v + 0.0008, f'{v:.4f}',
               ha='center', fontsize=7.2, zorder=4)
ax[0].annotate('', xy=(3, 0.0813), xytext=(3, 0.0731),
               arrowprops=dict(arrowstyle='->', color=C['green'], lw=1.4))
ax[0].text(3.12, 0.0770, '+11%', fontsize=8, color=C['green'], fontweight='bold')
ax[0].axhline(0.0731, color=C['orange'], lw=0.7, ls=':', zorder=2)
ax[0].set_ylabel('Best objective at 360 queries [$\\mu$m]')
ax[0].set_ylim(0.055, 0.088)
ax[0].minorticks_on()
ax[0].set_title('(a)  Inverse-engine ablation')
ax[0].tick_params(axis='x', labelsize=6.8)
ann_box(ax[0], 'neither phase suffices alone:\ncomposition is the mechanism',
        (0.03, 0.97), fontsize=6.6)

# ---------- (b) fabrication-objective ablation ----------
foms = ['mean $\\mu$', '$P_5$ floor', 'CVaR$_{5\\%}$']
nominal = [0.0795, 0.0758, 0.0744]
musig = [0.0805, 0.0767, 0.0753]
cvar = [0.0839, 0.0797, 0.0781]
x = np.arange(3); w = 0.26
ax[1].bar(x - w, nominal, w, color=C['gray'], label='Nominal $\\theta^{\\star}$',
          zorder=3, edgecolor='white', linewidth=0.5)
ax[1].bar(x, musig, w, color=C['orange'], label='$\\mu-\\beta\\sigma$ correction',
          zorder=3, edgecolor='white', linewidth=0.5)
ax[1].bar(x + w, cvar, w, color=C['blue'], label='CVaR$_{5\\%}$ correction',
          zorder=3, edgecolor='white', linewidth=0.5)
for xi, v, v0 in zip(x + w, cvar, nominal):
    ax[1].text(xi, v + 0.0007, f'+{(v / v0 - 1) * 100:.1f}%', ha='center',
               fontsize=7, color=C['blue'], fontweight='bold', zorder=4)
for xi, v, v0 in zip(x, musig, nominal):
    ax[1].text(xi, v + 0.0007, f'+{(v / v0 - 1) * 100:.1f}%', ha='center',
               fontsize=5.8, color='#8A5A00', zorder=4)
ax[1].set_xticks(x); ax[1].set_xticklabels(foms, fontsize=8)
ax[1].set_ylabel('Performance under corruption [$\\mu$m]')
ax[1].set_ylim(0.070, 0.089)
ax[1].minorticks_on()
ax[1].legend(fontsize=6.4, loc='upper left')
ax[1].set_title('(b)  Fabrication-objective ablation')
fig.tight_layout(); fig.savefig(OUT + 'Fig_ablation.svg', bbox_inches='tight')
print('Fig_ablation done')
