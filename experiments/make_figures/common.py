"""Shared style and data helpers for the paper's figure pipeline.

Professional defaults: Okabe-Ito colorblind-safe palette, no top/right spines,
subtle grid, bold left-aligned panel titles, annotation boxes.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'svg.fonttype': 'none', 'font.size': 8.5, 'font.family': 'sans-serif',
    'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False, 'legend.fontsize': 7.5,
    'lines.linewidth': 1.5, 'figure.dpi': 110,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})

# Okabe-Ito colorblind-safe palette
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
     'red': '#D55E00', 'purple': '#CC79A7', 'gray': '#7F7F7F', 'sky': '#56B4E9'}

from pathlib import Path
_HERE = Path(__file__).resolve().parent
CSV = str(_HERE.parent / 'augmented_reshaped_dataset.csv')   # the released FEM database
OUT = str(_HERE.parent.parent.parent / 'latex') + '/'        # SVGs written into latex/


def load():
    return pd.read_csv(CSV)


def combo_objective(df, f_tgt, halfwin=0.5):
    """Per-combo objective: max displacement within f_tgt +/- halfwin MHz."""
    rows = []
    for key, g in df.groupby(['t_e', 't_np', 't_ox', 't_w']):
        m = (g.frequency_MHz >= f_tgt - halfwin) & (g.frequency_MHz <= f_tgt + halfwin)
        val = g.loc[m, 'displacement_um'].max() if m.any() else g.displacement_um.min()
        rows.append(list(key) + [val])
    arr = np.array(rows)
    return arr[:, :4], arr[:, 4]


def combo_objective_band(df, f_tgt, halfwin=0.5):
    """Bandwidth-aware objective (industrial practice, cf. TSMC dual-layer
    grating couplers: co-optimize loss AND bandwidth): the *mean* displacement
    across the band f_tgt +/- halfwin."""
    rows = []
    for key, g in df.groupby(['t_e', 't_np', 't_ox', 't_w']):
        m = (g.frequency_MHz >= f_tgt - halfwin) & (g.frequency_MHz <= f_tgt + halfwin)
        val = g.loc[m, 'displacement_um'].mean() if m.any() else g.displacement_um.min()
        rows.append(list(key) + [val])
    arr = np.array(rows)
    return arr[:, :4], arr[:, 4]


def ann_box(ax, text, xy, fontsize=7.5, fc='#FFFFFF'):
    """Annotation box in axes-fraction coordinates (top-left anchored)."""
    ax.annotate(text, xy=xy, xycoords='axes fraction', fontsize=fontsize,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.35', fc=fc, ec='#B0BEC5', lw=0.6))
