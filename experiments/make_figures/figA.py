"""Fig_pinn_train.svg — surrogate convergence + held-out-combination parity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

rng = np.random.default_rng(0)
df = load()
X = df[['t_e', 't_np', 't_ox', 't_w', 'frequency_MHz']].values
y = df['displacement_um'].values[:, None]
Xm, Xs = X.mean(0), X.std(0); ym, ys = y.mean(), y.std()
Xn, yn = (X - Xm) / Xs, (y - ym) / ys
combos = df.groupby(['t_e', 't_np', 't_ox', 't_w']).ngroup().values
uc = np.unique(combos); rng.shuffle(uc)
test_c = set(uc[:74])                      # 20% of combos held out entirely
te = np.array([c in test_c for c in combos]); tr = ~te


def mlp_train(Xtr, ytr, Xval, yval, epochs=25, h=48, lr=2e-3, seed=1):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 0.4, (Xtr.shape[1], h)); b1 = np.zeros(h)
    W2 = r.normal(0, 0.4, (h, h)); b2 = np.zeros(h)
    W3 = r.normal(0, 0.4, (h, 1)); b3 = np.zeros(1)
    ps = [W1, b1, W2, b2, W3, b3]; mom = [np.zeros_like(q) for q in ps]
    trl, vll = [], []
    n = len(Xtr); bs = 512
    for ep in range(epochs):
        idx = r.permutation(n)
        for s in range(0, n, bs):
            i = idx[s:s + bs]; xb, yb = Xtr[i], ytr[i]
            a1 = np.tanh(xb @ W1 + b1); a2 = np.tanh(a1 @ W2 + b2)
            out = a2 @ W3 + b3
            d3 = 2 * (out - yb) / len(xb)
            gW3 = a2.T @ d3; gb3 = d3.sum(0)
            d2 = (d3 @ W3.T) * (1 - a2 ** 2); gW2 = a1.T @ d2; gb2 = d2.sum(0)
            d1 = (d2 @ W2.T) * (1 - a1 ** 2); gW1 = xb.T @ d1; gb1 = d1.sum(0)
            for q, g, m in zip(ps, [gW1, gb1, gW2, gb2, gW3, gb3], mom):
                m *= 0.9; m += lr * g; q -= m
        mse = lambda Xa, ya: float(np.mean(
            (np.tanh(np.tanh(Xa @ W1 + b1) @ W2 + b2) @ W3 + b3 - ya) ** 2))
        trl.append(mse(Xtr, ytr)); vll.append(mse(Xval, yval))
    pred = lambda Xa: np.tanh(np.tanh(Xa @ W1 + b1) @ W2 + b2) @ W3 + b3
    return trl, vll, pred


fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7))
fracs = [0.10, 0.50, 1.00]; cols = [C['gray'], C['orange'], C['blue']]
pred_full = None
for fr, c in zip(fracs, cols):
    ntr = int(tr.sum() * fr)
    sel = np.where(tr)[0]; rng.shuffle(sel); sel = sel[:ntr]
    trl, vll, pred = mlp_train(Xn[sel], yn[sel], Xn[te], yn[te])
    ax[0].semilogy(trl, color=c, ls='--', lw=1.0, alpha=0.75)
    ax[0].semilogy(vll, color=c, lw=1.7, label=f'{int(fr * 100)}% of database')
    if fr == 1.0:
        pred_full = pred
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('MSE (normalized)')
ax[0].legend(loc='upper right', title='held-out combinations', title_fontsize=7)
ax[0].set_title('(a) Surrogate convergence')
ann_box(ax[0], 'solid: validation\ndashed: training', (0.03, 0.22))

yp = (pred_full(Xn[te]) * ys + ym).ravel(); ya = y[te].ravel()
r2 = 1 - np.sum((ya - yp) ** 2) / np.sum((ya - ya.mean()) ** 2)
mae = float(np.abs(ya - yp).mean())
hb = ax[1].hexbin(ya, yp, gridsize=45, cmap='Blues', mincnt=1, linewidths=0)
lims = [min(ya.min(), yp.min()), max(ya.max(), yp.max())]
ax[1].plot(lims, lims, color=C['red'], lw=1.0)
cb = plt.colorbar(hb, ax=ax[1], shrink=0.9, pad=0.02)
cb.set_label('samples', fontsize=7); cb.ax.tick_params(labelsize=6.5)
ax[1].set_xlabel('FEM displacement [$\\mu$m]')
ax[1].set_ylabel('Predicted [$\\mu$m]')
ax[1].set_title('(b) Held-out parity')
ann_box(ax[1], f'$R^2$ = {r2:.3f}\nMAE = {mae * 1e3:.1f} nm', (0.04, 0.96))
fig.tight_layout(); fig.savefig(OUT + 'Fig_pinn_train.svg', bbox_inches='tight')
print('Fig_pinn_train done, R2 =', round(r2, 4))
np.save(str(Path(__file__).resolve().parent / 'r2.npy'), r2)
