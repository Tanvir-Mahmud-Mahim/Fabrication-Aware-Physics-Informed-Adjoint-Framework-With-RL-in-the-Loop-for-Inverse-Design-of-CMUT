"""E0: audit of the released CMUT database and of the original pipeline.
(i) augmentation copies, (ii) frequency flatness of the profiles,
(iii) block-wise smoothness pattern (suspected input/output misalignment in
the t_e = 0.24 um block), (iv) held-out-design GP accuracy per block,
(v) original RBF objective at the original theta* and theta_rob."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import *
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck

b, d, df = load_base()
out = {}
allc = df.groupby(PCOLS).ngroup()
out['n_entries'] = int(allc.nunique()); out['n_fem_designs'] = int(len(d))
# augmentation copies: nearest base design and output difference
from scipy.spatial.distance import cdist
s = df.groupby(PCOLS)['displacement_um'].mean().reset_index()
X = s[PCOLS].values; lab = cdist(X / X.std(0), d / X.std(0)).argmin(1)
base_m = b.groupby('design')['displacement_um'].mean().values
out['aug_max_rel_input_perturbation'] = (onp.abs(X - d[lab]) / d[lab]).max(0).round(4).tolist()
out['aug_max_rel_output_difference'] = float(onp.max(onp.abs(s.displacement_um.values - base_m[lab]) / base_m[lab]))
# frequency flatness
P = onp.stack([g.sort_values('frequency_MHz')['displacement_um'].values for _, g in b.groupby('design')])
cv = P.std(1) / P.mean(1)
out['profile_cv_median'] = float(onp.median(cv)); out['profile_cv_max'] = float(cv.max())
out['profile_range_rel_median'] = float(onp.median((P.max(1) - P.min(1)) / P.mean(1)))
# block-wise jump pattern in sorted order
m = P.mean(1); order = onp.lexsort((d[:, 3], d[:, 2], d[:, 1], d[:, 0]))
mo, do = m[order], d[order]
jumps = onp.where(onp.abs(onp.diff(mo)) / mo[:-1] > 0.10)[0]
blk = do[:-1, 0][jumps]
out['jump_positions_te024'] = jumps[blk < 0.3].tolist(); out['jump_positions_te044_064'] = jumps[blk >= 0.3].tolist()
out['n_tw_levels_te024'] = int(len(onp.unique(d[d[:, 0] < 0.3, 3]))); out['n_tw_levels_te044_064'] = int(len(onp.unique(d[d[:, 0] >= 0.3, 3])))
# GP held-out-design accuracy per block
lo, hi = d.min(0), d.max(0); Xn = (d - lo) / (hi - lo)
def gp_cv(idx):
    yt, yp = [], []
    for te in group_folds(len(idx), 5, 0):
        tr = onp.setdiff1d(onp.arange(len(idx)), te)
        g = GPR(Ck(1.0) * RBF([0.5] * 4) + WhiteKernel(1e-2), normalize_y=True, n_restarts_optimizer=2, random_state=0)
        g.fit(Xn[idx][tr], onp.log(m[idx][tr])); yp += list(onp.exp(g.predict(Xn[idx][te]))); yt += list(m[idx][te])
    yt, yp = onp.array(yt), onp.array(yp); return dict(R2=r2(yt, yp), MAE_nm=float(onp.abs(yt - yp).mean() * 1e3))
out['gp_cv'] = {'all123': gp_cv(onp.arange(len(d))), 'te044_064_block54': gp_cv(onp.where(d[:, 0] >= 0.3)[0]),
                'te024_block69': gp_cv(onp.where(d[:, 0] < 0.3)[0])}
# original pipeline: thin-plate RBF on all 369 entries, objective = max W in 4.3 +/- 0.5 MHz
rows = []
for key, g in df.groupby(PCOLS):
    mm = (g.frequency_MHz >= 3.8) & (g.frequency_MHz <= 4.8); rows.append(list(key) + [g.loc[mm, 'displacement_um'].max()])
A = onp.array(rows); TH, J = A[:, :4], A[:, 4]; tlo, thi = TH.min(0), TH.max(0)
rbf = RBFInterpolator((TH - tlo) / (thi - tlo), J, kernel='thin_plate_spline', smoothing=1e-6)
f = lambda t: float(rbf(((onp.array(t) - tlo) / (thi - tlo))[None])[0])
th_star = [0.35210798, 0.97167677, 0.96616655, 3.01779175]; th_rob = [0.27394401, 0.97167677, 0.96616655, 2.99268422]
out['orig_nominal_J_theta_star'] = f(th_star); out['orig_nominal_J_theta_rob'] = f(th_rob)
out['orig_best_entry_J'] = float(J.max()); out['orig_best_entry'] = TH[J.argmax()].round(4).tolist()
out['orig_te_levels_fem'] = onp.unique(d[:, 0]).tolist()
save_json('e0_audit.json', out)
print(json.dumps(out, indent=0))
