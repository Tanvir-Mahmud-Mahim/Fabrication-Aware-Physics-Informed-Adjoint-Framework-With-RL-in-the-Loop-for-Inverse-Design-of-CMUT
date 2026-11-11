"""E2d: robust correction of the nominal inverse-design solution (kappa = 1).
Search: sample-average objectives on K = 400 common draws; evaluation: 4000
fresh common draws.  Reports the nominal deviation at every design."""
import sys, warnings, time; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_design as E2
from rev_common import *
WT = [0.020, 0.025, 0.030, 0.035, 0.040, 0.045]
try:
    out = load_json('e2d_robust.json')
except FileNotFoundError:
    out = {}
t0 = time.time()
xs = onp.random.default_rng(7).normal(size=(400, 4)); xe = onp.random.default_rng(8).normal(size=(4000, 4))
for wt in WT:
    k = f'{wt:.3f}'
    if k in out: continue
    if time.time() - t0 > 30: break
    th0, q, ok, _ = E2.engine('gradient', wt, 200, onp.random.default_rng(int(wt * 1e4)))
    rec = {'theta_nominal': th0.tolist(), 'nominal': E2.stats(th0, wt, xe)}
    for kind in ['mv', 'cvar']:
        th, nf = E2.robust_opt(th0, wt, kind, xs)
        rec['theta_' + kind] = th.tolist(); rec[kind] = E2.stats(th, wt, xe); rec[kind]['nfev'] = nf
    out[k] = rec; save_json('e2d_robust.json', out)
    print(k, {m: {kk: round(v, 4) for kk, v in rec[m].items()} for m in ['nominal', 'mv', 'cvar']}, flush=True)
print('done', len(out))
