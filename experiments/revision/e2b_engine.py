"""E2b: inverse-engine comparison at an equal budget of 200 surrogate queries
(one query = one surrogate evaluation; one gradient iteration = one query plus
its reverse-mode pass).  Success = |W/W_tgt - 1| < 0.5 %."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_design as E2
from rev_common import *
WT = onp.linspace(0.018, 0.045, 20)
out = {}
for method in ['random', 'gradient', 'seeded']:
    rows = []
    curves = []
    for s in range(3):
        rng = onp.random.default_rng(1000 + s)
        for wt in WT:
            th, q, ok, c = E2.engine(method, wt, 200, rng)
            fin = float(abs(E2.W(th[None])[0] / wt - 1))
            rows.append((ok, q, fin)); c = c + [c[-1]] * (200 - len(c)); curves.append(c[:200])
    ok = onp.array([r[0] for r in rows]); q = onp.array([r[1] for r in rows]); fin = onp.array([r[2] for r in rows])
    out[method] = dict(success_rate=float(ok.mean()), median_queries_success=float(onp.median(q[ok])) if ok.any() else None,
                       mean_queries=float(q.mean()), median_final_dev=float(onp.median(fin)),
                       curve_median=onp.median(onp.array(curves), 0).tolist(),
                       curve_q25=onp.percentile(onp.array(curves), 25, 0).tolist(),
                       curve_q75=onp.percentile(onp.array(curves), 75, 0).tolist())
    print(method, {k: v for k, v in out[method].items() if not k.startswith('curve')}, flush=True)
save_json('e2b_engine.json', out)
