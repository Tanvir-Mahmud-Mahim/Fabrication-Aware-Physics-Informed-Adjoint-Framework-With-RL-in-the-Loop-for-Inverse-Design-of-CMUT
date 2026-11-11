"""E2c: original objective (maximize band-averaged displacement) on the revised
surrogate, equal budget of 200 queries, 10 seeds per engine."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_design as E2
from rev_common import *
negW = lambda th: -E2.W(th[None])[0]
gW = grad(negW)
def refine(u, iters, lr=0.03):
    v = onp.zeros(4); h = []
    for t in range(iters):
        th = E2.to_box(u); h.append(float(E2.W(th[None])[0]))
        g = onp.asarray(gW(th)) * (E2.IHI - E2.ILO)
        v = 0.7 * v - lr * 0.97 ** t * g / (onp.abs(g).max() + 1e-12); u = onp.clip(u + v, 0, 1)
    return h
res = {}
for m in ['random', 'gradient', 'seeded']:
    best = []
    for s in range(10):
        r = onp.random.default_rng(50 + s)
        if m == 'random':
            best.append(float(onp.max(onp.asarray(E2.W(E2.to_box(r.random((200, 4))))))))
        elif m == 'gradient':
            best.append(max(max(refine(r.random(4), 50)) for _ in range(4)))
        else:
            U = r.random((100, 4)); w = onp.asarray(E2.W(E2.to_box(U)))
            best.append(max([w.max()] + [max(refine(U[i], 25)) for i in onp.argsort(w)[-4:]]))
    res[m] = dict(mean=float(onp.mean(best)), sd=float(onp.std(best)), max=float(onp.max(best)))
U = onp.random.default_rng(9).random((200000, 4)); w = onp.asarray(E2.W(E2.to_box(U)))
res['dense_200k_max'] = float(w.max()); res['argmax'] = E2.to_box(U[w.argmax()]).round(3).tolist()
res['fem_best'] = float(E2.E1.Wbar.max())
save_json('e2c_max.json', res); print(res)
