"""E4 (Reviewer 1, point 3): photonic benchmark with the ceviche-challenges
2 um x 2 um waveguide bend (40 x 40 design pixels, 1550 nm).
(a) Neural-adjoint gradient fidelity: an MLP surrogate of the transmission
    T = |S21|^2 over a 6 x 6 block parameterization is trained on FDFD data;
    its reverse-mode gradient is compared with the exact FDFD adjoint gradient
    (ceviche, autograd) at held-out designs.
(b) Fabrication-aware design with yield analysis: density filter + projection;
    erosion/dilation modeled by the projection threshold eta ~ N(0.5, 0.05^2)
    (clipped to [0.35, 0.65]); nominal, worst-of-three (eta = 0.4/0.5/0.6) and
    CVaR (worst 2 of 6 sampled eta per iteration) objectives, exact adjoint.
Usage: python e4_photonic.py data <part> <seconds> | grad <seconds> | fit |
       opt <kind> <seconds> | yield <seconds>"""
import sys, time, pickle, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import *
from ceviche_challenges import waveguide_bend

SPEC = waveguide_bend.prefabs.waveguide_bend_2umx2um_spec()
PRM = waveguide_bend.prefabs.waveguide_bend_sim_params()
MODEL = waveguide_bend.model.WaveguideBendModel(PRM, SPEC)
N = MODEL.design_variable_shape[0]
UP = onp.kron(onp.eye(6), onp.ones((7, 1)))[:N]           # (40,6) block upsampling


def T_of_density(rho):
    s, _ = MODEL.simulate(rho); return np.abs(s[0, 0, 1]) ** 2

def T_of_c(c):           # c: (36,)
    C = np.reshape(c, (6, 6)); return T_of_density(np.dot(np.dot(UP, C), UP.T))

# ---------------- (a) data, gradients, surrogate ----------------
def gen_data(part, budget, n=230):
    fn = RES / f'e4_data_p{part}.npz'; r = onp.random.default_rng(100 + part)
    Cs = r.random((n, 36)); done = []
    if fn.exists():
        done = list(onp.load(fn)['T'])
    t0 = time.time()
    while len(done) < n and time.time() - t0 < budget:
        done.append(float(T_of_c(Cs[len(done)])))
        if len(done) % 20 == 0: onp.savez(fn, C=Cs[:len(done)], T=onp.array(done))
    onp.savez(fn, C=Cs[:len(done)], T=onp.array(done)); return len(done)

def exact_grads(budget):
    fn = RES / 'e4_grads.npz'; d = onp.load(RES / 'e4_data_p1.npz'); C = d['C'][-30:]
    G = list(onp.load(fn)['G']) if fn.exists() else []
    g = grad(T_of_c); t0 = time.time()
    while len(G) < 30 and time.time() - t0 < budget:
        G.append(onp.asarray(g(C[len(G)])))
    onp.savez(fn, C=C[:len(G)], G=onp.array(G)); return len(G)

def fit_and_compare():
    d0, d1 = onp.load(RES / 'e4_data_p0.npz'), onp.load(RES / 'e4_data_p1.npz')
    Ctr = onp.r_[d0['C'], d1['C'][:-30]]; Ttr = onp.r_[d0['T'], d1['T'][:-30]]
    Cte, Tte = d1['C'][-30:], d1['T'][-30:]; G = onp.load(RES / 'e4_grads.npz')['G']
    from autograd.misc.flatten import flatten
    from scipy.optimize import minimize
    out = {}
    for ntr in [100, 200, len(Ctr)]:
        r = onp.random.default_rng(0); X = Ctr[:ntr] * 2 - 1; y = Ttr[:ntr]
        p = init_mlp([36, 64, 64, 1], r); ym, ys = y.mean(), y.std()
        def loss(p): return np.mean((mlp(p, X)[:, 0] - (y - ym) / ys) ** 2) + 1e-3 * sum(np.sum(W ** 2) for W, _ in p)
        fl, un = flatten(p); vg = value_and_grad(lambda q: loss(un(q)))
        o = minimize(lambda q: (lambda v, g: (float(v), onp.asarray(g)))(*vg(q)), fl, jac=True, method='L-BFGS-B', options=dict(maxiter=1500))
        p = un(o.x); f = lambda c: mlp(p, (c * 2 - 1)[None])[0, 0] * ys + ym
        pred = onp.array([f(c) for c in Cte]); gn = onp.array([onp.asarray(grad(f)(c)) for c in Cte])
        cos = onp.sum(gn * G, 1) / (onp.linalg.norm(gn, axis=1) * onp.linalg.norm(G, axis=1))
        rel = onp.linalg.norm(gn - G, axis=1) / onp.linalg.norm(G, axis=1)
        out[str(ntr)] = dict(R2_test=r2(Tte, pred), cos_median=float(onp.median(cos)), cos_q25=float(onp.percentile(cos, 25)),
                             cos_q75=float(onp.percentile(cos, 75)), relerr_median=float(onp.median(rel)), cos_all=cos.tolist())
        print(ntr, {k: round(v, 3) for k, v in out[str(ntr)].items() if k != 'cos_all'}, flush=True)
    out['T_train_mean'] = float(Ttr.mean()); out['T_train_sd'] = float(Ttr.std())
    save_json('e4a_fidelity.json', out)

# ---------------- (b) fabrication-aware design ----------------
xg = onp.arange(N); Gk = onp.exp(-0.5 * ((xg[:, None] - xg[None, :]) / 1.5) ** 2); Gk = Gk / Gk.sum(1, keepdims=True)
BETA = 8.0
def density(x, eta):
    xt = np.dot(np.dot(Gk, x), Gk.T)
    return (onp.tanh(BETA * eta) + np.tanh(BETA * (xt - eta))) / (onp.tanh(BETA * eta) + onp.tanh(BETA * (1 - eta)))

def objective(x, kind, etas):
    Ts = [T_of_density(density(x, e)) for e in etas]
    if kind == 'nominal': return -Ts[0]
    if kind == 'triple': return -smin(Ts)
    return -cvar_low(Ts)

def smin(Ts):            # exact minimum (sub-gradient)
    m = Ts[0]
    for t in Ts[1:]: m = np.minimum(m, t)
    return m

def cvar_low(Ts):        # mean of the worst 2 of 6
    from autograd.tracer import getval
    v = onp.array([float(getval(t)) for t in Ts]); idx = onp.argsort(v)[:2]
    return (Ts[idx[0]] + Ts[idx[1]]) / 2

def optimize(kind, budget, iters=30, lr=0.05):
    fn = RES / f'e4_opt_{kind}.pkl'
    if fn.exists():
        st = pickle.load(open(fn, 'rb'))
    else:
        st = dict(x=onp.full((N, N), 0.5), m=onp.zeros((N, N)), v=onp.zeros((N, N)), t=0, hist=[], rng=onp.random.default_rng(5))
    t0 = time.time()
    while st['t'] < iters and time.time() - t0 < budget:
        if kind == 'nominal': etas = [0.5]
        elif kind == 'triple': etas = [0.4, 0.5, 0.6]
        else: etas = list(onp.clip(st['rng'].normal(0.5, 0.05, 6), 0.35, 0.65))
        val, g = value_and_grad(objective)(st['x'], kind, etas); st['t'] += 1
        st['m'] = 0.9 * st['m'] + 0.1 * g; st['v'] = 0.999 * st['v'] + 0.001 * g * g
        st['x'] = onp.clip(st['x'] - lr * (st['m'] / (1 - 0.9 ** st['t'])) / (onp.sqrt(st['v'] / (1 - 0.999 ** st['t'])) + 1e-8), 0, 1)
        st['hist'].append(-float(val))
    pickle.dump(st, open(fn, 'wb')); return st

def yield_eval(budget):
    fn = RES / 'e4b_yield.json'
    res = load_json(fn.name) if fn.exists() else {}
    etas = onp.clip(onp.random.default_rng(77).normal(0.5, 0.05, 100), 0.35, 0.65); t0 = time.time()
    for kind in ['nominal', 'triple', 'cvar']:
        x = pickle.load(open(RES / f'e4_opt_{kind}.pkl', 'rb'))['x']
        r = res.get(kind, {'T': []})
        while len(r['T']) < 100 and time.time() - t0 < budget:
            r['T'].append(float(T_of_density(density(x, etas[len(r['T'])]))))
        if len(r['T']) == 100 and 'mean' not in r:
            T = onp.array(r['T']); k = 5
            r.update(nominal=float(T_of_density(density(x, 0.5))), mean=float(T.mean()), p5=float(onp.percentile(T, 5)),
                     cvar5=float(onp.sort(T)[:k].mean()), min=float(T.min()), yield_090=float(onp.mean(T >= 0.90)), yield_080=float(onp.mean(T >= 0.80)))
        res[kind] = r
    save_json(fn.name, res)
    return {k: {kk: (round(vv, 4) if not isinstance(vv, list) else len(vv)) for kk, vv in v.items()} for k, v in res.items()}

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'data': print('data', gen_data(int(sys.argv[2]), float(sys.argv[3])))
    elif mode == 'grad': print('grads', exact_grads(float(sys.argv[2])))
    elif mode == 'fit': fit_and_compare()
    elif mode == 'opt': st = optimize(sys.argv[2], float(sys.argv[3])); print(sys.argv[2], st['t'], [round(h, 3) for h in st['hist'][-3:]])
    elif mode == 'yield': print(yield_eval(float(sys.argv[2])))
