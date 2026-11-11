"""E3b: specification-conditioned correction policy trained with neural-adjoint
gradients (amortized optimization through the frozen surrogate), and zero-shot
evaluation at held-out specifications against (i) the nominal design,
(ii) a static correction vector transplanted from the reference specification
(W_tgt = 0.030 um, kappa = 1.0), (iii) the per-specification SAA optimum
(oracle), and (iv) the best model-free RL checkpoint (E3).
Policy: GCN encoder of E3 + per-parameter head; output du in [-0.5, 0.5]^4."""
import sys, time, pickle, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_rl as R
import e2_design as E2
from rev_common import *
from autograd.misc.flatten import flatten

TRAIN = [(w, k) for w in R.W_TRAIN for k in R.K_TRAIN]
XS = onp.random.default_rng(7).normal(size=(400, 4))
XE = onp.random.default_rng(8).normal(size=(4000, 4))

def obs(wt, k):
    us = R.u_of(R.TSTAR[float(wt)]); return onp.r_[us, us, (wt - 0.02) / 0.02, (k - 1.0) / 0.4]

def du_of(A, O):
    mu, _ = R.actor_out(A, O, True); return 0.5 * np.tanh(mu)

def cvar_ru(th, wt, k, t):   # Rockafellar-Uryasev SAA, minimized jointly over t
    dv = E2.dev(th, XS, wt, k); return t + np.mean(np.maximum(dv - t, 0)) / 0.05

STEPS = 200

def make_loss():
    O = onp.array([obs(w, k) for w, k in TRAIN]); US = O[:, :4]
    def loss(P):
        A_, T_ = P
        du = du_of(A_, O); tot = 0.0
        for i, (w, k) in enumerate(TRAIN):
            th = E2.ILO + (E2.IHI - E2.ILO) * np.clip(US[i] + du[i], 0, 1)
            tot = tot + cvar_ru(th, w, k, T_[i])
        return tot / len(TRAIN)
    return loss

def train(seed, budget):
    fn = RES / f'e3b_policy_s{seed}.pkl'
    if fn.exists():
        st = pickle.load(open(fn, 'rb'))
    else:
        r = onp.random.default_rng(seed); P = [R.init_actor(r, True), onp.full(len(TRAIN), 0.1)]
        st = dict(P=P, opt=R.Adam(P, 3e-3), step=0, hist=[])
    loss = make_loss(); g = grad(loss); t0 = time.time()
    while st['step'] < STEPS and time.time() - t0 < budget:
        st['P'] = st['opt'].step(g(st['P'])); st['step'] += 1
        if st['step'] % 20 == 0: st['hist'].append((st['step'], float(loss(st['P']))))
    pickle.dump(st, open(fn, 'wb'))
    return st

def evaluate(designs_by_spec, specs):
    return {f'{w}|{k}': E2.stats(designs_by_spec[(w, k)], w, XE, k) for w, k in specs}

if __name__ == '__main__':
    st = train(int(sys.argv[1]), float(sys.argv[2])); print('seed', sys.argv[1], 'step', st['step'], st['hist'][-1:] , flush=True)
