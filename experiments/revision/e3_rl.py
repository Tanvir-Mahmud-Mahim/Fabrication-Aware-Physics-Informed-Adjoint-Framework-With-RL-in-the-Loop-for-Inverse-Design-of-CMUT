"""E3 (Reviewer 1, point 2): learned, specification-conditioned correction
policy.  Specification s = (W_tgt, kappa).  Agents: GCN-SAC + PER (full),
GCN-SAC with uniform replay, MLP-SAC + PER, and GCN-PPO; 3 seeds each.
Episode: start at the nominal inverse-design solution theta*(W_tgt), H = 3
bounded corrections, reward = decrease of the CVaR_5% deviation (K = 400 draws
fixed per episode, scaled by 1/0.01).  Train specifications:
W_tgt in {0.020, 0.0225, ..., 0.040} um x kappa in {0.6, 1.0, 1.4}; held-out
test specifications: W_tgt in {0.02125, 0.02875, 0.03625} x kappa in {0.8, 1.2}.
Resumable: python e3_rl.py <agent> <seed> <seconds>."""
import sys, time, pickle, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_design as E2
from rev_common import *
from autograd.misc.flatten import flatten

H, STEP, ALPHA, KR = 1, 0.5, 0.05, 400
W_TRAIN = list(onp.round(onp.arange(0.020, 0.04001, 0.0025), 5)); K_TRAIN = [0.6, 1.0, 1.4]
W_TEST = [0.02125, 0.02875, 0.03625]; K_TEST = [0.8, 1.2]
TOTAL, WARM, EVAL_EVERY = 2000, 200, 250
TS_FILE = RES / 'e3_theta_star.json'


def theta_star_table():
    try:
        return {float(k): onp.array(v) for k, v in load_json(TS_FILE.name).items()}
    except FileNotFoundError:
        tab = {}
        for wt in W_TRAIN + W_TEST:
            th, q, ok, _ = E2.engine('gradient', wt, 200, onp.random.default_rng(int(wt * 1e5)))
            tab[float(wt)] = th
        save_json(TS_FILE.name, {str(k): v.tolist() for k, v in tab.items()})
        return tab

TSTAR = theta_star_table()
def u_of(th): return (onp.asarray(th) - E2.ILO) / (E2.IHI - E2.ILO)


class Env:
    def __init__(self, rng): self.rng = rng
    def reset(self, wt, kappa, xi=None):
        self.wt, self.k = wt, kappa
        self.xi = self.rng.normal(size=(KR, 4)) if xi is None else xi
        self.us = u_of(TSTAR[float(wt)]); self.u = self.us.copy(); self.t = 0
        self.score = self._score(self.u); self.s0 = self.score
        return self.obs()
    def _score(self, u):
        return -E2.cvar_upper(onp.asarray(E2.dev(E2.to_box(u), self.xi, self.wt, self.k)), ALPHA)
    def obs(self):
        return onp.r_[self.u, self.us, (self.wt - 0.02) / 0.02, (self.k - 1.0) / 0.4]
    def step(self, a):
        self.u = onp.clip(self.u + STEP * onp.clip(a, -1, 1), 0, 1)
        s = self._score(self.u); r = (s - self.score) / 0.01; self.score = s; self.t += 1
        return self.obs(), r, self.t >= H


# ---------------- encoders ----------------
A_ADJ = onp.eye(5)
for i, j in [(0, 1), (2, 3), (1, 2)] + [(4, k) for k in range(4)]:
    A_ADJ[i, j] = A_ADJ[j, i] = 1
dg = A_ADJ.sum(1); A_HAT = A_ADJ / onp.sqrt(onp.outer(dg, dg))

def nodes(o):   # o: (B,10) -> (B,5,11)
    B = o.shape[0]; u, us, w, k = o[:, :4], o[:, 4:8], o[:, 8:9], o[:, 9:10]
    pn = onp.stack([u, us, u - us, onp.zeros_like(u)], -1)             # (B,4,4)
    sn = onp.concatenate([onp.zeros((B, 1, 3)), onp.ones((B, 1, 1))], -1)
    x = onp.concatenate([pn, sn], 1)
    ctx = onp.repeat(onp.concatenate([w, k], 1)[:, None, :], 5, 1)
    return onp.concatenate([x, ctx, onp.broadcast_to(onp.eye(5), (B, 5, 5))], -1)

def gcn(p, o):
    h = nodes(o)
    for W, b in p:
        h = np.maximum(np.matmul(A_HAT, np.matmul(h, W)) + b, 0)
    return h                                                           # (B,5,32)

def init_gcn(r): return [(r.normal(0, onp.sqrt(1 / 11), (11, 32)), onp.zeros(32)), (r.normal(0, onp.sqrt(1 / 32), (32, 32)), onp.zeros(32))]

def actor_out(P, o, graph):
    if graph:
        h = gcn(P['enc'], o); g = np.mean(h, 1, keepdims=True)
        z = np.concatenate([h[:, :4, :], np.repeat(g, 4, 1)], -1)    # (B,4,64)
        out = mlp(P['head'], z)                                        # (B,4,2)
        return out[:, :, 0], np.clip(out[:, :, 1], -4, 0.5)
    out = mlp(P['head'], o); return out[:, :4], np.clip(out[:, 4:], -4, 0.5)

def embed(P, o, graph):
    return np.mean(gcn(P['enc'], o), 1) if graph else o

def init_actor(r, graph):
    return {'enc': init_gcn(r), 'head': init_mlp([64, 32, 2], r)} if graph else {'enc': [], 'head': init_mlp([10, 64, 64, 8], r)}

def init_critic(r, graph):
    return {'enc': init_gcn(r) if graph else [], 'q': init_mlp([(32 if graph else 10) + 4, 64, 64, 1], r)}

def qval(C, o, a, graph):
    return mlp(C['q'], np.concatenate([embed(C, o, graph), a], 1))[:, 0]


class Adam:
    def __init__(self, P, lr):
        self.tmpl = P; self.f, self.un = flatten(P); self.m = onp.zeros_like(self.f); self.v = onp.zeros_like(self.f); self.t = 0; self.lr = lr
    def __getstate__(self):
        d = dict(self.__dict__); d.pop('un', None); return d
    def __setstate__(self, d):
        self.__dict__.update(d); self.un = flatten(self.tmpl)[1]
    def step(self, g):
        g = flatten(g)[0]; self.t += 1
        self.m = 0.9 * self.m + 0.1 * g; self.v = 0.999 * self.v + 0.001 * g * g
        self.f = self.f - self.lr * (self.m / (1 - 0.9 ** self.t)) / (onp.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)
        return self.un(self.f)


# ---------------- SAC ----------------
GAMMA, TAU, BS = 0.9, 0.01, 64

def sac_sample(A, o, graph, eps):
    mu, ls = actor_out(A, o, graph); s = np.exp(ls); pre = mu + s * eps; a = np.tanh(pre)
    logp = np.sum(-0.5 * eps ** 2 - ls - 0.5 * onp.log(2 * onp.pi) - np.log(1 - a ** 2 + 1e-6), 1)
    return a, logp

def sac_init(graph, per, seed):
    r = onp.random.default_rng(seed)
    S = dict(kind='sac', graph=graph, per=per, rng=r, A=init_actor(r, graph), C1=init_critic(r, graph), C2=init_critic(r, graph),
             logalpha=onp.array([-3.0]), buf=[], prio=[], steps=0, log=[], ep_ret=0.0)
    S['T1'] = pickle.loads(pickle.dumps(S['C1'])); S['T2'] = pickle.loads(pickle.dumps(S['C2']))
    S['oA'] = Adam(S['A'], 1e-3); S['o1'] = Adam(S['C1'], 1e-3); S['o2'] = Adam(S['C2'], 1e-3); S['oa'] = Adam(S['logalpha'], 1e-3)
    return S

def sac_update(S):
    g = S['graph']; r = S['rng']; n = len(S['buf'])
    if S['per']:
        pr = onp.array(S['prio']) ** 0.6; P = pr / pr.sum(); idx = r.choice(n, BS, p=P)
        beta = min(1.0, 0.4 + 0.6 * S['steps'] / TOTAL); w = (n * P[idx]) ** (-beta); w = w / w.max()
    else:
        idx = r.integers(0, n, BS); w = onp.ones(BS)
    o, a, rew, o2, d = [onp.array([S['buf'][i][j] for i in idx]) for j in range(5)]
    alpha = float(onp.exp(S['logalpha'][0]))
    a2, lp2 = sac_sample(S['A'], o2, g, r.normal(size=(BS, 4)))
    tq = onp.minimum(qval(S['T1'], o2, a2, g), qval(S['T2'], o2, a2, g)) - alpha * onp.asarray(lp2)
    y = rew + GAMMA * (1 - d) * tq
    td = None
    for k in ('1', '2'):
        def lq(C): return np.mean(w * (qval(C, o, a, g) - y) ** 2)
        if k == '1':
            td = onp.abs(onp.asarray(qval(S['C1'], o, a, g)) - y)
        S['C' + k] = S['o' + k].step(grad(lq)(S['C' + k]))
    eps = r.normal(size=(BS, 4))
    def la(A):
        an, lp = sac_sample(A, o, g, eps)
        return np.mean(alpha * lp - np.minimum(qval(S['C1'], o, an, g), qval(S['C2'], o, an, g)))
    S['A'] = S['oA'].step(grad(la)(S['A']))
    _, lp = sac_sample(S['A'], o, g, eps); lp = onp.asarray(lp)
    S['logalpha'] = S['oa'].step(grad(lambda L: -np.mean(L[0] * (lp - 4.0)))(S['logalpha']))
    for k in ('1', '2'):
        fc, unc = flatten(S['C' + k]); ft, _ = flatten(S['T' + k]); S['T' + k] = unc((1 - TAU) * ft + TAU * fc)
    if S['per']:
        for i, t in zip(idx, td): S['prio'][i] = float(t) + 1e-3

def sac_act(S, o, det=False):
    mu, ls = actor_out(S['A'], o[None], S['graph']); mu, ls = onp.asarray(mu)[0], onp.asarray(ls)[0]
    return onp.tanh(mu) if det else onp.tanh(mu + onp.exp(ls) * S['rng'].normal(size=4))


# ---------------- PPO ----------------
def ppo_init(seed):
    r = onp.random.default_rng(seed)
    S = dict(kind='ppo', graph=True, rng=r, A=init_actor(r, True), V=init_critic(r, True), steps=0, log=[], roll=[])
    S['oA'] = Adam(S['A'], 3e-4); S['oV'] = Adam(S['V'], 1e-3)
    return S

def ppo_logp(A, o, a):
    mu, ls = actor_out(A, o, True)
    return np.sum(-0.5 * ((a - mu) / np.exp(ls)) ** 2 - ls - 0.5 * onp.log(2 * onp.pi), 1)

def vval(V, o): return mlp(V['q'], np.concatenate([embed(V, o, True), onp.zeros((o.shape[0], 4))], 1))[:, 0]

def ppo_act(S, o, det=False):
    mu, ls = actor_out(S['A'], o[None], True); mu, ls = onp.asarray(mu)[0], onp.asarray(ls)[0]
    return mu if det else mu + onp.exp(ls) * S['rng'].normal(size=4)

def ppo_update(S):
    R = S['roll']; o = onp.array([x[0] for x in R]); a = onp.array([x[1] for x in R]); rew = onp.array([x[2] for x in R])
    d = onp.array([x[4] for x in R]); lp_old = onp.asarray(ppo_logp(S['A'], o, a)); v = onp.asarray(vval(S['V'], o))
    o2 = onp.array([x[3] for x in R]); v2 = onp.asarray(vval(S['V'], o2))
    adv = onp.zeros(len(R)); last = 0.0
    for t in range(len(R) - 1, -1, -1):
        delta = rew[t] + GAMMA * (1 - d[t]) * v2[t] - v[t]; last = delta + GAMMA * 0.95 * (1 - d[t]) * last; adv[t] = last
    ret = adv + v; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for ep in range(10):
        for mb in onp.array_split(S['rng'].permutation(len(R)), 4):
            def lpi(A):
                ratio = np.exp(ppo_logp(A, o[mb], a[mb]) - lp_old[mb])
                return -np.mean(np.minimum(ratio * adv[mb], np.clip(ratio, 0.8, 1.2) * adv[mb]))
            S['A'] = S['oA'].step(grad(lpi)(S['A']))
            S['V'] = S['oV'].step(grad(lambda V: np.mean((vval(V, o[mb]) - ret[mb]) ** 2))(S['V']))
    S['roll'] = []


# ---------------- evaluation ----------------
XI_EVAL = onp.random.default_rng(4242).normal(size=(2000, 4))
def policy_eval(S, specs):
    env = Env(onp.random.default_rng(0)); gains = []; finals = []
    for wt, k in specs:
        o = env.reset(wt, k, XI_EVAL)
        for t in range(H):
            a = sac_act(S, o, True) if S['kind'] == 'sac' else onp.clip(ppo_act(S, o, True), -1, 1)
            o, _, _ = env.step(a)
        gains.append(env.score - env.s0); finals.append(E2.to_box(env.u).tolist())
    return float(onp.mean(gains)), finals

TEST_SPECS = [(w, k) for w in W_TEST for k in K_TEST]


def run(agent, seed, budget):
    fn = RES / f'e3_{agent}_s{seed}.pkl'
    if fn.exists():
        S = pickle.load(open(fn, 'rb'))
    else:
        S = ppo_init(seed) if agent == 'ppo_gcn' else sac_init(agent.endswith('gcn_per') or agent == 'sac_gcn_uniform', agent.endswith('per'), seed)
        S['env'] = Env(onp.random.default_rng(seed + 500)); S['o'] = None
    t0 = time.time(); env = S['env']; r = S['rng']
    while S['steps'] < TOTAL and time.time() - t0 < budget:
        if S['o'] is None:
            S['o'] = env.reset(W_TRAIN[r.integers(len(W_TRAIN))], K_TRAIN[r.integers(len(K_TRAIN))])
        o = S['o']
        if S['kind'] == 'sac':
            a = r.uniform(-1, 1, 4) if S['steps'] < WARM else sac_act(S, o)
        else:
            a = ppo_act(S, o)
        o2, rew, done = env.step(a)
        if S['kind'] == 'sac':
            S['buf'].append((o, a, rew, o2, float(done))); S['prio'].append(max(S['prio'], default=1.0))
            if S['steps'] >= WARM: sac_update(S)
        else:
            S['roll'].append((o, a, rew, o2, float(done)))
            if len(S['roll']) >= 240: ppo_update(S)
        S['o'] = None if done else o2; S['steps'] += 1
        if S['steps'] % EVAL_EVERY == 0:
            g, _ = policy_eval(S, TEST_SPECS); S['log'].append((S['steps'], g))
    pickle.dump(S, open(fn, 'wb'))
    return S


if __name__ == '__main__':
    S = run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
    print(sys.argv[1], sys.argv[2], 'steps', S['steps'], 'log', [(s, round(g, 4)) for s, g in S['log']], flush=True)
