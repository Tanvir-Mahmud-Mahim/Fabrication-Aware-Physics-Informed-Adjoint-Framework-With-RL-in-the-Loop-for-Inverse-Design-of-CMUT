"""E3c: zero-shot evaluation at the six held-out specifications (4000 fresh
common draws at the specification's kappa)."""
import sys, pickle, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_rl as R, e3b_policy as P, e2_design as E2
from rev_common import *
import __main__; __main__.Adam = R.Adam; __main__.Env = R.Env
XE = P.XE; XS = P.XS
specs = R.TEST_SPECS
res = {'specs': [list(s) for s in specs]}
# reference static correction (W_tgt = 0.030, kappa = 1.0)
th_ref, _ = E2.robust_opt(R.TSTAR[0.030], 0.030, 'cvar', XS, kappa=1.0)
du_static = R.u_of(th_ref) - R.u_of(R.TSTAR[0.030])
def box(u): return E2.ILO + (E2.IHI - E2.ILO) * onp.clip(u, 0, 1)
methods = {}
methods['nominal'] = {s: R.TSTAR[s[0]] for s in specs}
methods['static'] = {s: box(R.u_of(R.TSTAR[s[0]]) + du_static) for s in specs}
methods['oracle_saa'] = {s: E2.robust_opt(R.TSTAR[s[0]], s[0], 'cvar', XS, kappa=s[1])[0] for s in specs}
for seed in (0, 1, 2):
    A = pickle.load(open(RES / f'e3b_policy_s{seed}.pkl', 'rb'))['P'][0]
    methods[f'adjoint_policy_s{seed}'] = {s: box(R.u_of(R.TSTAR[s[0]]) + onp.asarray(P.du_of(A, P.obs(*s)[None]))[0]) for s in specs}
curves = {}
for ag, seeds in [('sac_gcn_per', (0, 1)), ('sac_gcn_uniform', (0,)), ('sac_mlp_per', (0,)), ('ppo_gcn', (0, 1))]:
    for sd in seeds:
        S = pickle.load(open(RES / f'e3_{ag}_s{sd}.pkl', 'rb'))
        curves[f'{ag}_s{sd}'] = S['log']
        mm = {}
        for s in specs:
            o = P.obs(*s)
            a = R.sac_act(S, o, True) if S['kind'] == 'sac' else onp.clip(R.ppo_act(S, o, True), -1, 1)
            mm[s] = box(R.u_of(R.TSTAR[s[0]]) + R.STEP * onp.clip(a, -1, 1))
        methods[f'{ag}_s{sd}'] = mm
for name, mm in methods.items():
    st = {f'{s[0]}|{s[1]}': E2.stats(mm[s], s[0], XE, s[1]) for s in specs}
    res[name] = dict(per_spec=st, mean_cvar5=float(onp.mean([v['cvar5_dev'] for v in st.values()])),
                     mean_yield5=float(onp.mean([v['yield_5pct'] for v in st.values()])),
                     mean_abs_nominal_dev=float(onp.mean([abs(v['nominal_dev']) for v in st.values()])))
    print(f"{name:22s} CVaR5={res[name]['mean_cvar5']:.4f} Y5={res[name]['mean_yield5']:.3f} |nom|={res[name]['mean_abs_nominal_dev']:.4f}", flush=True)
res['rl_curves'] = curves
res['adjoint_policy_hist'] = {sd: pickle.load(open(RES / f'e3b_policy_s{sd}.pkl', 'rb'))['hist'] for sd in (0, 1, 2)}
save_json('e3c_eval.json', res)
