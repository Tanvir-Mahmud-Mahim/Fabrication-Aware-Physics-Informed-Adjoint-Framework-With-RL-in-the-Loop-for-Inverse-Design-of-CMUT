"""Experiment 3 (Fig. F7/F8): RL in the fabrication loop.

PPO learns a yield-aware correction to the nominal adjoint design, evaluated
through the frozen PINN under sampled process corruptions.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from parl_id.data.cmut_loader import BOUNDS  # noqa: E402
from parl_id.pinn.train import make_cmut_pinn  # noqa: E402
from parl_id.rl.evo_warmstart import evolutionary_warmstart, inject_warmstart  # noqa: E402
from parl_id.rl.fab_env import FabricationEnv  # noqa: E402
from parl_id.rl.gcn_sac import GCNSAC, DeviceGraph  # noqa: E402
from parl_id.rl.ppo_agent import PPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", default="outputs/cmut_pinn.pt")
    ap.add_argument("--design", default="outputs/adjoint_design.pt")
    ap.add_argument("--target-freq", type=float, default=4.3)
    ap.add_argument("--rl-iters", type=int, default=20,
                    help="PPO iterations (x256 steps) / GCN-SAC steps = 100x this")
    ap.add_argument("--agent", choices=["gcn-sac", "ppo"], default="gcn-sac",
                    help="graph-encoded SAC with prioritized replay (main) or PPO baseline")
    ap.add_argument("--reward", choices=["cvar", "mean_std"], default="cvar",
                    help="tail-risk CVaR reward (paper Eq. 17) or mu-beta*sigma baseline")
    ap.add_argument("--no-evo-warmstart", action="store_true",
                    help="disable the evolutionary warm start of the policy")
    args = ap.parse_args()

    device = "cpu"  # env stepping dominated by small forward passes
    model = make_cmut_pinn(device)
    model.load_state_dict(torch.load(args.pinn, map_location=device,
                                     weights_only=False)["model"])
    model.eval()

    theta_nom = torch.load(args.design, map_location=device,
                           weights_only=False)["theta"].cpu().numpy()
    omega = torch.tensor([[args.target_freq / 10.0]])
    xy_q = torch.rand(64, 2)

    @torch.no_grad()
    def performance(theta_np: np.ndarray) -> float:
        theta = torch.tensor(theta_np, dtype=torch.float32)
        inp = torch.cat([xy_q, theta.expand(64, 4), omega.expand(64, 1)], dim=1)
        return float(model(inp)[:, 0:1].mean())

    names = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])

    env = FabricationEnv(performance, theta_nom, (lo, hi), seed=0,
                         reward_mode=args.reward,
                         spec=args.target_freq / 10.0)

    # --- yield of nominal design (baseline) ---
    r_nom, mu_nom, sd_nom = env.yield_reward(theta_nom)
    print(f"nominal design: yield-reward={r_nom:.4e} mean={mu_nom:.4e} "
          f"std={sd_nom:.4e} [agent: {args.agent}, reward: {args.reward}]")

    # --- agent training (with Evo-PHORCED-style warm start on the yield reward) ---
    if args.agent == "gcn-sac":
        agent = GCNSAC(env, graph=DeviceGraph(n_params=4), seed=0)
        if not args.no_evo_warmstart:
            a0, r0 = evolutionary_warmstart(env, generations=10, pop=16, seed=0)
            with torch.no_grad():
                agent.actor.mu.bias.copy_(
                    torch.arctanh(torch.tensor(np.clip(a0, -0.999, 0.999),
                                               dtype=torch.float32)))
            print(f"evolutionary warm start: reward={r0:.4e}")
        agent.train(total_steps=100 * args.rl_iters)
    else:
        agent = PPO(env, batch_steps=256)
        if not args.no_evo_warmstart:
            a0, r0 = evolutionary_warmstart(env, generations=10, pop=16, seed=0)
            inject_warmstart(agent, a0)
            print(f"evolutionary warm start: reward={r0:.4e}")
        agent.train(iterations=args.rl_iters)

    # --- evaluate learned correction ---
    obs, _ = env.reset()
    if args.agent == "gcn-sac":
        a = agent.act(obs)
    else:
        with torch.no_grad():
            a = agent.ac.pi(torch.tensor(obs)).numpy()
    env.step(a)
    theta_rob = env._theta
    r_rob, mu_rob, sd_rob = env.yield_reward(theta_rob)
    print(f"robust design:  yield-reward={r_rob:.4e} mean={mu_rob:.4e} "
          f"std={sd_rob:.4e}")
    print("correction (um):", np.round(theta_rob - theta_nom, 4))

    Path("outputs").mkdir(exist_ok=True)
    np.savez("outputs/rl_fab_loop.npz", theta_nominal=theta_nom,
             theta_robust=theta_rob, rewards=np.array(agent.reward_history))


if __name__ == "__main__":
    main()
