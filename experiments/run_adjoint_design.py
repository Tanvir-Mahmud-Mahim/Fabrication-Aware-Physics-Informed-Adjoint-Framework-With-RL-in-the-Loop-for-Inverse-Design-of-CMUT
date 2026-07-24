"""Experiment 2 (Fig. F5): neural-adjoint inverse design through the CMUT PINN.

Objective: maximize average membrane displacement at a target frequency
(equivalently minimize J = -w_avg), the CMUT sensitivity criterion.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from parl_id.adjoint.neural_adjoint import DesignSpec, NeuralAdjointOptimizer  # noqa: E402
from parl_id.data.cmut_loader import BOUNDS  # noqa: E402
from parl_id.pinn.train import make_cmut_pinn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cmut_pinn.pt")
    ap.add_argument("--target-freq", type=float, default=4.3, help="MHz")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_cmut_pinn(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen surrogate

    names = list(BOUNDS)
    lo = torch.tensor([BOUNDS[k][0] for k in names], device=device)
    hi = torch.tensor([BOUNDS[k][1] for k in names], device=device)
    spec = DesignSpec(names, lo, hi)

    omega = torch.tensor([[args.target_freq / 10.0]], device=device)
    xy_q = torch.rand(128, 2, device=device)

    def objective(theta):
        inp = torch.cat([xy_q, theta.expand(128, 4),
                         omega.expand(128, 1)], dim=1)
        w_avg = model(inp)[:, 0:1].mean()
        return -w_avg  # maximize displacement

    opt = NeuralAdjointOptimizer(objective, spec, method="adam", lr=2e-2)
    theta_star, j_star = opt.run(iters=args.iters, restarts=4)

    print("optimal design (um):")
    for k, v in zip(names, theta_star.tolist()):
        print(f"  {k:5s} = {v:.4f}")
    print(f"objective J* = {j_star:.4e} (w_avg = {-j_star:.4e} normalized)")

    Path("outputs").mkdir(exist_ok=True)
    torch.save({"theta": theta_star, "J": j_star,
                "trajectory": opt.trajectory}, "outputs/adjoint_design.pt")


if __name__ == "__main__":
    main()
