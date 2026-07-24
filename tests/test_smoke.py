"""Smoke tests: every module imports and every stage runs a few steps.
Run:  python -m pytest tests/ -q     (or python tests/test_smoke.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from parl_id.adjoint.neural_adjoint import (DesignSpec, NeuralAdjointOptimizer,
                                            gradient_fidelity)
from parl_id.data.cmut_loader import BOUNDS, synthetic_database
from parl_id.data.ceviche_loader import fallback_waveguide_dataset
from parl_id.physics.acoustic_loading import radiation_impedance
from parl_id.physics.cmut_plate import CMUTPlateResidual
from parl_id.physics.helmholtz import HelmholtzResidual, fdfd_solve
from parl_id.pinn.model import PINN, HardBCPlate
from parl_id.pinn.train import make_cmut_pinn, train_cmut_pinn
from parl_id.rl.fab_env import FabricationEnv
from parl_id.rl.ppo_agent import PPO


def test_pinn_forward():
    m = make_cmut_pinn()
    x = torch.rand(10, 7)
    out = m(x)
    assert out.shape == (10, 2)
    # hard BC: w = 0 on boundary
    xb = torch.rand(10, 7); xb[:, 0] = 0.0
    assert torch.allclose(m(xb)[:, 0], torch.zeros(10), atol=1e-6)


def test_plate_residual():
    phys = CMUTPlateResidual()
    xy = torch.rand(32, 2, requires_grad=True)
    m = make_cmut_pinn()
    inp = torch.cat([xy, torch.full((32, 4), 0.5), torch.full((32, 1), 0.4)], 1)
    out = m(inp)
    r1, r2 = phys.residual(out[:, 0:1] * 1e-8, out[:, 1:2] * 1e-8, xy,
                           torch.tensor(0.6e-6), torch.tensor(1.5e-6),
                           torch.tensor(2 * np.pi * 4e6))
    assert r1.shape == (32, 1) and r2.shape == (32, 1)
    assert torch.isfinite(r1).all() and torch.isfinite(r2).all()


def test_train_few_epochs():
    data = synthetic_database(n_combos=5, n_freq=5)
    m = make_cmut_pinn()
    hist = train_cmut_pinn(m, data, epochs=3, n_colloc=64, log_every=10)
    assert len(hist) == 3 and np.isfinite(hist[-1]["total"])


def test_helmholtz_fdfd_and_residual():
    eps, ez = fallback_waveguide_dataset(nx=60, ny=40, n_samples=1)
    assert np.isfinite(ez).all() and np.abs(ez).max() > 0
    phys = HelmholtzResidual()
    xy = torch.rand(16, 2, requires_grad=True)
    net = PINN(in_dim=3, out_dim=2)
    out = net(torch.cat([xy, torch.full((16, 1), 2.0)], 1))
    r_re, r_im = phys.residual(out[:, 0:1], out[:, 1:2], xy,
                               torch.full((16, 1), 2.0))
    assert torch.isfinite(r_re).all() and torch.isfinite(r_im).all()


def test_neural_adjoint_quadratic():
    lo = torch.zeros(3); hi = torch.ones(3)
    spec = DesignSpec(["a", "b", "c"], lo, hi)
    tgt = torch.tensor([0.3, 0.7, 0.5])
    opt = NeuralAdjointOptimizer(lambda t: ((t - tgt) ** 2).sum(), spec,
                                 lr=5e-2)
    theta, j = opt.run(iters=150, restarts=2)
    assert j < 1e-3
    assert gradient_fidelity(torch.ones(4), torch.ones(4)) > 0.999


def test_fab_env_and_ppo():
    names = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])
    theta0 = (lo + hi) / 2
    env = FabricationEnv(lambda t: -float(((t - theta0) ** 2).sum()),
                         theta0, (lo, hi), n_corruptions=4, horizon=4, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (8,)
    _, r, *_ = env.step(np.zeros(4))
    assert np.isfinite(r)
    agent = PPO(env, batch_steps=32, epochs=2)
    agent.train(iterations=1, verbose=False)
    assert len(agent.reward_history) > 0


def test_acoustic_loading():
    z = radiation_impedance(2 * np.pi * 4e6, 16e-6)
    assert np.isfinite(z.real) and np.isfinite(z.imag)


def test_gcn_sac_per():
    from parl_id.rl.gcn_sac import GCNSAC, DeviceGraph, PrioritizedReplayBuffer
    names = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in names])
    hi = np.array([BOUNDS[k][1] for k in names])
    theta0 = (lo + hi) / 2
    env = FabricationEnv(lambda t: -float(((t - theta0) ** 2).sum()),
                         theta0, (lo, hi), n_corruptions=4, horizon=4,
                         spec=0.43, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (9,)  # 2*4 params + spec node
    agent = GCNSAC(env, graph=DeviceGraph(n_params=4), batch=16,
                   start_steps=20, seed=0)
    agent.train(total_steps=60, verbose=False)
    a = agent.act(obs)
    assert a.shape == (4,) and np.isfinite(a).all()
    # PER sanity: priorities updated and sampling respects them
    buf = PrioritizedReplayBuffer(32, 9, 4)
    for i in range(10):
        buf.add(np.zeros(9), np.zeros(4), float(i), np.zeros(9), 0.0)
    idx, w = buf.sample(8, beta=0.5, rng=np.random.default_rng(0))
    buf.update_priorities(idx, np.linspace(0.1, 5.0, 8))
    assert np.isfinite(w).all() and buf.prio[idx].max() > 1.0


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print("all smoke tests passed")
