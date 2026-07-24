"""Evolutionary warm-start for the fabrication-loop PPO agent.

Inspired by Evo-PHORCED (Meghwar et al., 2025), which cures the RL cold-start
problem with a coarse evolutionary search before policy-gradient learning ---
but applied here to the *yield* objective (CVaR of the corrupted-performance
distribution) evaluated through the millisecond PINN surrogate, rather than to
a nominal objective through full-wave simulation. The best evolutionary
individual initializes the PPO policy mean, so the agent starts inside a good
correction basin instead of exploring from zero.
"""

from __future__ import annotations

import numpy as np
import torch


def evolutionary_warmstart(env, generations: int = 10, pop: int = 16,
                           sigma0: float = 0.3, elite_frac: float = 0.25,
                           seed: int = 0) -> tuple[np.ndarray, float]:
    """(mu, sigma)-ES over the correction action space of a FabricationEnv.

    Returns (best_action, best_reward). Each individual is a single correction
    action applied to the nominal design; fitness is the env's yield reward.
    """
    rng = np.random.default_rng(seed)
    d = env.action_space.shape[0]
    mean, sigma = np.zeros(d), sigma0
    n_elite = max(1, int(elite_frac * pop))
    best_a, best_r = np.zeros(d), -np.inf
    for g in range(generations):
        cands = np.clip(mean + sigma * rng.standard_normal((pop, d)), -1, 1)
        fits = []
        for a in cands:
            env.reset()
            _, r, *_ = env.step(a)
            fits.append(r)
        fits = np.asarray(fits)
        order = np.argsort(fits)[::-1]
        if fits[order[0]] > best_r:
            best_r, best_a = float(fits[order[0]]), cands[order[0]].copy()
        elite = cands[order[:n_elite]]
        mean = elite.mean(axis=0)
        sigma = max(0.05, float(elite.std(axis=0).mean()))
    return best_a, best_r


def inject_warmstart(agent, action: np.ndarray) -> None:
    """Bias the PPO actor's final layer so the initial policy mean equals the
    warm-start action (state-independent offset; tanh-inverted)."""
    a = np.clip(action, -0.999, 0.999)
    pre = np.arctanh(a)
    last = agent.ac.pi[-2]  # Linear layer before the final Tanh
    with torch.no_grad():
        last.bias.copy_(torch.tensor(pre, dtype=last.bias.dtype))
