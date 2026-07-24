"""Virtual fabrication-loop environment (Gymnasium API).

The RL agent receives a nominal (adjoint-optimized) design and a target spec,
applies a bounded correction, and is rewarded by the *yield-aware* performance
of the corrected design under sampled process corruptions:

    reward = mean_k J(theta + a, xi_k) - beta * std_k J(theta + a, xi_k)

Process corruption xi ~ published CMUT fabrication statistics:
  - film thickness deviation:   multiplicative N(1, sigma_t)
  - lateral over/under-etch:    additive N(0, sigma_e) on wall thickness
  - residual stress:            shifts effective rigidity, N(0, sigma_s)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _BASE = gym.Env
except ImportError:  # keep importable without gymnasium
    _BASE = object
    spaces = None


class FabricationEnv(_BASE):
    """theta layout (CMUT): [t_e, t_np, t_ox, t_w] in um."""

    metadata = {"render_modes": []}

    def __init__(self, performance_fn: Callable[[np.ndarray], float],
                 theta_nominal: np.ndarray,
                 bounds: tuple[np.ndarray, np.ndarray],
                 sigma_thickness: float = 0.03,
                 sigma_etch_um: float = 0.08,
                 sigma_stress: float = 0.05,
                 n_corruptions: int = 16,
                 beta: float = 1.0,
                 reward_mode: str = "cvar",
                 alpha: float = 0.05,
                 max_correction: float = 0.15,
                 horizon: int = 8,
                 spec: float | None = None,
                 seed: int | None = None):
        self.reward_mode = reward_mode
        self.alpha = alpha
        self.spec = spec  # normalized target specification appended to obs
        self.performance_fn = performance_fn
        self.theta_nominal = np.asarray(theta_nominal, dtype=np.float64)
        self.lo, self.hi = bounds
        self.sig_t, self.sig_e, self.sig_s = sigma_thickness, sigma_etch_um, sigma_stress
        self.K, self.beta = n_corruptions, beta
        self.max_corr = max_correction
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        d = self.theta_nominal.size
        obs_dim = 2 * d + (1 if spec is not None else 0)
        if spaces is not None:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(d,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,),
                                                dtype=np.float32)
        self._theta = self.theta_nominal.copy()
        self._t = 0

    # ---- process corruption model -------------------------------------
    def corrupt(self, theta: np.ndarray) -> np.ndarray:
        xi = theta.copy()
        xi[:2] *= self.rng.normal(1.0, self.sig_t, size=2)        # film thicknesses
        xi[3] += self.rng.normal(0.0, self.sig_e)                 # wall etch
        xi[2] *= self.rng.normal(1.0, self.sig_s)                 # stress proxy on oxide
        return np.clip(xi, self.lo, self.hi)

    def yield_reward(self, theta: np.ndarray) -> tuple[float, float, float]:
        """Yield-aware reward. mode='cvar' (default, paper Eq. 17) maximizes the
        conditional value-at-risk: the mean of the worst alpha-fraction of the
        corrupted-performance distribution. mode='mean_std' is the conventional
        mu - beta*sigma proxy (paper Eq. 16)."""
        perf = np.array([self.performance_fn(self.corrupt(theta))
                         for _ in range(self.K)])
        mu, sd = float(perf.mean()), float(perf.std())
        if self.reward_mode == "cvar":
            k = max(1, int(np.ceil(self.alpha * self.K)))
            reward = float(np.sort(perf)[:k].mean())
        else:
            reward = mu - self.beta * sd
        return reward, mu, sd

    # ---- gym API --------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._theta = self.theta_nominal.copy()
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        span = self.hi - self.lo
        self._theta = np.clip(
            self._theta + self.max_corr * span * np.asarray(action, dtype=np.float64),
            self.lo, self.hi)
        r, mu, sd = self.yield_reward(self._theta)
        self._t += 1
        terminated = self._t >= self.horizon
        return self._obs(), r, terminated, False, {"mean": mu, "std": sd}

    def _obs(self):
        span = self.hi - self.lo
        parts = [(self._theta - self.lo) / span,
                 (self.theta_nominal - self.lo) / span]
        if self.spec is not None:
            parts.append(np.array([self.spec]))
        return np.concatenate(parts).astype(np.float32)
