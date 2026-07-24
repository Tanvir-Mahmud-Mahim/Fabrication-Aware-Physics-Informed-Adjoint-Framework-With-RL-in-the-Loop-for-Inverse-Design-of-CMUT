"""Minimal PPO (clipped surrogate) for the fabrication-loop environment.
Self-contained; swap for stable-baselines3 PPO for the paper's final runs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, width: int = 64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, width), nn.Tanh(),
                                nn.Linear(width, width), nn.Tanh(),
                                nn.Linear(width, act_dim), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.v = nn.Sequential(nn.Linear(obs_dim, width), nn.Tanh(),
                               nn.Linear(width, width), nn.Tanh(),
                               nn.Linear(width, 1))

    def dist(self, obs):
        mu = self.pi(obs)
        return torch.distributions.Normal(mu, self.log_std.exp())


class PPO:
    def __init__(self, env, gamma: float = 0.99, lam: float = 0.95,
                 clip: float = 0.2, lr: float = 3e-4, epochs: int = 10,
                 batch_steps: int = 256, device: str = "cpu"):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.batch_steps = epochs, batch_steps
        self.device = device
        self.reward_history: list[float] = []

    def collect(self):
        obs_l, act_l, logp_l, rew_l, val_l, done_l = [], [], [], [], [], []
        obs, _ = self.env.reset()
        ep_rew = 0.0
        for _ in range(self.batch_steps):
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                d = self.ac.dist(o)
                a = d.sample()
                logp = d.log_prob(a).sum()
                v = self.ac.v(o).squeeze()
            nobs, r, term, trunc, _ = self.env.step(a.cpu().numpy())
            obs_l.append(o); act_l.append(a); logp_l.append(logp)
            rew_l.append(r); val_l.append(v); done_l.append(term or trunc)
            ep_rew += r
            obs = nobs
            if term or trunc:
                self.reward_history.append(ep_rew)
                ep_rew = 0.0
                obs, _ = self.env.reset()
        return (torch.stack(obs_l), torch.stack(act_l), torch.stack(logp_l),
                np.array(rew_l), torch.stack(val_l).cpu().numpy(),
                np.array(done_l))

    def gae(self, rew, val, done):
        adv = np.zeros_like(rew)
        last = 0.0
        for t in reversed(range(len(rew))):
            nonterm = 1.0 - float(done[t])
            nv = val[t + 1] if t + 1 < len(val) else 0.0
            delta = rew[t] + self.gamma * nv * nonterm - val[t]
            adv[t] = last = delta + self.gamma * self.lam * nonterm * last
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (torch.as_tensor(adv, dtype=torch.float32, device=self.device),
                torch.as_tensor(ret, dtype=torch.float32, device=self.device))

    def train(self, iterations: int = 20, verbose: bool = True):
        for it in range(iterations):
            obs, act, logp_old, rew, val, done = self.collect()
            adv, ret = self.gae(rew, val, done)
            for _ in range(self.epochs):
                d = self.ac.dist(obs)
                logp = d.log_prob(act).sum(-1)
                ratio = (logp - logp_old).exp()
                l_pi = -torch.min(ratio * adv,
                                  ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
                l_v = ((self.ac.v(obs).squeeze(-1) - ret) ** 2).mean()
                loss = l_pi + 0.5 * l_v - 0.01 * d.entropy().sum(-1).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            if verbose and self.reward_history:
                print(f"[PPO {it:3d}] mean ep reward "
                      f"{np.mean(self.reward_history[-8:]):.4f}")
        return self.reward_history
