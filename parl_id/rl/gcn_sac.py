"""Graph-encoded Soft Actor-Critic with Prioritized Experience Replay for the
virtual fabrication loop (PARL-ID Stage 3, main agent).

Why this composition:
- The device is encoded as a *graph*: one node per design parameter plus one
  specification node (target frequency), with edges expressing physical
  adjacency in the layer stack. A shared GCN encoder therefore produces a
  structure-aware state embedding whose size is independent of the number of
  design parameters -- the property that lets one trained policy transfer
  across devices with different parameter counts (CMUT thickness vector,
  photonic density patch), which a fixed-size MLP policy cannot do.
- Soft actor-critic (SAC) is off-policy: every corruption-ensemble reward
  evaluation is stored and re-used, instead of being discarded after one
  on-policy update as in PPO.
- Prioritized experience replay (PER) replays transitions with large TD error
  first. Under the CVaR reward these are precisely the rare lower-tail process
  outcomes -- the ones that determine manufacturing yield -- so PER and the
  tail-risk objective are mutually reinforcing.

Self-contained (PyTorch only; no torch-geometric dependency).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Device graph
# --------------------------------------------------------------------------
class DeviceGraph:
    """Nodes: one per design parameter (+1 spec node appended last).
    Node features: [theta_cur_i, theta_star_i, is_spec, spec_value].
    Edges: physical adjacency; defaults to the CMUT layer-stack topology.
    """

    #            t_e  t_np t_ox t_w  spec
    CMUT_EDGES = [(0, 1),          # electrode <-> nitride (composite membrane)
                  (2, 3),          # oxide <-> wall (support structure)
                  (0, 2), (1, 3),  # membrane <-> support coupling
                  (4, 0), (4, 1), (4, 2), (4, 3)]  # spec node to all params

    def __init__(self, n_params: int = 4, edges: list[tuple[int, int]] | None = None):
        self.n_params = n_params
        self.n_nodes = n_params + 1
        e = edges if edges is not None else self.CMUT_EDGES
        A = np.eye(self.n_nodes, dtype=np.float32)          # self loops
        for i, j in e:
            A[i, j] = A[j, i] = 1.0
        d = A.sum(1)
        self.A_hat = torch.tensor(A / np.sqrt(np.outer(d, d)),
                                  dtype=torch.float32)      # sym-normalized

    def node_features(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [B, 2*n_params(+1 spec)] -> node features [B, n_nodes, 4]."""
        B = obs.shape[0]
        n = self.n_params
        cur, star = obs[:, :n], obs[:, n:2 * n]
        spec = (obs[:, 2 * n:2 * n + 1] if obs.shape[1] > 2 * n
                else torch.zeros(B, 1, device=obs.device))
        x = torch.zeros(B, self.n_nodes, 4, device=obs.device)
        x[:, :n, 0] = cur
        x[:, :n, 1] = star
        x[:, n, 2] = 1.0                                    # spec-node flag
        x[:, :, 3] = spec                                   # broadcast spec
        return x


class GCNEncoder(nn.Module):
    """Two-layer graph convolution H' = relu(A_hat H W), mean-pooled."""

    def __init__(self, graph: DeviceGraph, in_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.register_buffer("A", graph.A_hat)
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # [B, N, F]
        h = F.relu(self.w1(torch.einsum("nm,bmf->bnf", self.A, x)))
        h = F.relu(self.w2(torch.einsum("nm,bmf->bnf", self.A, h)))
        return h.mean(dim=1)                                 # [B, hidden]


# --------------------------------------------------------------------------
# Actor / critics
# --------------------------------------------------------------------------
LOG_STD_MIN, LOG_STD_MAX = -8.0, 2.0


class GCNActor(nn.Module):
    def __init__(self, graph: DeviceGraph, act_dim: int, hidden: int = 64):
        super().__init__()
        self.graph = graph
        self.enc = GCNEncoder(graph, hidden=hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        z = self.enc(self.graph.node_features(obs))
        mu = self.mu(z)
        log_std = self.log_std(z).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        mu, log_std = self(obs)
        dist = torch.distributions.Normal(mu, log_std.exp())
        u = dist.rsample()
        a = torch.tanh(u)
        # tanh-squashed log-prob correction
        logp = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp, torch.tanh(mu)


class GCNCritic(nn.Module):
    """Twin Q-networks over the shared graph embedding + action."""

    def __init__(self, graph: DeviceGraph, act_dim: int, hidden: int = 64):
        super().__init__()
        self.graph = graph
        self.enc = GCNEncoder(graph, hidden=hidden)

        def head():
            return nn.Sequential(nn.Linear(hidden + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = head(), head()

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        z = self.enc(self.graph.node_features(obs))
        za = torch.cat([z, act], dim=-1)
        return self.q1(za).squeeze(-1), self.q2(za).squeeze(-1)


# --------------------------------------------------------------------------
# Prioritized replay
# --------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    """Proportional PER (Schaul et al., 2016): P(i) ~ (|delta_i| + eps)^omega,
    with importance-sampling weights annealed by beta."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 omega: float = 0.6, beta0: float = 0.4, eps: float = 1e-4):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.nobs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.prio = np.zeros(capacity, np.float32)
        self.omega, self.beta0, self.eps = omega, beta0, eps
        self.ptr, self.size = 0, 0

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.nobs[i], self.done[i] = no, d
        self.prio[i] = self.prio[:self.size].max() if self.size else 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, beta: float, rng: np.random.Generator):
        p = self.prio[:self.size] ** self.omega
        p = p / p.sum()
        idx = rng.choice(self.size, size=batch, p=p)
        w = (self.size * p[idx]) ** (-beta)
        w = w / w.max()
        return idx, w.astype(np.float32)

    def update_priorities(self, idx, td_err):
        self.prio[idx] = np.abs(td_err) + self.eps


# --------------------------------------------------------------------------
# SAC agent
# --------------------------------------------------------------------------
class GCNSAC:
    def __init__(self, env, graph: DeviceGraph | None = None,
                 gamma: float = 0.99, tau: float = 0.005, lr: float = 3e-4,
                 buffer_size: int = 50_000, batch: int = 128,
                 start_steps: int = 200, updates_per_step: int = 1,
                 device: str = "cpu", seed: int = 0):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.graph = graph or DeviceGraph(n_params=act_dim)
        self.actor = GCNActor(self.graph, act_dim).to(device)
        self.critic = GCNCritic(self.graph, act_dim).to(device)
        self.critic_t = GCNCritic(self.graph, act_dim).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # automatic entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)
        self.buf = PrioritizedReplayBuffer(buffer_size, obs_dim, act_dim)
        self.gamma, self.tau, self.batch = gamma, tau, batch
        self.start_steps, self.ups = start_steps, updates_per_step
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.reward_history: list[float] = []

    # ---------------- interaction ----------------
    def train(self, total_steps: int = 2000, verbose: bool = True):
        obs, _ = self.env.reset()
        ep_rew, step_in_ep = 0.0, 0
        for t in range(total_steps):
            if t < self.start_steps:
                a = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    o = torch.tensor(obs, dtype=torch.float32,
                                     device=self.device)[None]
                    a, _, _ = self.actor.sample(o)
                    a = a[0].cpu().numpy()
            nobs, r, term, trunc, _ = self.env.step(a)
            d = float(term)
            self.buf.add(obs, a, r, nobs, d)
            ep_rew += r; step_in_ep += 1
            obs = nobs
            if term or trunc:
                self.reward_history.append(ep_rew)
                obs, _ = self.env.reset()
                ep_rew, step_in_ep = 0.0, 0
            if self.buf.size >= self.batch:
                beta = self.buf.beta0 + (1 - self.buf.beta0) * t / total_steps
                for _ in range(self.ups):
                    self._update(beta)
            if verbose and t % 200 == 0 and self.reward_history:
                print(f"[GCN-SAC {t:5d}] mean ep reward "
                      f"{np.mean(self.reward_history[-8:]):.4f} "
                      f"alpha={self.log_alpha.exp().item():.3f}")
        return self.reward_history

    # ---------------- SAC update with PER ----------------
    def _update(self, beta: float):
        idx, w = self.buf.sample(self.batch, beta, self.rng)
        to = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)
        o, a = to(self.buf.obs[idx]), to(self.buf.act[idx])
        r, no = to(self.buf.rew[idx]), to(self.buf.nobs[idx])
        d, w_t = to(self.buf.done[idx]), to(w)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            na, nlogp, _ = self.actor.sample(no)
            q1t, q2t = self.critic_t(no, na)
            q_targ = r + self.gamma * (1 - d) * (torch.min(q1t, q2t)
                                                 - alpha * nlogp)
        q1, q2 = self.critic(o, a)
        td = q_targ - q1
        loss_c = (w_t * (td ** 2)).mean() + (w_t * ((q_targ - q2) ** 2)).mean()
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        self.buf.update_priorities(idx, td.detach().cpu().numpy())

        pa, logp, _ = self.actor.sample(o)
        q1p, q2p = self.critic(o, pa)
        loss_a = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

        loss_alpha = -(self.log_alpha.exp()
                       * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(),
                             self.critic_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

    # ---------------- deployment ----------------
    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.tensor(obs, dtype=torch.float32, device=self.device)[None]
        _, _, mu = self.actor.sample(o)
        return mu[0].cpu().numpy()
