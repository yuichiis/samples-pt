"""
PyTorch gSDE SAC with logging

This file implements Soft Actor-Critic (SAC) with generalized State-Dependent Exploration (gSDE).
Now extended to log training steps and sigma values for debugging.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================
# Utilities
# ==========================
def mlp(sizes, activation=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

@dataclass
class SACConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1_000_000
    start_steps: int = 10000
    update_after: int = 1000
    update_every: int = 50
    alpha: float = 0.2  # entropy temperature (fixed)
    # gSDE
    use_gsde: bool = True
    sde_sample_freq: int = 1
    feature_dim: int = 64
    log_std_init: float = -0.5

# ==========================
# gSDE distribution
# ==========================
class gSDEDiagGaussian(nn.Module):
    def __init__(self, action_dim: int, feature_dim: int, log_std_init: float = -0.5, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.device = device
        self.log_std = nn.Parameter(torch.ones(action_dim, device=device) * log_std_init)
        E = torch.randn(self.feature_dim, self.action_dim, device=self.device)
        self.register_buffer('E', E)
        self.step_count = 0

    @property
    def sigma(self):
        return torch.exp(self.log_std)

    def maybe_resample(self, sde_sample_freq: int):
        if sde_sample_freq == -1:
            return
        if (self.step_count % max(1, sde_sample_freq)) == 0:
            with torch.no_grad():
                self.E.normal_(0.0, 1.0)
        self.step_count += 1

    def rsample(self, mu: torch.Tensor, phi: torch.Tensor, deterministic: bool = False):
        B = mu.shape[0]
        if deterministic:
            noise_term = torch.zeros(B, self.action_dim, device=mu.device)
        else:
            noise_term = phi @ self.E
        action = mu + noise_term * self.sigma
        return action

# ==========================
# Actor & Critics
# ==========================
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high, cfg: SACConfig, device: torch.device):
        super().__init__()
        self.device = device
        self.act_low = torch.as_tensor(act_low, dtype=torch.float32, device=device)
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32, device=device)
        self.feature_net = mlp([obs_dim, 256, cfg.feature_dim])
        self.mu_head = mlp([cfg.feature_dim, 256, act_dim], activation=nn.ReLU, out_act=nn.Identity)
        if cfg.use_gsde:
            self.dist = gSDEDiagGaussian(act_dim, cfg.feature_dim, cfg.log_std_init, device)
        else:
            self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * cfg.log_std_init)

    def forward(self, obs: torch.Tensor, deterministic=False, sde_sample_freq: Optional[int] = None):
        phi = self.feature_net(obs)
        mu = self.mu_head(phi)
        if hasattr(self, 'dist'):
            if sde_sample_freq is not None:
                self.dist.maybe_resample(sde_sample_freq)
            action = self.dist.rsample(mu, phi, deterministic)
        else:
            sigma = torch.exp(self.log_std).view(1, -1)
            action = mu if deterministic else mu + sigma * torch.randn_like(mu)
        action = torch.tanh(action)
        action = self.act_low + 0.5 * (action + 1.0) * (self.act_high - self.act_low)
        return action

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, 256, 256, 1])
    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)

# ==========================
# Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.obs2_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.acts_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rews_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.max_size, self.ptr, self.size = size, 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

# ==========================
# SAC Agent
# ==========================
class SACAgent:
    def __init__(self, env: gym.Env, cfg: SACConfig, device: torch.device):
        self.env = env
        self.cfg = cfg
        self.device = device
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, act_dim, env.action_space.low, env.action_space.high, cfg, device).to(device)
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.buf = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=cfg.lr)

    def update(self, data):
        obs, obs2, acts, rews, done = data['obs'], data['obs2'], data['acts'], data['rews'], data['done']
        with torch.no_grad():
            next_acts = self.actor(obs2)
            q1_next = self.q1_target(obs2, next_acts)
            q2_next = self.q2_target(obs2, next_acts)
            q_next = torch.min(q1_next, q2_next)
            target = rews + self.cfg.gamma * (1 - done) * (q_next - self.cfg.alpha * 0)
        q1 = self.q1(obs, acts)
        q2 = self.q2(obs, acts)
        q1_loss = ((q1 - target) ** 2).mean()
        q2_loss = ((q2 - target) ** 2).mean()
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()
        pi = self.actor(obs)
        q1_pi = self.q1(obs, pi)
        q2_pi = self.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.cfg.alpha * 0 - q_pi).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)

    def train(self):
        obs, _ = self.env.reset(seed=self.cfg.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        ep_ret, ep_len = 0, 0
        for t in range(self.cfg.total_steps):
            if t < self.cfg.start_steps:
                act = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    act = self.actor(obs.unsqueeze(0)).squeeze(0).cpu().numpy()
            next_obs, rew, terminated, truncated, _ = self.env.step(act)
            done = terminated or truncated
            self.buf.store(obs, torch.as_tensor(act, dtype=torch.float32, device=self.device), torch.tensor(rew, device=self.device), torch.as_tensor(next_obs, dtype=torch.float32, device=self.device), float(done))
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            ep_ret += rew; ep_len += 1
            if done:
                sigma_info = None
                if hasattr(self.actor, 'dist'):
                    sigma_info = self.actor.dist.sigma.detach().cpu().numpy()
                print(f"Step {t}, Episode return {ep_ret:.1f}, length {ep_len}, sigma {sigma_info}")
                obs, _ = self.env.reset()
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                ep_ret, ep_len = 0, 0
            if t >= self.cfg.update_after and t % self.cfg.update_every == 0:
                for _ in range(self.cfg.update_every):
                    batch = self.buf.sample_batch(self.cfg.batch_size)
                    self.update(batch)

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cfg = SACConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    env = gym.make(cfg.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(env, cfg, device)
    agent.train()
