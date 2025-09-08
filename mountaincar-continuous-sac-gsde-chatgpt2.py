"""
PyTorch SAC + gSDE (corrected & stabilized)

This file replaces the earlier simplified SAC implementation with a more
correct, stable version that:
  - Implements gSDE actor with pre-tanh sampling and analytic log-prob.
  - Applies tanh squashing with change-of-variable correction to log-probs.
  - Uses proper SAC update: target uses entropy term, actor loss minimizes
    (alpha * logp - Q). Supports automatic entropy tuning (recommended).
  - Logs step count, episode returns, sigma and alpha for debugging.
  - Adds observation normalization helper (important for MountainCarContinuous).

Key hyperparameters you can tune for MountainCarContinuous-v0:
  - cfg.use_gsde: True/False
  - cfg.sde_sample_freq: set -1 to keep E fixed during episode (often helps)
  - cfg.log_std_init: initial log-std (try -2.0..-0.5)
  - cfg.batch_size, lr, alpha target
  - obs normalization is enabled by default

This implementation is intentionally explicit and educational; it's not as
optimized as production libraries, but it fixes several issues that caused
instability earlier (missing log-prob correction, missing entropy term, etc.).

Run:
  python sac_gsde_fixed.py

"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================
# Config
# ==========================
@dataclass
class SACConfig:
    env_id: str = "MountainCarContinuous-v0"
    seed: int = 0
    total_steps: int = 200_000
    start_steps: int = 1000
    replay_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    # automatic entropy tuning
    target_entropy: Optional[float] = 2.0 # None  # if None uses -action_dim
    use_auto_alpha: bool = True
    init_alpha: float = 0.2
    # gSDE
    use_gsde: bool = True
    sde_sample_freq: int = -1  # -1: E fixed during episode; helps stability
    feature_dim: int = 64
    log_std_init: float = -1.0
    # obs normalization
    obs_normalize: bool = True

# ==========================
# Utils
# ==========================

def mlp(sizes, activation=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

# Simple running mean/std for observation normalization
class RunningStat:
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape, np.float64)
        self._S = np.zeros(shape, np.float64)

    def push(self, x):
        x = np.asarray(x)
        if x.shape != self._M.shape:
            raise ValueError("shape mismatch")
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return (self._S / (self._n - 1)) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

# ==========================
# gSDE noise module
# ==========================
class gSDENoise(nn.Module):
    def __init__(self, action_dim: int, feature_dim: int, log_std_init: float = -1.0, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.device = device
        self.log_std = nn.Parameter(torch.ones(action_dim, device=device) * log_std_init)
        E = torch.randn(self.feature_dim, self.action_dim, device=device)
        self.register_buffer('E', E)
        self.step_count = 0

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    def maybe_resample(self, sde_sample_freq: int):
        if sde_sample_freq == -1:
            return
        if (self.step_count % max(1, sde_sample_freq)) == 0:
            with torch.no_grad():
                self.E.normal_(0.0, 1.0)
        self.step_count += 1

    def noise_term(self, phi: torch.Tensor) -> torch.Tensor:
        return phi @ self.E

# ==========================
# Actor with gSDE
# ==========================
class GSDEActor(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, cfg:SACConfig, device:torch.device):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.feature_net = mlp([obs_dim, 256, cfg.feature_dim], activation=nn.ReLU)
        self.mu_head = mlp([cfg.feature_dim, 256, act_dim], activation=nn.ReLU, out_act=nn.Identity)
        self.use_gsde = cfg.use_gsde
        if self.use_gsde:
            self.noise = gSDENoise(act_dim, cfg.feature_dim, cfg.log_std_init, device)
        else:
            self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * cfg.log_std_init)

    def forward_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.feature_net(obs)
        mu = self.mu_head(phi)
        return mu, phi

    def sample_pre_tanh(self, obs: torch.Tensor, sde_sample_freq: Optional[int] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, phi = self.forward_features(obs)
        if self.use_gsde:
            if sde_sample_freq is not None:
                self.noise.maybe_resample(sde_sample_freq)
            if deterministic:
                pre_tanh = mu
            else:
                noise = self.noise.noise_term(phi)
                pre_tanh = mu + noise * self.noise.sigma
            std_state = torch.norm(phi, p=2, dim=1, keepdim=True)
            std = std_state * self.noise.sigma.view(1, -1)
        else:
            sigma = torch.exp(self.log_std)
            if deterministic:
                pre_tanh = mu
            else:
                pre_tanh = mu + torch.randn_like(mu) * sigma.view(1, -1)
            std = sigma.view(1, -1).expand_as(mu)
        std = std + 1e-8
        var = std * std
        log_prob = -0.5 * (((pre_tanh - mu) ** 2) / var + 2 * torch.log(std) + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=1)
        return pre_tanh, log_prob

    def log_prob_from_pre_tanh(self, obs: torch.Tensor, pre_tanh: torch.Tensor) -> torch.Tensor:
        mu, phi = self.forward_features(obs)
        if self.use_gsde:
            std_state = torch.norm(phi, p=2, dim=1, keepdim=True)
            std = std_state * self.noise.sigma.view(1, -1)
        else:
            std = torch.exp(self.log_std).view(1, -1).expand_as(mu)
        std = std + 1e-8
        var = std * std
        log_prob = -0.5 * (((pre_tanh - mu) ** 2) / var + 2 * torch.log(std) + math.log(2 * math.pi))
        return log_prob.sum(dim=1)

# ==========================
# Squash (tanh) helpers
# ==========================
def squash_and_correct(pre_tanh: torch.Tensor, logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tanh_a = torch.tanh(pre_tanh)
    # log|det Jacobian| = sum log(1 - tanh^2)
    log_det = torch.log(1 - tanh_a.pow(2) + 1e-6).sum(dim=1)
    corrected = logp - log_det
    return tanh_a, corrected

# ==========================
# Q network
# ==========================
class QNetwork(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, 256, 256, 1], activation=nn.ReLU)
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=1)
        return self.net(x).squeeze(-1)

# ==========================
# Replay buffer (numpy-backed for efficiency)
# ==========================
class ReplayBuffer:
    def __init__(self, obs_dim:int, act_dim:int, size:int, device:torch.device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size:int):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idx], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(self.act_buf[idx], dtype=torch.float32, device=self.device)
        rews = torch.as_tensor(self.rew_buf[idx], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(self.next_obs_buf[idx], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(self.done_buf[idx], dtype=torch.float32, device=self.device)
        return obs, acts, rews, next_obs, done

# ==========================
# SAC Agent
# ==========================
class SACAgent:
    def __init__(self, env:gym.Env, cfg:SACConfig, device:torch.device):
        self.env = env
        self.cfg = cfg
        self.device = device
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # networks
        self.actor = GSDEActor(obs_dim, act_dim, cfg, device).to(device)
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_targ = QNetwork(obs_dim, act_dim).to(device)
        self.q2_targ = QNetwork(obs_dim, act_dim).to(device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        # optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)

        # entropy alpha (auto)
        if cfg.use_auto_alpha:
            self.log_alpha = torch.tensor(math.log(cfg.init_alpha), requires_grad=True, device=device)
            self.log_alpha = nn.Parameter(self.log_alpha)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -act_dim if cfg.target_entropy is None else cfg.target_entropy
        else:
            self.log_alpha = None
            self.alpha = cfg.init_alpha

        # replay
        self.replay = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, device)

        # obs normalizer
        if cfg.obs_normalize:
            self.obs_rstat = RunningStat((obs_dim,))
        else:
            self.obs_rstat = None

        self.total_steps = 0

    def normalize_obs(self, o: np.ndarray) -> np.ndarray:
        if self.obs_rstat is None:
            return o
        self.obs_rstat.push(o)
        mean = self.obs_rstat.mean
        std = self.obs_rstat.std
        return (o - mean) / (std + 1e-8)

    def scale_action(self, tanh_action: torch.Tensor) -> torch.Tensor:
        return tanh_action * self.action_scale + self.action_bias

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_n = self.normalize_obs(obs.copy()) if self.obs_rstat is not None else obs
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            pre_tanh, logp = self.actor.sample_pre_tanh(obs_t, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None), deterministic=deterministic)
            tanh_a, logp_corr = squash_and_correct(pre_tanh, logp)
            a_env = self.scale_action(tanh_a)
        return a_env.cpu().numpy().squeeze(0)

    def update(self):
        if self.replay.size < self.cfg.batch_size:
            return
        obs, acts, rews, next_obs, done = self.replay.sample(self.cfg.batch_size)

        # compute alpha
        if self.log_alpha is not None:
            alpha = torch.exp(self.log_alpha)
        else:
            alpha = torch.tensor(self.alpha, device=self.device)

        with torch.no_grad():
            pre_next, logp_next = self.actor.sample_pre_tanh(next_obs, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None), deterministic=False)
            tanh_next, logp_next_corr = squash_and_correct(pre_next, logp_next)
            a_next_env = self.scale_action(tanh_next)
            q1_next = self.q1_targ(next_obs, a_next_env)
            q2_next = self.q2_targ(next_obs, a_next_env)
            q_next = torch.min(q1_next, q2_next)
            target_v = q_next - alpha * logp_next_corr
            target_q = rews + (1.0 - done) * self.cfg.gamma * target_v

        q1 = self.q1(obs, acts)
        q2 = self.q2(obs, acts)
        q_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # policy update
        pre_pi, logp_pi = self.actor.sample_pre_tanh(obs, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None), deterministic=False)
        tanh_pi, logp_pi_corr = squash_and_correct(pre_pi, logp_pi)
        a_pi_env = self.scale_action(tanh_pi)
        q1_pi = self.q1(obs, a_pi_env)
        q2_pi = self.q2(obs, a_pi_env)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (alpha * logp_pi_corr - q_pi).mean()

        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        # alpha update
        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (logp_pi_corr + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha = torch.exp(self.log_alpha)

        # soft updates
        for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
            p_targ.data.mul_(1 - self.cfg.tau)
            p_targ.data.add_(self.cfg.tau * p.data)
        for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
            p_targ.data.mul_(1 - self.cfg.tau)
            p_targ.data.add_(self.cfg.tau * p.data)

    def train(self):
        env = self.env
        obs, _ = env.reset(seed=self.cfg.seed)
        if self.obs_rstat is not None:
            obs_n = self.normalize_obs(obs.copy())
        else:
            obs_n = obs
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        ep_ret, ep_len = 0.0, 0
        start = time.time()
        for step in range(1, self.cfg.total_steps + 1):
            self.total_steps = step
            if step < self.cfg.start_steps:
                a = env.action_space.sample()
            else:
                a = self.select_action(obs, deterministic=False)

            next_obs, rew, terminated, truncated, _ = env.step(a)
            done = float(terminated or truncated)
            # store raw obs (not normalized) and action
            self.replay.add(obs.copy(), a.copy(), rew, next_obs.copy(), done)

            # update obs normalization stats
            if self.obs_rstat is not None:
                self.normalize_obs(next_obs.copy())

            obs = next_obs.copy()
            ep_ret += float(rew)
            ep_len += 1

            if done or (ep_len >= env.spec.max_episode_steps):
                # log
                sigma_val = None
                alpha_val = None
                if self.cfg.use_gsde:
                    sigma_val = self.actor.noise.sigma.detach().cpu().numpy()
                if self.log_alpha is not None:
                    alpha_val = float(torch.exp(self.log_alpha).detach().cpu().numpy())
                print(f"Step {step}, Episode return {ep_ret:.1f}, length {ep_len}, sigma {sigma_val}, alpha {alpha_val}")
                obs, _ = env.reset()
                obs = obs
                ep_ret, ep_len = 0.0, 0

            # learning step
            if step >= self.cfg.start_steps and (step - self.cfg.start_steps) % 1 == 0:
                # do a single SGD update per environment step
                self.update()

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cfg = SACConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(env, cfg, device)
    agent.train()
