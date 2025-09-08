"""
PyTorch gSDE PPO (fixed)

This file is a compact PyTorch implementation of PPO with generalized
State-Dependent Exploration (gSDE) following Raffin et al. (arXiv:2005.05719).

Fixes applied in this version (per user's report):
  - gSDEDiagGaussian is now a proper `nn.Module` so its trainable
    parameter `log_std` is registered and updated by the optimizer.
  - Exploration matrix `E` is stored as a buffer (resampled in-place).
  - `sigma` (exp(log_std)) will now be optimized by Adam.
  - Training prints the mean episode return observed during each rollout
    so it's easier to tell if learning is happening.

Usage:
  Ensure PyTorch and Gymnasium are installed. Run:
    python gsde_ppo_fixed.py

Notes:
  - This is a minimal educational implementation. For production use,
    prefer Stable-Baselines3's implementation.
"""
from __future__ import annotations

import math
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

def mlp(sizes, activation=nn.Tanh, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

@dataclass
class PPOConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 200_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    batch_size: int = 64
    rollout_steps: int = 2048
    # gSDE settings
    use_gsde: bool = True
    sde_sample_freq: int = 4  # set to -1 to keep E constant during rollout (smoother)
    feature_dim: int = 64     # dim of ϕ(s)
    log_std_init: float = -0.5

# ==========================
# gSDE distribution (fixed: subclass nn.Module)
# ==========================
class gSDEDiagGaussian(nn.Module):
    """Generalized SDE diagonal Gaussian with state-dependent std.

    Implements: a(s) = mu(s) + (phi(s) @ E) ⊙ sigma
    where E ~ N(0, I) resampled periodically and sigma = exp(log_std) is trainable.
    """
    def __init__(self, action_dim: int, feature_dim: int, log_std_init: float = -0.5, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.device = device
        # Trainable per-action log std (σ = exp(log_std))
        self.log_std = nn.Parameter(torch.ones(action_dim, device=device) * log_std_init)
        # Exploration matrix E (feature_dim × action_dim), stored as buffer and resampled
        E = torch.randn(self.feature_dim, self.action_dim, device=self.device)
        self.register_buffer('E', E)
        # counter for sampling schedule (not a buffer since it's an int)
        self.step_count = 0

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_std)  # (A,)

    def maybe_resample(self, sde_sample_freq: int):
        # sde_sample_freq == -1 -> keep E fixed
        if sde_sample_freq == -1:
            return
        if (self.step_count % max(1, sde_sample_freq)) == 0:
            with torch.no_grad():
                # resample in-place to keep buffer identity
                self.E.normal_(mean=0.0, std=1.0)
        self.step_count += 1

    def sample(self, mu: torch.Tensor, phi: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mu: (B, A)
            phi: (B, F)
        Returns:
            action: (B, A)
            log_prob: (B,)
            entropy: (B,)
        """
        B = mu.shape[0]
        # State-dependent noise term: (B,F) @ (F,A) -> (B,A)
        if deterministic:
            noise_term = torch.zeros(B, self.action_dim, device=mu.device)
        else:
            noise_term = phi @ self.E  # (B, A)
        # scale each action dim with σ
        action = mu + noise_term * self.sigma  # (B, A)

        # Log-prob under diag Gaussian with std_j(s) = ||phi(s)|| * σ_j
        std_state = torch.norm(phi, p=2, dim=1, keepdim=True)  # (B,1)
        std = std_state * self.sigma.view(1, -1)               # (B,A)
        var = std * std
        # Numerical stability
        std = std + 1e-8
        var = var + 1e-8
        log_prob = -0.5 * (((action - mu) ** 2) / var + 2 * torch.log(std) + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=1)  # (B,)
        # Entropy of diagonal Gaussian with those stds
        entropy = 0.5 + 0.5 * torch.log(2 * torch.pi * var)
        entropy = entropy.sum(dim=1)
        return action, log_prob, entropy

# ==========================
# Actor-Critic with gSDE
# ==========================
class ActorCritic(nn.Module):
    def __init__(self, obs_space: gym.Space, act_space: gym.Space, cfg: PPOConfig, device: torch.device):
        super().__init__()
        assert isinstance(act_space, gym.spaces.Box)
        self.device = device
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        self.act_low = torch.as_tensor(act_space.low, dtype=torch.float32, device=device)
        self.act_high = torch.as_tensor(act_space.high, dtype=torch.float32, device=device)

        # Feature extractor ϕ(s)
        self.feature_net = mlp([obs_dim, 128, cfg.feature_dim])
        # Policy mean μ(s)
        self.mu_head = mlp([cfg.feature_dim, 64, act_dim], activation=nn.Tanh, out_act=nn.Identity)
        # Value function V(s)
        self.v_head = mlp([cfg.feature_dim, 64, 1], activation=nn.Tanh, out_act=nn.Identity)

        if cfg.use_gsde:
            # gSDE module registers its parameters with this parent module
            self.dist = gSDEDiagGaussian(act_dim, cfg.feature_dim, cfg.log_std_init, device)
        else:
            # Fallback to classic diagonal Gaussian with state-independent std
            self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * cfg.log_std_init)

    def clamp_action(self, a: torch.Tensor) -> torch.Tensor:
        return torch.clamp(a, self.act_low, self.act_high)

    def forward(self, obs: torch.Tensor, deterministic: bool = False, sde_sample_freq: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.feature_net(obs)
        mu = self.mu_head(phi)
        if hasattr(self, 'dist'):
            if sde_sample_freq is not None:
                self.dist.maybe_resample(sde_sample_freq)
            action, logp, ent = self.dist.sample(mu, phi, deterministic)
        else:
            sigma = torch.exp(self.log_std).view(1, -1)
            std = sigma.expand_as(mu)
            if deterministic:
                action = mu
                ent = torch.zeros(mu.size(0), device=mu.device)
            else:
                action = mu + torch.randn_like(mu) * std
                ent = (0.5 + 0.5 * torch.log(2 * math.pi * std * std)).sum(dim=1)
            var = std * std
            logp = -0.5 * (((action - mu) ** 2) / (var + 1e-8) + 2 * torch.log(std + 1e-8) + math.log(2 * math.pi))
            logp = logp.sum(dim=1)
        v = self.v_head(phi).squeeze(-1)
        a_clipped = self.clamp_action(action)
        return a_clipped, logp, ent, v

# ==========================
# Rollout buffer
# ==========================
class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.acts = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.logp = torch.zeros(size, dtype=torch.float32, device=device)
        self.rews = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.vals = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0
        self.max_size = size

    def add(self, obs, act, logp, rew, done, val):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.logp[self.ptr] = logp
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = done
        self.vals[self.ptr] = val
        self.ptr += 1

    def reset(self):
        self.ptr = 0

# ==========================
# PPO with gSDE
# ==========================
class PPOAgent:
    def __init__(self, env: gym.Env, cfg: PPOConfig, device: torch.device):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.ac = ActorCritic(env.observation_space, env.action_space, cfg, device).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=cfg.lr)

    def collect_rollout(self) -> Tuple[RolloutBuffer, torch.Tensor, float]:
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        buf = RolloutBuffer(self.cfg.rollout_steps, obs_dim, act_dim, self.device)
        obs, _ = self.env.reset(seed=self.cfg.seed)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        ep_ret = 0.0
        ep_rets: List[float] = []
        for t in range(self.cfg.rollout_steps):
            with torch.no_grad():
                a, logp, _, v = self.ac(obs_t.unsqueeze(0), deterministic=False, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None))
            a = a.squeeze(0)
            next_obs, rew, terminated, truncated, _ = self.env.step(a.cpu().numpy())
            done = float(terminated or truncated)
            buf.add(obs_t, a, logp.squeeze(0), torch.tensor(rew, device=self.device), torch.tensor(done, device=self.device), v.squeeze(0))
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            ep_ret += float(rew)
            if terminated or truncated:
                ep_rets.append(ep_ret)
                obs, _ = self.env.reset()
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                ep_ret = 0.0
        # If last episode didn't finish during rollout, we still include its running return
        if ep_ret != 0.0:
            ep_rets.append(ep_ret)
        with torch.no_grad():
            _, _, _, last_val = self.ac(obs_t.unsqueeze(0), deterministic=True)
        mean_ret = float(np.mean(ep_rets)) if len(ep_rets) > 0 else float('nan')
        return buf, last_val.squeeze(0), mean_ret

    def compute_gae(self, buf: RolloutBuffer, last_val: torch.Tensor):
        # GAE-Lambda advantage computation
        adv = torch.zeros_like(buf.rews)
        last_gae = 0.0
        for t in reversed(range(buf.ptr)):
            next_nonterminal = 1.0 - buf.dones[t]
            next_value = last_val if t == buf.ptr - 1 else buf.vals[t+1]
            delta = buf.rews[t] + self.cfg.gamma * next_value * next_nonterminal - buf.vals[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + buf.vals[:buf.ptr]
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv[:buf.ptr], ret[:buf.ptr]

    def update(self, buf: RolloutBuffer, adv: torch.Tensor, ret: torch.Tensor):
        B = buf.ptr
        idxs = np.arange(B)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, B, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                mb_idx = idxs[start:end]
                mb_obs = buf.obs[mb_idx]
                mb_act = buf.acts[mb_idx]
                mb_logp_old = buf.logp[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]

                a, logp, ent, v = self.ac(mb_obs, deterministic=False, sde_sample_freq=(self.cfg.sde_sample_freq if self.cfg.use_gsde else None))

                # Policy loss with clipping
                ratio = torch.exp(logp - mb_logp_old)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss (MSE)
                v_loss = 0.5 * (mb_ret - v).pow(2).mean()

                # Entropy bonus
                ent_bonus = ent.mean()

                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent_bonus

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

    def train(self):
        obs_dim = self.env.observation_space.shape[0]
        print(f"Obs dim: {obs_dim}, Action dim: {self.env.action_space.shape[0]}")
        start = time.time()
        steps = 0
        while steps < self.cfg.total_steps:
            buf, last_val, mean_ret = self.collect_rollout()
            adv, ret = self.compute_gae(buf, last_val)
            self.update(buf, adv, ret)
            steps += buf.ptr
            fps = int(steps / (time.time() - start))
            # Report the current gSDE sigma (per action)
            if hasattr(self.ac, 'dist'):
                with torch.no_grad():
                    sig = self.ac.dist.sigma.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    sig = torch.exp(self.ac.log_std).detach().cpu().numpy()
            print(f"Steps: {steps:,} | ret: {mean_ret:.1f} | fps: {fps} | sigma: {np.round(sig, 3)}")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cfg = PPOConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = gym.make(cfg.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPOAgent(env, cfg, device)
    agent.train()
