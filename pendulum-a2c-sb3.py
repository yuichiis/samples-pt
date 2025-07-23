import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Callable, Dict, Tuple

# --- Step 1: ポリシーネットワーク (変更なし) ---
class MyReplicatedActorCriticPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, log_std_init: float, ortho_init: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
    def _get_distribution(self, obs: torch.Tensor) -> Normal:
        mean_action = self.policy_net(obs)
        action_std = torch.exp(self.log_std)
        return Normal(mean_action, action_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._get_distribution(obs)
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.rsample()
        
        value = self.value_net(obs).flatten() 
        log_prob = distribution.log_prob(action).sum(axis=-1)
        return action, value, log_prob

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._get_distribution(obs)
        log_prob = distribution.log_prob(action).sum(axis=-1)
        value = self.value_net(obs).flatten()
        entropy = distribution.entropy().sum(axis=-1)
        return value, log_prob, entropy

# --- Step 2: ロールアウトバッファ (変更なし) ---
class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, action_dim: int, gae_lambda: float, gamma: float):
        self.n_steps, self.n_envs, self.obs_dim, self.action_dim = n_steps, n_envs, obs_dim, action_dim
        self.gae_lambda, self.gamma = gae_lambda, gamma
        self.reset()

    def reset(self):
        self.observations = torch.zeros((self.n_steps, self.n_envs, self.obs_dim))
        self.actions = torch.zeros((self.n_steps, self.n_envs, self.action_dim))
        self.rewards = torch.zeros((self.n_steps, self.n_envs))
        self.dones = torch.zeros((self.n_steps, self.n_envs))
        self.values = torch.zeros((self.n_steps, self.n_envs))
        self.log_probs = torch.zeros((self.n_steps, self.n_envs))
        self.advantages = torch.zeros((self.n_steps, self.n_envs))
        self.returns = torch.zeros((self.n_steps, self.n_envs))
        self.pos = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.observations[self.pos] = torch.as_tensor(obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = torch.as_tensor(reward)
        self.dones[self.pos] = torch.as_tensor(done)
        self.values[self.pos] = value.detach()
        self.log_probs[self.pos] = log_prob.detach()
        self.pos += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor):
        last_advantage = 0
        last_value = last_value.detach()
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[t] = last_advantage
        self.returns = self.advantages + self.values

    def get(self) -> Dict[str, torch.Tensor]:
        return {"observations": self.observations.reshape(-1, self.obs_dim),
                "actions": self.actions.reshape(-1, self.action_dim),
                "advantages": self.advantages.flatten(),
                "returns": self.returns.flatten()}


# --- 新規追加: 評価関数 ---
def evaluate_policy(
    policy: MyReplicatedActorCriticPolicy,
    eval_env: gym.Env,
    n_eval_episodes: int,
    deterministic: bool = True
) -> Tuple[float, float]:
    """ポリシーを評価し、平均報酬と標準偏差を返す"""
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                # eval_envはn_envs=1のベクトル化環境なのでobsは(1, obs_dim)
                action, _, _ = policy(torch.as_tensor(obs).float(), deterministic=deterministic)
            # actionをnumpyに変換してenv.stepに渡す
            next_obs, reward, terminated, truncated, _ = eval_env.step(action.numpy())
            done_vec = terminated | truncated
            total_reward += reward[0] # 報酬は(1,)の配列なので[0]で値を取得
            obs = next_obs
            if done_vec[0]:
                done = True
        episode_rewards.append(total_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


# --- Step 3: A2Cクラス本体の修正 ---
class MyA2C:
    def __init__(
        self,
        env,
        policy_class,
        learning_rate: Callable[[float], float],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        policy_kwargs: Dict,
        seed: int,
        # 評価用の引数を追加
        eval_env: gym.Env = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
    ):
        self.env = env
        self.n_envs = env.num_envs
        self.learning_rate_schedule = learning_rate
        self.n_steps = n_steps
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        
        # 評価関連の属性を追加
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_timestep = 0

        torch.manual_seed(seed)
        np.random.seed(seed)
        
        obs_dim = self.env.observation_space.shape[1]
        action_dim = self.env.action_space.shape[1]

        self.policy = policy_class(obs_dim, action_dim, **policy_kwargs)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate_schedule(1.0))
        
        self.rollout_buffer = RolloutBuffer(n_steps, self.n_envs, obs_dim, action_dim, gae_lambda, gamma)
        
        self.num_timesteps = 0
        self._n_updates = 0
        self._last_obs = None

    def _collect_rollouts(self):
        if self._last_obs is None: self._last_obs, _ = self.env.reset(seed=42)
        self.rollout_buffer.reset()
        for _ in range(self.n_steps):
            with torch.no_grad():
                action, value, log_prob = self.policy(torch.as_tensor(self._last_obs).float())
            next_obs, reward, term, trunc, _ = self.env.step(action.numpy())
            done = term | trunc
            self.rollout_buffer.add(self._last_obs, action, reward, done, value, log_prob)
            self._last_obs = next_obs
            self.num_timesteps += self.n_envs
        with torch.no_grad():
            _, last_value, _ = self.policy(torch.as_tensor(self._last_obs).float())
        self.rollout_buffer.compute_returns_and_advantages(last_value, torch.as_tensor(done).float())

    def train(self):
        self._n_updates += 1
        progress = self.num_timesteps / self.total_timesteps
        new_lr = self.learning_rate_schedule(1.0 - progress)
        for pg in self.optimizer.param_groups: pg["lr"] = new_lr

        data = self.rollout_buffer.get()
        values, log_prob, entropy = self.policy.evaluate_actions(data["observations"].float(), data["actions"].float())
        
        policy_loss = -(data["advantages"] * log_prob).mean()
        value_loss = F.mse_loss(data["returns"], values)
        entropy_loss = -torch.mean(entropy)
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self._n_updates % 100 == 0:
            print(f"Timesteps: {self.num_timesteps}, LR: {new_lr:.2e}, Loss: {loss.item():.4f}")

    def learn(self, total_timesteps: int):
        self.total_timesteps = total_timesteps
        while self.num_timesteps < self.total_timesteps:
            self._collect_rollouts()
            self.train()

            # --- 評価ロジックを追加 ---
            if self.eval_env is not None and self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
                self.last_eval_timestep = self.num_timesteps
                mean_reward, std_reward = evaluate_policy(
                    self.policy, self.eval_env, self.n_eval_episodes
                )
                print("-" * 60)
                print(f"Evaluation at timestep {self.num_timesteps}")
                print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} over {self.n_eval_episodes} episodes")
                print("-" * 60)


# --- 実行部分の修正 ---
if __name__ == "__main__":
    hyperparams = {
        "n_steps": 8, "gamma": 0.9, "gae_lambda": 0.9, "ent_coef": 0.0,
        "vf_coef": 0.4, "max_grad_norm": 0.5,
        "policy_kwargs": dict(log_std_init=-2, ortho_init=False),
    }
    N_ENVS, TOTAL_TIMESTEPS = 8, 1_000_000
    ENV_ID, SEED = "Pendulum-v1", 42

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        return lambda progress_remaining: progress_remaining * initial_value

    # 学習用と評価用の環境を準備
    env = gym.make_vec(ENV_ID, num_envs=N_ENVS)
    eval_env = gym.make_vec(ENV_ID, num_envs=1)

    # EvalCallbackのeval_freqを再現
    eval_freq = max(10000 // N_ENVS, 1) * N_ENVS

    model = MyA2C(
        env=env,
        policy_class=MyReplicatedActorCriticPolicy,
        learning_rate=linear_schedule(7e-4),
        eval_env=eval_env,          # 評価環境を渡す
        eval_freq=eval_freq,       # 評価頻度を渡す
        **hyperparams,
        seed=SEED,
    )

    print("自作A2Cクラスで学習を開始します...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("学習が完了しました。")

    env.close()
    eval_env.close()