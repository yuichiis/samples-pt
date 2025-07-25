import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Callable, Dict, Tuple, Optional


# SDEをSB3のgSDE方式に近づける修正案
class MyGSDELikeActorCriticPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        log_std_init: float,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        # ... 他の引数
    ):
        super().__init__()
        # ... (共通ネットワークの定義は同じ) ...
        self.policy_net_base = nn.Sequential(...)
        self.value_net_base = nn.Sequential(...)
        self.action_net = nn.Linear(64, action_dim) # 平均を出すヘッド
        self.value_net = nn.Linear(64, 1)

        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.latent_sde_dim = 64 # 特徴量の次元

        if self.use_sde:
            # 状態に依存しない学習可能なlog_stdパラメータ
            self.log_std = nn.Parameter(torch.ones(self.latent_sde_dim, action_dim) * log_std_init)
            self.exploration_mat = None # ノイズ射影行列
            self.exploration_noise = None # SB3のSDEではこれは使わない
            # 最初のサンプリング
            self.sample_exploration_matrix()
        else:
            # 通常のSDEでない場合
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
            
    def sample_exploration_matrix(self):
        """ノイズ射影行列をサンプリングする"""
        # 正規分布からサンプリング
        std = torch.exp(self.log_std)
        self.exploration_mat = torch.randn_like(self.log_std) * std

    def get_sde_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        """gSDEのノイズを計算する"""
        # 毎ステップ、特徴量と行列を掛ける
        return torch.mm(latent_sde, self.exploration_mat)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        latent_pi = self.policy_net_base(obs)
        latent_vf = self.value_net_base(obs)
        value = self.value_net(latent_vf).flatten()
        mean_actions = self.action_net(latent_pi)
        
        if deterministic:
            action = mean_actions
            # 決定論的な場合でも、分布の計算は必要
            if self.use_sde:
                # gSDEの分散は状態に依存する
                variance = torch.mm(latent_pi**2, torch.exp(self.log_std)**2)
                distribution = Normal(mean_actions, torch.sqrt(variance + 1e-6))
            else:
                distribution = Normal(mean_actions, torch.exp(self.log_std))
            log_prob = distribution.log_prob(action).sum(axis=-1)

        else: # 探索的な行動
            if self.use_sde:
                # gSDEの探索時の実装
                # sde_sample_freq > 0 の場合、その頻度でノイズ行列を再サンプリング
                if self.sde_sample_freq > 0 and self.training and np.random.randint(self.sde_sample_freq) == 0:
                    self.sample_exploration_matrix()
                
                noise = self.get_sde_noise(latent_pi)
                action = mean_actions + noise
                
                # log_prob計算用の分布を構築
                variance = torch.mm(latent_pi**2, torch.exp(self.log_std)**2)
                distribution = Normal(mean_actions, torch.sqrt(variance + 1e-6))
                log_prob = distribution.log_prob(action).sum(axis=-1)
            else:
                # 通常のサンプリング
                distribution = Normal(mean_actions, torch.exp(self.log_std))
                action = distribution.rsample()
                log_prob = distribution.log_prob(action).sum(axis=-1)

        return action, value, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi = self.policy_net_base(obs)
        latent_vf = self.value_net_base(obs)
        mean_actions = self.action_net(latent_pi)
        value = self.value_net(latent_vf).flatten()

        if self.use_sde:
            variance = torch.mm(latent_pi**2, torch.exp(self.log_std)**2)
            distribution = Normal(mean_actions, torch.sqrt(variance + 1e-6))
        else:
            distribution = Normal(mean_actions, torch.exp(self.log_std))
            
        log_prob = distribution.log_prob(action).sum(axis=-1)
        entropy = distribution.entropy().sum(axis=-1)
        return value, log_prob, entropy

## --- Step 1: ポリシーネットワークをSDE対応に大幅修正 ---
#class MyReplicatedActorCriticPolicy(nn.Module):
#    """
#    SDE (State-Dependent Exploration) に対応したActor-Criticポリシー。
#    """
#    def __init__(
#        self,
#        obs_dim: int,
#        action_dim: int,
#        log_std_init: float,
#        # SDE関連の引数を追加
#        use_sde: bool = False,
#        sde_sample_freq: int = -1, # -1は毎ステップサンプリングを意味する
#        ortho_init: bool = False,
#    ):
#        super().__init__()
#        self.obs_dim = obs_dim
#        self.action_dim = action_dim
#        self.use_sde = use_sde
#        self.sde_sample_freq = sde_sample_freq
#        
#        # 共有ネットワークや個別のネットワークを定義
#        # SB3ではMlpExtractorがこの役割を担うが、今回は直接定義
#        # Actor (方策) と Critic (価値) は同じ構造のネットワークとする
#        self.policy_net_base = nn.Sequential(
#            nn.Linear(obs_dim, 64), nn.Tanh(),
#            nn.Linear(64, 64), nn.Tanh()
#        )
#        self.value_net_base = nn.Sequential(
#            nn.Linear(obs_dim, 64), nn.Tanh(),
#            nn.Linear(64, 64), nn.Tanh()
#        )
#        # 最後の線形層 (ヘッド)
#        self.action_net = nn.Linear(64, action_dim)
#        self.value_net = nn.Linear(64, 1)
#
#        if self.use_sde:
#            # SDE用のノイズ射影ネットワーク
#            # 特徴量と同じ次元(64)から行動次元への射影
#            self.sde_net = nn.Linear(64, action_dim)
#            # SDEで使うための、使い回すノイズ(epsilon)
#            self.exploration_noise = None
#        else:
#            # SDEを使わない場合: 状態に依存しない学習可能なlog_std
#            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
#
#    def _get_std(self, latent_sde: torch.Tensor) -> torch.Tensor:
#        """SDE用の標準偏差を計算する"""
#        # SB3ではlog_stdを-20から2の範囲にクリップする
#        log_std = self.sde_net(latent_sde)
#        log_std = torch.clamp(log_std, -20, 2)
#        return torch.exp(log_std)
#
#    def _get_distribution(
#        self,
#        mean_actions: torch.Tensor,
#        latent_sde: Optional[torch.Tensor] = None
#    ) -> Normal:
#        """観測から行動の確率分布を構築する"""
#        if self.use_sde:
#            # SDEの場合、標準偏差は状態に依存する
#            std = self._get_std(latent_sde)
#        else:
#            # SDEでない場合、標準偏差は状態に依存しない
#            std = torch.exp(self.log_std)
#        return Normal(mean_actions, std)
#
#    def forward(
#        self,
#        obs: torch.Tensor,
#        deterministic: bool = False
#    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#        """行動を決定し、その価値と対数確率を返す"""
#        # 特徴抽出
#        latent_pi = self.policy_net_base(obs)
#        latent_vf = self.value_net_base(obs)
#        # 価値を計算
#        value = self.value_net(latent_vf).flatten()
#        # 行動の平均を計算
#        mean_actions = self.action_net(latent_pi)
#        
#        # 行動分布を構築
#        distribution = self._get_distribution(mean_actions, latent_sde=latent_pi)
#
#        if deterministic:
#            action = mean_actions
#        else:
#            if self.use_sde:
#                # SDEのサンプリング
#                # SB3の簡易再現: sde_sample_freqごとにノイズを再サンプリング
#                if self.exploration_noise is None or self.sde_sample_freq > 0 and self.training and np.random.randint(self.sde_sample_freq) == 0:
#                    self.exploration_noise = torch.randn_like(mean_actions)
#                # 行動 = 平均 + std * ε
#                action = mean_actions + distribution.stddev * self.exploration_noise
#            else:
#                # 通常のサンプリング
#                action = distribution.rsample()
#
#        log_prob = distribution.log_prob(action).sum(axis=-1)
#        return action, value, log_prob
#
#    def evaluate_actions(
#        self,
#        obs: torch.Tensor,
#        action: torch.Tensor
#    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#        """与えられた観測と行動に対して、価値、対数確率、エントロピーを評価する"""
#        latent_pi = self.policy_net_base(obs)
#        latent_vf = self.value_net_base(obs)
#        mean_actions = self.action_net(latent_pi)
#        value = self.value_net(latent_vf).flatten()
#
#        distribution = self._get_distribution(mean_actions, latent_sde=latent_pi)
#        log_prob = distribution.log_prob(action).sum(axis=-1)
#        entropy = distribution.entropy().sum(axis=-1)
#        return value, log_prob, entropy

# --- ロールアウトバッファ (変更なし) ---
class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, action_dim, gae_lambda, gamma):
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
    def compute_returns_and_advantages(self, last_value, last_done):
        last_advantage = 0
        last_value = last_value.detach()
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal, next_values = 1.0 - last_done, last_value
            else:
                next_non_terminal, next_values = 1.0 - self.dones[t + 1], self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[t] = last_advantage
        self.returns = self.advantages + self.values
    def get(self): return {"observations": self.observations.reshape(-1, self.obs_dim), "actions": self.actions.reshape(-1, self.action_dim), "advantages": self.advantages.flatten(), "returns": self.returns.flatten()}


# --- 評価関数 (変更なし) ---
def evaluate_policy(policy, eval_env, n_eval_episodes, deterministic=True):
    rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        done, total_reward = False, 0.0
        while not done:
            with torch.no_grad():
                action, _, _ = policy(torch.as_tensor(obs).float(), deterministic=deterministic)
            next_obs, reward, term, trunc, _ = eval_env.step(action.numpy())
            done = (term | trunc)[0]
            total_reward += reward[0]
            obs = next_obs
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


# --- A2Cクラス本体の修正 (SDE関連の引数を渡す) ---
class MyA2C:
    def __init__(
        self, env, policy_class, learning_rate, n_steps, gamma, gae_lambda,
        ent_coef, vf_coef, max_grad_norm, policy_kwargs, seed,
        use_sde: bool = False, # SDEフラグを追加
        sde_sample_freq: int = -1,
        eval_env=None, eval_freq=10000, n_eval_episodes=5,
    ):
        self.env, self.n_envs = env, env.num_envs
        self.lr_schedule, self.n_steps = learning_rate, n_steps
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.eval_env, self.eval_freq, self.n_eval_episodes = eval_env, eval_freq, n_eval_episodes
        self.last_eval_timestep = 0

        torch.manual_seed(seed), np.random.seed(seed)
        
        obs_dim, action_dim = self.env.observation_space.shape[1], self.env.action_space.shape[1]

        # policy_kwargsにSDE関連の引数を追加して渡す
        policy_kwargs.update({"use_sde": use_sde, "sde_sample_freq": sde_sample_freq})
        self.policy = policy_class(obs_dim, action_dim, **policy_kwargs)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_schedule(1.0))
        
        self.buffer = RolloutBuffer(n_steps, self.n_envs, obs_dim, action_dim, gae_lambda, gamma)
        self.num_timesteps, self._n_updates, self._last_obs = 0, 0, None

    def _collect_rollouts(self):
        if self._last_obs is None: self._last_obs, _ = self.env.reset(seed=42)
        self.buffer.reset()
        # ポリシーを学習モードに設定 (SDEのサンプリング頻度に影響)
        self.policy.train()
        for _ in range(self.n_steps):
            with torch.no_grad():
                action, value, log_prob = self.policy(torch.as_tensor(self._last_obs).float())
            next_obs, reward, term, trunc, _ = self.env.step(action.numpy())
            done = term | trunc
            self.buffer.add(self._last_obs, action, reward, done, value, log_prob)
            self._last_obs, self.num_timesteps = next_obs, self.num_timesteps + self.n_envs
        with torch.no_grad():
            # ポリシーを評価モードに設定
            self.policy.eval()
            _, last_value, _ = self.policy(torch.as_tensor(self._last_obs).float())
        self.buffer.compute_returns_and_advantages(last_value, torch.as_tensor(done).float())

    def train(self):
        self._n_updates += 1
        progress = self.num_timesteps / self.total_timesteps
        new_lr = self.lr_schedule(1.0 - progress)
        for pg in self.optimizer.param_groups: pg["lr"] = new_lr

        data = self.buffer.get()
        values, log_prob, entropy = self.policy.evaluate_actions(data["observations"].float(), data["actions"].float())
        
        loss = -(data["advantages"] * log_prob).mean() + self.vf_coef * F.mse_loss(data["returns"], values) - self.ent_coef * torch.mean(entropy)
        
        self.optimizer.zero_grad(), loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self._n_updates % 100 == 0: print(f"Timesteps: {self.num_timesteps}, LR: {new_lr:.2e}, Loss: {loss.item():.4f}")

    def learn(self, total_timesteps: int):
        self.total_timesteps = total_timesteps
        while self.num_timesteps < total_timesteps:
            self._collect_rollouts()
            self.train()

            if self.eval_env is not None and self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
                self.last_eval_timestep = self.num_timesteps
                # 評価時はポリシーを評価モードに設定
                self.policy.eval()
                mean_reward, std_reward = evaluate_policy(self.policy, self.eval_env, self.n_eval_episodes)
                print("-" * 60)
                print(f"Eval at timestep {self.num_timesteps}: Mean reward {mean_reward:.2f} +/- {std_reward:.2f}")
                print("-" * 60)


# --- 実行部分の修正 (use_sde=Trueを追加) ---
if __name__ == "__main__":
    hyperparams = {
        "n_steps": 8, "gamma": 0.9, "gae_lambda": 0.9, "ent_coef": 0.0,
        "vf_coef": 0.4, "max_grad_norm": 0.5,
        "use_sde": True, # ★ SDEを有効化 ★
        "policy_kwargs": dict(log_std_init=-2, ortho_init=False),
    }
    N_ENVS, TOTAL_TIMESTEPS = 8, 1_000_000
    ENV_ID, SEED = "Pendulum-v1", 42

    def linear_schedule(initial_value): return lambda p: p * initial_value
    
    env = gym.make_vec(ENV_ID, num_envs=N_ENVS)
    eval_env = gym.make_vec(ENV_ID, num_envs=1)
    eval_freq = max(10000 // N_ENVS, 1) * N_ENVS

    model = MyA2C(
        env=env, policy_class=MyGSDELikeActorCriticPolicy, #MyReplicatedActorCriticPolicy,
        learning_rate=linear_schedule(7e-4),
        eval_env=eval_env, eval_freq=eval_freq, **hyperparams, seed=SEED,
    )

    print("自作A2Cクラス(SDE対応)で学習を開始します...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("学習が完了しました。")
    env.close(), eval_env.close()