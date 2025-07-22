# 必要なライブラリをインストールしてください
# pip install torch torchvision torchaudio numpy gymnasium imageio

import numpy as np
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
import imageio

# --- 正規化ラッパーとヘルパークラス (変更なし、Numpyベースなのでそのまま使えます) ---
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
    def update(self, x):
        batch_mean, batch_var, batch_count = np.mean(x, axis=0), np.var(x, axis=0), x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class NormalizeWrapper(gym.Wrapper):
    def __init__(self, env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99):
        super().__init__(env)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = 1e-8
        if self.norm_obs:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        if self.norm_reward:
            self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(())
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        unnormalized_reward = reward
        if self.norm_reward:
            self.returns = self.returns * self.gamma + reward
            self.ret_rms.update(np.array([self.returns]))
            reward = reward / (np.sqrt(self.ret_rms.var) + self.epsilon)
            reward = np.clip(reward, -self.clip_reward, self.clip_reward)
        if self.norm_obs:
            self.obs_rms.update(np.array([obs]))
            obs = (obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + self.epsilon)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        if terminated or truncated:
            self.returns = 0.0
        info['unnormalized_reward'] = unnormalized_reward
        return obs, reward, terminated, truncated, info
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.norm_obs:
            self.obs_rms.update(np.array([obs]))
            obs = (obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + self.epsilon)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        self.returns = 0.0
        return obs, info

# --- PyTorch版 Actor-Criticモデル ---
class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        # 共通ネットワーク
        self.common_net = nn.Sequential(
            nn.Linear(input_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Actorヘッド
        self.mu_head = nn.Linear(64, action_dim)
        # Criticヘッド
        self.value_head = nn.Linear(64, 1)
        # log_std (学習可能なパラメータとして定義)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0)

    def forward(self, x):
        features = self.common_net(x)
        mu = torch.tanh(self.mu_head(features)) # 平均値はtanhで-1~1に
        value = self.value_head(features)
        
        # log_stdをクリップして、expでstdに変換
        std = torch.exp(self.log_std.clamp(-5, 2))
        
        return mu, std, value

# === 行動選択関数 (PyTorch版) ===
def get_action(actor_critic, state, action_bound, device):
    with torch.no_grad():
        state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
        mu_normalized, std, _ = actor_critic(state_tensor)
        
        dist = Normal(mu_normalized * action_bound, std)
        action = dist.sample()
        
        return action.cpu().numpy().flatten()

def get_best_action(actor_critic, state, action_bound, device):
    with torch.no_grad():
        state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
        mu_normalized, _, _ = actor_critic(state_tensor)
        action = mu_normalized * action_bound
        return action.cpu().numpy().flatten()


# === 学習関数 (PyTorch版) ===
def update(actor_critic, optimizer, experiences, gamma, gae_lambda, value_loss_weight,
           entropy_weight, max_grad_norm, standardize, action_bound, device):
    
    states = np.asarray([e["state"] for e in experiences], dtype=np.float32)
    actions = np.asarray([e["action"] for e in experiences], dtype=np.float32)
    rewards = np.asarray([e["reward"] for e in experiences], dtype=np.float32)
    dones = np.asarray([e["done"] for e in experiences], dtype=np.bool_)
    n_states = np.asarray([e["n_state"] for e in experiences], dtype=np.float32)

    # Numpy -> Torch Tensor
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    dones_t = torch.from_numpy(dones).to(device)
    n_states_t = torch.from_numpy(n_states).to(device)
    
    # GAE計算
    with torch.no_grad():
        _, _, values = actor_critic(states_t)
        _, _, next_values = actor_critic(n_states_t)
        values = values.flatten()
        next_values = next_values.flatten()
        
        advantages = torch.zeros_like(rewards_t)
        last_gae_lam = 0
        for t in reversed(range(len(rewards_t))):
            next_non_terminal = 1.0 - dones_t[t].float()
            delta = rewards_t[t] + gamma * next_values[t] * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * last_gae_lam * next_non_terminal
        
        returns = advantages + values
    
    # Advantage正規化
    if standardize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # モデルのフォワードパス
    mu_normalized, std, pred_values = actor_critic(states_t)
    
    dist = Normal(mu_normalized.squeeze() * action_bound, std)
    entropy = dist.entropy().mean()
    log_prob = dist.log_prob(actions_t.squeeze()).unsqueeze(1)
    
    # 損失計算
    policy_loss = -(log_prob * advantages.detach().unsqueeze(1)).mean()
    value_loss = nn.functional.huber_loss(pred_values.squeeze(), returns.detach())
    
    loss = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy

    # 勾配更新
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
    optimizer.step()
    
    return policy_loss.item(), value_loss.item(), entropy.item()

# === 学習ループ (PyTorch版) ===
def train(env, actor_critic, action_bound, device):
    standardize = False
    total_timesteps = 1_000_000
    n_steps = 8
    gamma = 0.9
    gae_lambda = 0.9
    
    lr_initial = 7e-4
    lr_final = 1e-6
    
    max_grad_norm = 0.5
    value_loss_weight = 0.5
    entropy_weight = 0.0

    print('A2C for Pendulum-v1 (PyTorch - RL Zoo Config)')
    print(f'standardize = {standardize}, total_timesteps = {total_timesteps}, n_steps = {n_steps}')
    
    # オプティマイザ (SB3のA2Cのデフォルトに合わせてalpha=0.99, eps=1e-5)
    optimizer = RMSprop(actor_critic.parameters(), lr=lr_initial, alpha=0.99, eps=1e-5)

    print("--- 学習開始 ---")
    start_time = time.time()
    all_rewards, all_episode_steps = [], []
    all_actor_losses, all_critic_losses, all_entropies = [], [], []
    
    experiences = []
    episode_count, episode_reward_sum, episode_step_count = 0, 0, 0
    state, _ = env.reset()

    for global_step in range(1, total_timesteps + 1):
        # 学習率の線形減衰
        frac = 1.0 - (global_step - 1.0) / total_timesteps
        lr_now = frac * (lr_initial - lr_final) + lr_final
        optimizer.param_groups[0]['lr'] = lr_now

        episode_step_count += 1
        action = get_action(actor_critic, state, action_bound, device)
        clipped_action = np.clip(action, -action_bound, action_bound)
        
        n_state, reward, terminated, truncated, info = env.step(clipped_action)
        episode_reward_sum += info['unnormalized_reward']

        experiences.append({
            "state": state, "action": clipped_action, "reward": reward,
            "n_state": n_state, "done": terminated or truncated
        })
        state = n_state

        if terminated or truncated:
            all_rewards.append(episode_reward_sum)
            all_episode_steps.append(episode_step_count)
            episode_count += 1
            episode_reward_sum, episode_step_count = 0, 0
            state, _ = env.reset()

        if len(experiences) >= n_steps:
            actor_loss, critic_loss, entropy = update(
                actor_critic, optimizer, experiences, gamma, gae_lambda, value_loss_weight,
                entropy_weight, max_grad_norm, standardize, action_bound, device
            )
            all_actor_losses.append(actor_loss)
            all_critic_losses.append(critic_loss)
            all_entropies.append(entropy)
            experiences = []

        if (global_step % (n_steps * 200) == 0) or global_step == total_timesteps:
            avg_reward = np.mean(all_rewards[-20:]) if all_rewards else -1600
            print(f'St:{global_step//1000}k | Ep:{episode_count} | AvgRwd:{avg_reward:.1f} | '
                  f'ActorL:{np.mean(all_actor_losses[-200:]):.3f} | CriticL:{np.mean(all_critic_losses[-200:]):.3f} | '
                  f'Entr:{np.mean(all_entropies[-200:]):.3f} | LR:{lr_now:.2e}')
            
            if avg_reward > -200 and len(all_rewards) > 20:
                print(f"環境がクリアされました！ (平均報酬: {avg_reward})")
                break
    
    print("--- 学習終了 ---")
    print(f"実行時間: {time.time() - start_time:.4f}秒")
    # プロット処理は省略...

# === メイン処理 (PyTorch版) ===
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_raw = gym.make("Pendulum-v1")
    env = NormalizeWrapper(env_raw, gamma=0.9)

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor_critic = ActorCritic(obs_shape, action_dim).to(device)
    print("--- Actor-Critic Model (PyTorch) ---")
    print(actor_critic)

    model_file = 'pendulum-a2c-pytorch.pth'
    
    if os.path.isfile(model_file):
        print(f"学習済みモデル {model_file} を読み込みます。")
        actor_critic.load_state_dict(torch.load(model_file))
    else:
        train(env, actor_critic, action_bound, device)
        print(f"学習済みモデルを保存します。")
        torch.save(actor_critic.state_dict(), model_file)
    env.close()

    # テスト実行は省略...