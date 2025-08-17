import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim
from collections import deque
import time
import imageio # GIF保存のためにインポート

# rl-zoo3のPendulum-v1用PPOハイパーパラメータを参考
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
hyperparams = {
    'total_timesteps': 600000,
    'n_steps': 1024,
    'gamma': 0.9,
    'learning_rate': 0.001,
    'gae_lambda': 0.95,
    'batch_size': 64,
    'n_epochs': 10,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'net_arch': [64, 64],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

TARGET_SCORE = -250

class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, net_arch):
        super(ActorCritic, self).__init__()
        
        shared_layers = []
        last_dim = observation_dim
        for layer_size in net_arch:
            shared_layers.append(nn.Linear(last_dim, layer_size))
            shared_layers.append(nn.Tanh())
            last_dim = layer_size
        self.shared_net = nn.Sequential(*shared_layers)

        self.policy_net = nn.Linear(last_dim, action_dim)
        self.value_net = nn.Linear(last_dim, 1)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        shared_features = self.shared_net(obs)
        action_mean = self.policy_net(shared_features)
        value = self.value_net(shared_features)
        
        action_std = torch.exp(self.log_std)
        distribution = Normal(action_mean, action_std)
        
        return distribution, value

    def get_action_and_value(self, obs, action=None):
        distribution, value = self.forward(obs)
        
        if action is None:
            action = distribution.sample()
        
        log_prob = distribution.log_prob(action).sum(axis=-1)
        entropy = distribution.entropy().sum(axis=-1)
        
        return action, log_prob, entropy, value.flatten()

class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, device, gamma, gae_lambda):
        self.n_steps, self.obs_dim, self.action_dim = n_steps, obs_dim, action_dim
        self.device, self.gamma, self.gae_lambda = device, gamma, gae_lambda
        self.reset()

    def reset(self):
        self.observations = torch.zeros((self.n_steps, self.obs_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.n_steps, self.action_dim), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device)
        self.dones = torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device)
        self.log_probs = torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device)
        self.values = torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device)
        self.advantages = torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device)
        self.returns = torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device)
        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        if self.ptr < self.n_steps:
            self.observations[self.ptr], self.actions[self.ptr] = obs, action
            self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32).to(self.device)
            self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32).to(self.device)
            self.log_probs[self.ptr], self.values[self.ptr] = log_prob, value
            self.ptr += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        last_advantage = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal, next_values = 1.0 - last_done, last_value
            else:
                next_non_terminal, next_values = 1.0 - self.dones[t + 1], self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[t] = last_advantage
        self.returns = self.advantages + self.values

    def get(self, batch_size):
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            yield tuple(data[indices[start:start+batch_size]] for data in (
                self.observations, self.actions, self.log_probs, self.advantages, self.returns
            ))

class PPO:
    def __init__(self, env, **kwargs):
        self.env = env
        self.device = kwargs['device']
        
        self.n_steps, self.gamma, self.learning_rate = kwargs['n_steps'], kwargs['gamma'], kwargs['learning_rate']
        self.gae_lambda, self.batch_size, self.n_epochs = kwargs['gae_lambda'], kwargs['batch_size'], kwargs['n_epochs']
        self.clip_range, self.ent_coef, self.vf_coef = kwargs['clip_range'], kwargs['ent_coef'], kwargs['vf_coef']
        self.max_grad_norm = kwargs['max_grad_norm']

        obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.action_low = torch.tensor(env.action_space.low, device=self.device)
        self.action_high = torch.tensor(env.action_space.high, device=self.device)

        self.policy = ActorCritic(obs_dim, action_dim, kwargs['net_arch']).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(self.n_steps, obs_dim, action_dim, self.device, self.gamma, self.gae_lambda)
        
        self.num_timesteps = 0
        self.last_obs, _ = self.env.reset()
        self.last_obs = torch.tensor(self.last_obs, dtype=torch.float32).to(self.device)
        self.last_done = False

    def collect_rollouts(self):
        self.policy.eval()
        for _ in range(self.n_steps):
            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(self.last_obs)
            clipped_action = torch.clamp(action, self.action_low, self.action_high)
            next_obs, reward, terminated, truncated, _ = self.env.step(clipped_action.cpu().numpy())
            self.last_done = terminated or truncated
            self.num_timesteps += 1
            self.buffer.add(self.last_obs, action, reward, self.last_done, log_prob, value)
            self.last_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            if self.last_done:
                self.last_obs, _ = self.env.reset()
                self.last_obs = torch.tensor(self.last_obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, _, _, last_value = self.policy.get_action_and_value(self.last_obs)
        self.buffer.compute_returns_and_advantages(last_value, self.last_done)

    def train(self):
        self.policy.train()
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_losses, value_losses, entropy_losses = [], [], []
        for _ in range(self.n_epochs):
            for batch in self.buffer.get(self.batch_size):
                obs, actions, old_log_probs, batch_advantages, batch_returns = batch
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(obs, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                ).mean()
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                entropy_loss = -torch.mean(entropy)
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        self.buffer.reset()
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_losses)
    
    def learn(self, total_timesteps):
        start_time = time.time()
        episode_rewards = deque(maxlen=100)
        while self.num_timesteps < total_timesteps:
            self.collect_rollouts()
            pg_loss, v_loss, ent_loss = self.train()
            if hasattr(self.env, 'return_queue'):
                while len(self.env.return_queue) > 0:
                    episode_rewards.append(self.env.return_queue.popleft())
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                print(f"Timesteps: {self.num_timesteps}/{total_timesteps} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Policy Loss: {pg_loss:.4f} | "
                      f"Value Loss: {v_loss:.2f} | "
                      f"Entropy: {-ent_loss:.4f} | "  # Note: Displaying positive entropy
                      f"Time: {time.time() - start_time:.2f}s")
                
                if mean_reward >= TARGET_SCORE:
                    print(f"\n目標スコア {TARGET_SCORE} を達成しました！学習を終了します。")
                    break

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

if __name__ == "__main__":
    print(f"Using device: {hyperparams['device']}")
    
    env = gym.make("Pendulum-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = PPO(env, **hyperparams)
    model_file = 'pendulum-ppo.pth'
    if os.path.isfile(model_file):
        agent.load(model_file)
    else:
        agent.learn(total_timesteps=hyperparams['total_timesteps'])
        agent.save(model_file)

    # === ここからが修正・追加箇所 ===
    print("\n--- Testing Learned Policy ---")
    
    # 描画モードを "rgb_array" にして、フレームデータを取得できるようにする
    test_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    frames = []
    test_episode_rewards = []
    num_test_episodes = 5 # テストするエピソード数

    for episode in range(num_test_episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 現在のフレームをキャプチャ
            frame = test_env.render()
            frames.append(frame)

            # エージェントに行動を選択させる
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(hyperparams['device'])
                action, _, _, _ = agent.policy.get_action_and_value(obs_tensor)
                clipped_action = torch.clamp(action, agent.action_low, agent.action_high)

            # 環境を1ステップ進める
            obs, reward, terminated, truncated, _ = test_env.step(clipped_action.cpu().numpy())
            done = terminated or truncated
            episode_reward += reward
        
        test_episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

    test_env.close()

    # テスト結果の平均スコアを表示
    if test_episode_rewards:
        print(f"\nAverage Test Reward over {num_test_episodes} episodes: {np.mean(test_episode_rewards):.2f}")

    # キャプチャしたフレームをGIFとして保存
    if frames:
        gif_file = 'pendulum-ppo.gif'
        print(f"\nSaving test animation to '{gif_file}'...")
        imageio.mimsave(gif_file, frames, fps=30)
        print("Save complete.")
    # === ここまで ===