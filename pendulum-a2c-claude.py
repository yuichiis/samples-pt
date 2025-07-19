import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # 共通の特徴抽出層
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor（方策）ネットワーク
        # Pendulumは連続制御なので、平均と標準偏差を出力
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic（価値）ネットワーク
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        shared_features = self.shared_layers(state)
        
        # Actor出力
        action_mean = self.actor_mean(shared_features)
        action_std = F.softplus(self.actor_std(shared_features)) + 1e-5
        
        # Critic出力
        value = self.critic(shared_features)
        
        return action_mean, action_std, value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.action_dim = action_dim
        
        # ネットワークの初期化
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # 経験を保存するためのバッファ
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std, value = self.actor_critic(state)
        
        # 正規分布から行動をサンプリング
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Pendulumの行動範囲は[-2, 2]
        action = torch.clamp(action, -2.0, 2.0)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def store_transition(self, state, action, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_returns(self, next_value, done):
        returns = []
        R = next_value if not done else 0
        
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def update(self, next_state, done):
        # 次の状態の価値を計算
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, next_value = self.actor_critic(next_state)
            next_value = next_value.cpu().numpy()[0][0]
        
        # リターンを計算
        returns = self.compute_returns(next_value, done)
        
        # テンソルに変換
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        
        # 現在の方策での行動確率と価値を再計算
        action_means, action_stds, current_values = self.actor_critic(states)
        
        # アドバンテージを計算
        advantages = returns - current_values.squeeze()
        
        # 新しい行動確率を計算
        dist = torch.distributions.Normal(action_means, action_stds)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # 損失を計算
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(current_values.squeeze(), returns)
        
        # エントロピーボーナス（探索を促進）
        entropy = dist.entropy().sum(dim=-1).mean()
        
        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # 勾配更新
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()
        
        # バッファをクリア
        self.clear_buffer()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()
    
    def clear_buffer(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

def train_a2c():
    # 環境の設定
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # エージェントの初期化
    agent = A2CAgent(state_dim, action_dim, device=device)
    
    # 学習パラメータ
    num_episodes = 500
    update_interval = 20  # 20ステップごとに更新
    
    # 記録用
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    entropies = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            # 行動選択
            action, log_prob, value = agent.select_action(state)
            
            # 環境での行動実行
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 経験を保存
            agent.store_transition(state, action, reward, log_prob, value)
            
            episode_reward += reward
            step += 1
            
            # 定期的に更新
            if step % update_interval == 0 or done:
                actor_loss, critic_loss, entropy = agent.update(next_state, done)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropies.append(entropy)
            
            if done:
                break
                
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # 進捗表示
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    env.close()
    
    # 結果の可視化
    plt.figure(figsize=(15, 5))
    
    # 報酬の推移
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), 'r-')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # 損失の推移
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses, label='Actor Loss', alpha=0.7)
    plt.plot(critic_losses, label='Critic Loss', alpha=0.7)
    plt.title('Training Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # エントロピーの推移
    plt.subplot(1, 3, 3)
    plt.plot(entropies)
    plt.title('Policy Entropy')
    plt.xlabel('Update Step')
    plt.ylabel('Entropy')
    
    plt.tight_layout()
    plt.show()
    
    return agent

def test_agent(agent, num_episodes=5):
    """学習済みエージェントのテスト"""
    env = gym.make('Pendulum-v1', render_mode='human')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # 学習実行
    print("Starting A2C training on Pendulum...")
    trained_agent = train_a2c()
    
    # テスト実行（オプション）
    print("\nTesting the trained agent...")
    test_agent(trained_agent)
    