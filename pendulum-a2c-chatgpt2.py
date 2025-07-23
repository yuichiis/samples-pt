import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- 環境とパラメータ設定 ---
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

gamma = 0.99
actor_lr = 3e-4
critic_lr = 1e-3
entropy_coef = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Actorモデル ---
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

# --- Criticモデル ---
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.net(state)

# --- モデルと最適化 ---
actor = Actor().to(device)
critic = Critic().to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

# --- アクション選択 ---
def get_action(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    mean, std = actor(state)
    dist = Normal(mean, std)
    action = dist.sample()
    action_clipped = action.clamp(-action_bound, action_bound)
    log_prob = dist.log_prob(action).sum(dim=-1)
    return action_clipped.cpu().detach().numpy()[0], log_prob

# --- 学習ステップ ---
def train_step(state, action, reward, next_state, done, log_prob):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    action = torch.FloatTensor(action).unsqueeze(0).to(device)
    reward = torch.FloatTensor([reward]).to(device)
    done = torch.FloatTensor([done]).to(device)

    value = critic(state)
    next_value = critic(next_state)
    target = reward + (1 - done) * gamma * next_value
    advantage = target - value

    # --- Critic loss ---
    critic_loss = advantage.pow(2).mean()

    # --- Actor loss ---
    mean, std = actor(state)
    dist = Normal(mean, std)
    new_log_prob = dist.log_prob(action).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    actor_loss = (-new_log_prob * advantage.detach() - entropy_coef * entropy).mean()

    # --- 最適化 ---
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

# --- メインループ ---
max_episodes = 500
reward_history = []

for episode in range(max_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, log_prob = get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        train_step(state, action, reward, next_state, done, log_prob)
        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 10 == 0:
        avg = np.mean(reward_history[-10:])
        print(f"Episode {episode}, Avg Reward: {avg:.2f}")

env.close()
