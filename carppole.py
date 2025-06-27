import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

# GPUが利用可能な場合はGPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQNのためのニューラルネットワークの定義
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 経験リプレイ用のメモリクラス
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done, trunc):
        self.memory.append((state, action, next_state, reward, done, trunc))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQNエージェントの定義
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.05 #0.01
        self.epsilon_decay = 0.999 #0.995
        self.batch_size = 64
        self.learning_rate = 0.001
        
        # Q-Networkとターゲットネットワーク
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            act_values = self.policy_net(state)
            return torch.argmax(act_values).item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # バッチの各要素を取得
        states = torch.FloatTensor(np.array(batch[0])).to(device)
        actions = torch.LongTensor(np.array(batch[1])).view(-1, 1).to(device)
        next_states = torch.FloatTensor(np.array(batch[2])).to(device)
        rewards = torch.FloatTensor(np.array(batch[3])).view(-1, 1).to(device)
        dones = torch.FloatTensor(np.array(batch[4])).view(-1, 1).to(device)
        
        # Q値の計算
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Huber損失を使用
        loss = F.smooth_l1_loss(current_q, expected_q)
        
        # 最適化ステップ
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # epsilonの減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# トレーニング関数
def train_dqn(episodes=500, target_update=10):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    
    for e in range(episodes):
        state, info = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, trunc, info = env.step(action)
            
            # 早期終了の場合は罰則を与える
            if done and score < 499:
                reward = -10
            
            agent.memory.push(state, action, next_state, reward, done, trunc)
            state = next_state
            score += 1
            
            agent.learn()
            
            if done or trunc:
                if trunc:
                    print(done,trunc,info)

                print(f"Episode: {e+1}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
                scores.append(score)
                break
        
        # ターゲットネットワークの更新
        if e % target_update == 0:
            agent.update_target_net()
    
    # モデルを保存
    agent.save("cartpole_dqn.pth")
    
    # 学習曲線をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.savefig("learning_curve.png")
    plt.show()
    
    return agent, scores

# テスト関数
def test_agent(agent, episodes=10, render=True):
    env = gym.make('CartPole-v1')
    scores = []
    
    for e in range(episodes):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            if render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, trunc, info = env.step(action)
            state = next_state
            score += 1
            
            if done:
                print(done,trunc,info)
                print(f"Test Episode: {e+1}/{episodes}, Score: {score}")
                scores.append(score)
                break
    
    env.close()
    return scores

if __name__ == "__main__":
    # トレーニングの実行
    agent, train_scores = train_dqn(episodes=500, target_update=5)
    
    # エージェントのテスト
    test_scores = test_agent(agent, episodes=5)
    print(f"平均テストスコア: {np.mean(test_scores)}")
    