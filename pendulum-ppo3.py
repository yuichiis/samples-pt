import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.distributions import Normal # <- この行を削除
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import gymnasium as gym
import imageio

# ===================================================================
# ヘルパー関数 (Numpy実装なので変更なし)
# ===================================================================

def standardize(
    x,         # (rolloutSteps)
    ddof=None,
    ) :
    # baseline
    mean = np.mean(x)     # ()

    baseX = x - mean                    # (rolloutSteps)
    # std
    if ddof:
        n = len(x)-1
    else :
        n = len(x)

    variance = np.sum(np.square(baseX)) / n                 # ()
    stdDev = np.sqrt(variance)                              # ()
    # standardize
    result = baseX / (stdDev + 1e-8)                        # (rolloutSteps)
    return result                                           # (rolloutSteps)

def compute_advantages_and_returns(rewards, values, dones):
    """GAEとリターンを計算する (Numpy実装なので変更なし)"""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_advantage = delta
        else:
            delta = rewards[t] + GAMMA * values[t+1] - values[t]
            last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
        advantages[t] = last_advantage
        
    returns = advantages + values[:-1]
    
    return advantages, returns

# ===================================================================
# PPOの実装 (PyTorch版)
# ===================================================================

# -------------------------------------------------------------------- #
# (追加) torch.distributions を使わないための自作ヘルパー関数
# -------------------------------------------------------------------- #

def sample_from_distribution(mu, log_std):
    """PyTorch版: torch.distributionsなしで正規分布からサンプリング"""
    std = torch.exp(log_std)
    # reparameterization trick: sample = mu + std * N(0, 1)
    epsilon = torch.randn_like(mu)
    return mu + std * epsilon

def calculate_log_prob_entropy(mu, log_std, action):
    """
    PyTorch版: torch.distributionsなしで正規分布の統計量を計算する。
    Args:
        mu (torch.Tensor):      平均                       (batchsize, num_actions)
        log_std (torch.Tensor): 標準偏差のlog              (num_actions)
        action (torch.Tensor):  確率を計算したいアクション   (batchsize, num_actions)
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (log_prob, entropy)
    """
    # 数値安定性のための微小値
    epsilon = 1e-8
    pi = torch.tensor(np.pi, device=mu.device)
    std = torch.exp(log_std)
    stable_std = std + epsilon

    # 対数確率密度 (log_prob)
    log_prob = (
        -torch.log(stable_std)
        - 0.5 * torch.log(2.0 * pi)
        - 0.5 * torch.square((action - mu) / stable_std)
    )
    
    # エントロピー (entropy)
    entropy = 0.5 + 0.5 * torch.log(2.0 * pi) + torch.log(stable_std)
    # muの形状 (batch_size, action_dim) に合わせてブロードキャストする
    entropy = entropy.expand_as(mu)
    
    return log_prob, entropy  # (log_prob=(batch,act_dim), entropy=(batch,act_dim))


# -------------------------------------------------------------------- #
# ハイパーパラメータ (TensorFlow版と同じ)
# -------------------------------------------------------------------- #
ENV_NAME = 'Pendulum-v1'
GAMMA = 0.9
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2
N_EPOCHS = 10
BATCH_SIZE = 64
N_ROLLOUT_STEPS = 1024
TARGET_SCORE = -250
VALUE_LOSS_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
STANDARDIZE = True

# PyTorch用のデバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (変更なし) Actor-CriticモデルをPyTorchで実装
class ActorCritic(nn.Module):
    """Actor-Criticモデル (PyTorch版)"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common_layer1 = nn.Linear(state_dim, 128)
        self.common_layer2 = nn.Linear(128, 128)
        self.actor_mu = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.common_layer1(state))
        x = F.relu(self.common_layer2(x))
        mu = self.actor_mu(x)
        value = self.critic_head(x)
        return mu, value

# (変更なし) 決定論的な最善行動を取得する関数 (PyTorch版)
def get_best_action(model, state, action_bound):
    model.eval() # 評価モード
    with torch.no_grad(): # 勾配計算を無効化
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mu, _ = model(state_tensor)
        action = np.clip(mu.cpu().numpy().flatten(), -action_bound, action_bound)
    return action

def main():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    episode_count = 0
    total_step = 0
    episode_score = 0
    last_episode_scores = []
    
    while True:
        episode_count += 1
        
        # === 1. データ収集 (Rollout) (自作関数を使用) ===
        states_mem, actions_mem, rewards_mem, dones_mem, next_state_mem = [], [], [], [], []
        state, _ = env.reset()
        
        model.eval()
        for t in range(N_ROLLOUT_STEPS):
            total_step += 1
            
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                mu, _ = model(state_tensor)
                
                # (変更) 分布から行動をサンプリング (自作関数)
                action = sample_from_distribution(mu, model.log_std)
            
            clipped_action = np.clip(action.cpu().numpy().flatten(), -action_bound, action_bound)
            next_state, reward, done, truncated, _ = env.step(clipped_action)
            episode_score += reward
            
            states_mem.append(state)
            actions_mem.append(action.cpu().numpy().flatten())
            rewards_mem.append(reward)
            dones_mem.append(done or truncated)
            next_state_mem.append(next_state)
            
            state = next_state
            if done or truncated:
                state, _ = env.reset()
                if len(last_episode_scores) >= 100:
                    last_episode_scores.pop(0)
                last_episode_scores.append(episode_score)
                episode_score = 0

        # === 2. 学習データの準備 (変更なし) ===
        states_tensor = torch.tensor(np.array(states_mem), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(np.array(actions_mem), dtype=torch.float32).to(device)

        with torch.no_grad():
            old_mean_tensor, values_old_tensor = model(states_tensor)
            next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            _, next_value_tensor = model(next_state_tensor)
        
        values_for_gae = np.append(values_old_tensor.cpu().numpy().flatten(), next_value_tensor.cpu().numpy().flatten())

        # (変更) サンプリングした行動の対数確率を計算 (自作関数)
        with torch.no_grad():
            old_log_probs_tensor, _ = calculate_log_prob_entropy(old_mean_tensor, model.log_std, actions_tensor)
        
        advantages, returns = compute_advantages_and_returns(
            rewards_mem, values_for_gae, dones_mem
        )

        if STANDARDIZE:
            advantages = standardize(advantages)

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

        dataset = TensorDataset(states_tensor, actions_tensor, old_log_probs_tensor, advantages_tensor, returns_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # === 3. モデルの学習 (自作関数を使用) ===
        model.train()
        avg_loss, avg_a_loss, avg_c_loss, avg_entropy, avg_std = 0, 0, 0, 0, 0
        num_batches = 0
        
        for _ in range(N_EPOCHS):
            for batch in dataloader:
                states_b, actions_b, old_log_probs_b, advantages_b, returns_b = batch
                
                new_mu, new_values = model(states_b)
                new_values = new_values.squeeze()
                
                # (変更) 新しい分布での対数確率とエントロピーを計算 (自作関数)
                new_log_probs, entropy = calculate_log_prob_entropy(new_mu, model.log_std, actions_b)
                
                new_log_probs = new_log_probs.sum(axis=1)
                old_log_probs_b = old_log_probs_b.sum(axis=1)
                entropy = entropy.sum(axis=1)

                # 1. Actor Loss (Policy Loss)
                ratio = torch.exp(new_log_probs - old_log_probs_b)
                clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                actor_loss = -torch.mean(
                    torch.min(ratio * advantages_b, clipped_ratio * advantages_b)
                )
                
                # 2. Critic Loss (Value Loss)
                critic_loss = F.mse_loss(returns_b, new_values)
                
                # 3. Entropy Loss
                entropy_loss = -torch.mean(entropy)
                
                total_loss = actor_loss + VALUE_LOSS_WEIGHT * critic_loss + ENTROPY_WEIGHT * entropy_loss
                
                with torch.no_grad():
                    std_metric = torch.mean(torch.exp(model.log_std))

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                avg_loss += total_loss.item()
                avg_a_loss += actor_loss.item()
                avg_c_loss += critic_loss.item()
                avg_entropy += -entropy_loss.item()
                avg_std += std_metric.item()
                num_batches += 1

        avg_loss /= num_batches
        avg_a_loss /= num_batches
        avg_c_loss /= num_batches
        avg_entropy /= num_batches
        avg_std /= num_batches

        # === 4. 進捗の評価と表示 (変更なし) ===
        eval_env = gym.make(ENV_NAME)
        eval_scores = []
        for _ in range(10):
            state, _ = eval_env.reset()
            done, truncated = False, False
            score = 0
            while not (done or truncated):
                action = get_best_action(model, state, action_bound)
                state, reward, done, truncated, _ = eval_env.step(action)
                score += reward
            eval_scores.append(score)
        avg_evl_score = np.mean(eval_scores)
        
        avg_score = np.mean(last_episode_scores) if last_episode_scores else -1600
        
        print(f"Ep:{episode_count}, St:{total_step}, Scr:{avg_score:.1f}, ALoss:{avg_a_loss:.3f}, CLoss:{avg_c_loss:.3f}, Ety:{avg_entropy:.4f}, Std:{avg_std:.4f}, EvScr:{avg_evl_score:.1f}")

        if avg_score >= TARGET_SCORE:
            print(f"\n目標スコア {TARGET_SCORE} を達成しました！学習を終了します。")
            break

    print("\n--- テスト実行 ---")
    env_render = gym.make("Pendulum-v1", render_mode="rgb_array")
    frames = []
    for i in range(3):
        state, _ = env_render.reset()
        done, truncated = False, False
        test_reward = 0
        
        while not (done or truncated):
            frames.append(env_render.render())
            action = get_best_action(model, state, action_bound)
            state, reward, done, truncated, _ = env_render.step(action)
            test_reward += reward
        print(f"Test Episode {i+1}, Total Reward: {test_reward:.2f}")
    
    env_render.close()
    
    gif_path = 'pendulum-ppo-pytorch-no-dist.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")


if __name__ == '__main__':
    main()