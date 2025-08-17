import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
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

# (変更) Actor-CriticモデルをPyTorchで実装
class ActorCritic(nn.Module):
    """Actor-Criticモデル (PyTorch版)"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 共通層
        self.common_layer1 = nn.Linear(state_dim, 128)
        self.common_layer2 = nn.Linear(128, 128)
        
        # Actorヘッド: 行動の平均(mu)を出力
        self.actor_mu = nn.Linear(128, action_dim)
        
        # Actorの標準偏差(std): 学習可能な変数としてlog_stdを定義
        # torch.nn.Parameterでモデルの学習可能パラメータとして登録
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Criticヘッド
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.common_layer1(state))
        x = F.relu(self.common_layer2(x))
        
        # 平均(mu)と状態価値(value)を返す
        mu = self.actor_mu(x)
        value = self.critic_head(x)
        return mu, value

# (変更) 決定論的な最善行動を取得する関数 (PyTorch版)
def get_best_action(model, state, action_bound):
    model.eval() # 評価モード
    with torch.no_grad(): # 勾配計算を無効化
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mu, _ = model(state_tensor)
        # 平均値をそのまま行動とし、環境の行動範囲にクリップする
        action = np.clip(mu.cpu().numpy().flatten(), -action_bound, action_bound)
    return action

def main():
    # Pendulum環境のセットアップ
    env = gym.make(ENV_NAME)
    # 状態と行動の次元数、行動の範囲を取得
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
        
        # === 1. データ収集 (Rollout) (PyTorch版) ===
        states_mem, actions_mem, rewards_mem, dones_mem, log_probs_mem = [], [], [], [], []
        state, _ = env.reset()
        
        model.eval() # データ収集時は評価モード
        for t in range(N_ROLLOUT_STEPS):
            total_step += 1
            
            with torch.no_grad(): # 勾配計算は不要
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                
                # モデルからmuを取得し、正規分布を定義
                mu, _ = model(state_tensor)
                std = torch.exp(model.log_std)
                dist = Normal(mu, std)
                
                # 分布から行動をサンプリング
                action = dist.sample()
                # サンプリングした行動の対数確率を計算
                log_prob = dist.log_prob(action)
            
            # 行動を環境の範囲内にクリップして実行
            clipped_action = np.clip(action.cpu().numpy().flatten(), -action_bound, action_bound)
            next_state, reward, done, truncated, _ = env.step(clipped_action)
            episode_score += reward
            
            # メモリに保存 (Numpy配列に変換して保存)
            states_mem.append(state)
            actions_mem.append(action.cpu().numpy().flatten())
            rewards_mem.append(reward)
            dones_mem.append(done or truncated)
            log_probs_mem.append(log_prob.cpu().numpy().flatten())
            
            state = next_state
            if done or truncated:
                state, _ = env.reset()
                if len(last_episode_scores) >= 100:
                    last_episode_scores.pop(0)
                last_episode_scores.append(episode_score)
                episode_score = 0

        # === 2. 学習データの準備 (PyTorch版) ===
        states_tensor = torch.tensor(np.array(states_mem), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(np.array(actions_mem), dtype=torch.float32).to(device)
        old_log_probs_tensor = torch.tensor(np.array(log_probs_mem), dtype=torch.float32).to(device)

        # GAE計算のための価値関数値を取得
        with torch.no_grad():
            _, values_old_tensor = model(states_tensor)
            next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            _, next_value_tensor = model(next_state_tensor)
        
        values_for_gae = np.append(values_old_tensor.cpu().numpy().flatten(), next_value_tensor.cpu().numpy().flatten())
        
        advantages, returns = compute_advantages_and_returns(
            rewards_mem, values_for_gae, dones_mem
        )

        if STANDARDIZE:
            advantages = standardize(advantages)

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

        # PyTorchのDataLoaderを使用してバッチを作成
        dataset = TensorDataset(states_tensor, actions_tensor, old_log_probs_tensor, advantages_tensor, returns_tensor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # === 3. モデルの学習 (PyTorch版) ===
        model.train() # 学習モード
        avg_loss, avg_a_loss, avg_c_loss, avg_entropy = 0, 0, 0, 0
        num_batches = 0
        
        for _ in range(N_EPOCHS):
            for batch in dataloader:
                states_b, actions_b, old_log_probs_b, advantages_b, returns_b = batch
                
                # 現在のモデルで新しい分布を予測
                new_mu, new_values = model(states_b)
                new_values = new_values.squeeze()
                
                # 新しい分布での対数確率とエントロピーを計算
                new_std = torch.exp(model.log_std)
                new_dist = Normal(new_mu, new_std)
                new_log_probs = new_dist.log_prob(actions_b)
                entropy = new_dist.entropy()
                
                # 多次元行動も考慮し、log_probとentropyをスカラーに変換
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
                
                # Total Loss
                total_loss = actor_loss + VALUE_LOSS_WEIGHT * critic_loss + ENTROPY_WEIGHT * entropy_loss
                
                # 勾配計算とパラメータ更新
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 勾配クリッピング
                optimizer.step()
                
                # ログ用に損失を記録
                avg_loss += total_loss.item()
                avg_a_loss += actor_loss.item()
                avg_c_loss += critic_loss.item()
                avg_entropy += -entropy_loss.item() # エントロピー自体は正の値として記録
                num_batches += 1

        avg_loss /= num_batches
        avg_a_loss /= num_batches
        avg_c_loss /= num_batches
        avg_entropy /= num_batches

        # === 4. 進捗の評価と表示 ===
        eval_env = gym.make(ENV_NAME)
        eval_scores = []
        for _ in range(10):
            state, _ = eval_env.reset()
            done, truncated = False, False
            score = 0
            while not (done or truncated):
                # 評価時は決定論的な行動を選択
                action = get_best_action(model, state, action_bound)
                state, reward, done, truncated, _ = eval_env.step(action)
                score += reward
            eval_scores.append(score)
        avg_evl_score = np.mean(eval_scores)
        
        avg_score = np.mean(last_episode_scores) if last_episode_scores else -1600 # 初期値
        
        print(f"Ep:{episode_count}, St:{total_step}, Scr:{avg_score:.1f}, ALoss:{avg_a_loss:.3f}, CLoss:{avg_c_loss:.3f}, Ety:{avg_entropy:.4f}, EvScr:{avg_evl_score:.1f}")

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
    
    gif_path = 'pendulum-ppo-pytorch.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIFを'{gif_path}'に保存しました。")


if __name__ == '__main__':
    main()