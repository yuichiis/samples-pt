import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ==============================================================
# gSDE クラス（共通化部分）
# ==============================================================
class GeneralizedStateDependentNoise(tf.Module):
    def __init__(self, state_dim, action_dim, sigma_init=0.4, name="gSDE"):
        super().__init__(name=name)
        self.W = tf.Variable(tf.random.normal([state_dim, action_dim]), trainable=True)
        self.sigma_param = tf.Variable(tf.ones([action_dim]) * sigma_init, trainable=True)

    def sample(self, mean, state, deterministic=False):
        """
        mean: (batch, action_dim)
        state: (batch, state_dim)
        """
        batch_size = tf.shape(state)[0]
        sigma = tf.nn.softplus(self.sigma_param)

        if deterministic:
            return mean, tf.zeros_like(mean)

        eps = tf.random.normal([batch_size, tf.shape(mean)[1]])
        noise = tf.matmul(state, self.W) * eps
        action = mean + sigma * noise
        return action, noise

    def get_sigma(self):
        return tf.nn.softplus(self.sigma_param)

# ==============================================================
# SAC ネットワーク
# ==============================================================
class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action, gSDE=None):
        super().__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.mean = layers.Dense(action_dim)
        self.log_std = layers.Dense(action_dim)
        self.max_action = max_action
        self.gSDE = gSDE

    def call(self, state, deterministic=False):
        x = self.fc1(state)
        x = self.fc2(x)
        mean = self.mean(x)
        log_std = tf.clip_by_value(self.log_std(x), -20, 2)
        std = tf.exp(log_std)

        if self.gSDE is not None:
            # gSDE ノイズを利用
            action, _ = self.gSDE.sample(mean, state, deterministic)
            # SAC は tanh squashing
            action = tf.tanh(action) * self.max_action
            return action, mean, std
        else:
            # 通常の SAC ノイズ
            normal = tf.random.normal(tf.shape(mean))
            action = mean + std * normal
            action = tf.tanh(action) * self.max_action
            return action, mean, std


class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.q = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.q(x)


# ==============================================================
# SAC エージェント
# ==============================================================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, use_gsde=True):
        # gSDE クラスを用意
        self.gSDE = GeneralizedStateDependentNoise(state_dim, action_dim) if use_gsde else None

        self.actor = Actor(state_dim, action_dim, max_action, gSDE=self.gSDE)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)

        self.actor_opt = tf.keras.optimizers.Adam(3e-4)
        self.critic1_opt = tf.keras.optimizers.Adam(3e-4)
        self.critic2_opt = tf.keras.optimizers.Adam(3e-4)

        # 自動 α 調整
        self.log_alpha = tf.Variable(0.0, trainable=True)
        self.alpha_opt = tf.keras.optimizers.Adam(3e-4)
        self.target_entropy = -action_dim

        # ターゲットネット初期化
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

    def act(self, state, deterministic=False):
        state = state.reshape(1, -1).astype(np.float32)
        action, _, _ = self.actor(state, deterministic)
        return action.numpy()[0]

    # 学習処理（省略可、簡易版のみにしてます）
    # 本格的な実装では ReplayBuffer と update() が必要


# ==============================================================
# テスト実行
# ==============================================================
if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action, use_gsde=True)

    state, _ = env.reset()
    for t in range(10):
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"Step {t}, Action {action}, Reward {reward}")
        if done:
            state, _ = env.reset()
