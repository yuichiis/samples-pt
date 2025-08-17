import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from typing import Callable


# --- rl_zoo3のハイパーパラメータを完全に再現 ---
# learning_rateは後で直接指定するため、辞書からは削除
hyperparams = {
    "policy": "MlpPolicy",
    "n_steps": 8,
    "gamma": 0.9,
    "gae_lambda": 0.9,
    "ent_coef": 0.0,
    "vf_coef": 0.4,
    "max_grad_norm": 0.5,
    "use_sde": True,
    "normalize_advantage": False,
    "policy_kwargs": dict(log_std_init=-2, ortho_init=False),
}

# --- rl_zoo3の設定に合わせて環境設定も変更 ---
N_ENVS = 8
TOTAL_TIMESTEPS = 400_000
ENV_ID = "Pendulum-v1"
LOG_DIR = "./logs/a2c_pendulum/"
os.makedirs(LOG_DIR, exist_ok=True)

# --- 環境の準備（変更なし） ---
#train_env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=42) # seedをここでも設定
#train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=hyperparams["gamma"])
train_env = gym.make(ENV_ID)

# --- EvalCallbackの準備（変更なし） ---
#eval_env = make_vec_env(ENV_ID, n_envs=1, seed=42)
#eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
eval_env = gym.make(ENV_ID)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=LOG_DIR,
    log_path=LOG_DIR,
    eval_freq=max(10000 // N_ENVS, 1),
    deterministic=True,
    render=False
)

# --- 修正点: 学習率スケジュール関数を自前で定義 ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    線形に減衰する学習率スケジュールを作成する。

    :param initial_value: 初期学習率
    :return: progress_remaining (1.0 -> 0.0) を受け取り、
             現在の学習率を返す関数
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remainingは学習の進行度に応じて1.0から0.0まで減少する
        """
        return progress_remaining * initial_value

    return func

# --- モデルの作成（ここを修正） ---
model = A2C(
    env=train_env,
    verbose=0,
    learning_rate=linear_schedule(7e-4),
    **hyperparams,
    seed=43
)

print("完全にrl_zoo3の設定を再現して学習を開始します...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)
print("学習が完了しました。")

print(f"\n学習中のベストモデルは {LOG_DIR}best_model.zip に保存されています。")

# 環境を閉じる
train_env.close()
eval_env.close()