import os

import gymnasium as gym
from stable_baselines3 import DQN


os.makedirs("output", exist_ok=True)

env = gym.make("LunarLander-v3")

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=6.3e-4,
    buffer_size=50_000,
    learning_starts=0,
    batch_size=128,
    gamma=0.99,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    seed=42,
    device="cpu",
)

print("开始使用调优参数训练DQN……")

model.learn(
    total_timesteps=100_000,
    log_interval=10,
)

model.save("output/dqn_lunar_lander_tuned")

env.close()

print("\n训练完成")
print("模型已保存到：output/dqn_lunar_lander_tuned.zip")