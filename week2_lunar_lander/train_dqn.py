import os

import gymnasium as gym
from stable_baselines3 import DQN


# 创建模型保存目录
os.makedirs("output", exist_ok=True)

# 训练时不显示动画，否则训练速度会非常慢
env = gym.make("LunarLander-v3")

# 创建DQN智能体
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    seed=42,
    device="cpu",
)

print("开始训练DQN智能体……")

# 与环境交互并更新神经网络
model.learn(
    total_timesteps=150_000,
    log_interval=10,
)

# 保存训练后的模型
model.save("output/dqn_lunar_lander")

env.close()

print("\n训练完成")
print("模型已保存到：output/dqn_lunar_lander.zip")