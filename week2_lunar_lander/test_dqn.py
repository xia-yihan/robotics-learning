import gymnasium as gym
from stable_baselines3 import DQN


# 加载刚才训练的模型
model = DQN.load("output/dqn_lunar_lander_tuned")

# 测试时显示动画
env = gym.make("LunarLander-v3", render_mode="human")

test_episodes = 5

for episode in range(test_episodes):
    state, info = env.reset()
    total_reward = 0.0

    while True:
        # deterministic=True：测试时不再随机探索
        action, _ = model.predict(state, deterministic=True)

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"第{episode + 1}轮，总奖励：{total_reward:.2f}")

env.close()