import gymnasium as gym


# 创建月球着陆环境，并打开动画窗口
env = gym.make("LunarLander-v3", render_mode="human")

# 重置环境，得到一次任务的初始状态
state, info = env.reset()

print("初始状态：", state)
print("状态空间：", env.observation_space)
print("动作空间：", env.action_space)

total_reward = 0
step_count = 0

while True:
    # 从4种动作中随机选择一个
    action = env.action_space.sample()

    # 执行动作，环境返回新的状态和奖励
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    print(
        f"第{step_count}步："
        f"动作={action},"
        f"奖励={reward:.2f},"
        f"累计奖励={total_reward:.2f}"
    )

    # 更新当前状态
    state = next_state

    # 着陆、坠毁或超过时间限制时结束
    if terminated or truncated:
        break

env.close()

print("\n本轮结束")
print("总步数：", step_count)
print(f"总奖励：{total_reward:.2f}")