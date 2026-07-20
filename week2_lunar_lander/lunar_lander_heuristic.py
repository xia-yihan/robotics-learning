import gymnasium as gym


def heuristic(state):
    """根据飞船状态，按照人工规则选择动作。"""

    x_position = state[0]
    y_position = state[1]
    x_velocity = state[2]
    y_velocity = state[3]
    angle = state[4]
    angular_velocity = state[5]
    left_leg_contact = state[6]
    right_leg_contact = state[7]

    # 根据水平位置和水平速度，计算期望角度
    target_angle = x_position * 0.5 + x_velocity * 1.0

    # 限制期望角度，避免飞船倾斜过大
    target_angle = max(-0.4, min(0.4, target_angle))

    # 飞船偏离中心越远，希望保持的高度就越高
    target_height = 0.55 * abs(x_position)

    # 判断当前需要优先调整角度还是高度
    angle_control = (
        (target_angle - angle) * 0.5
        - angular_velocity * 1.0
    )

    height_control = (
        (target_height - y_position) * 0.5
        - y_velocity * 0.5
    )

    # 如果至少一条腿接触地面，主要任务变为减小下降速度
    if left_leg_contact or right_leg_contact:
        angle_control = 0.0
        height_control = -y_velocity * 0.5

    # 默认不启动发动机
    action = 0

    # 需要较强的竖直支撑时，启动主发动机
    if height_control > abs(angle_control) and height_control > 0.05:
        action = 2

    # 根据角度控制需求，启动左右方向发动机
    elif angle_control < -0.05:
        action = 3

    elif angle_control > 0.05:
        action = 1

    return action


# 创建带动画的月球着陆环境
env = gym.make("LunarLander-v3", render_mode="human")

state, info = env.reset()

total_reward = 0.0
step_count = 0

while True:
    # 根据当前状态，由启发式策略选择动作
    action = heuristic(state)

    # 把动作交给环境
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1
    state = next_state

    if terminated or truncated:
        break

env.close()

print("\n启发式策略运行结束")
print("总步数：", step_count)
print(f"总奖励：{total_reward:.2f}")