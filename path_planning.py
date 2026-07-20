import numpy as np
import matplotlib.pyplot as plt

# 设置起点、终点和总时间
start_position = 0
goal_position = 10
total_time = 5

# 生成时间点
time = np.linspace(0, total_time, 100)

# 将时间归一化到 0～1
s = time / total_time

# 使用平滑曲线规划位置
position = start_position + (goal_position - start_position) * (
    3 * s**2 - 2 * s**3
)

# 计算速度
velocity = np.gradient(position, time)

# 创建两幅图
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(time, position, color="blue", label="Planned position")
plt.scatter(0, start_position, color="green", label="Start")
plt.scatter(total_time, goal_position, color="red", label="Goal")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("1D Path Planning")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, velocity, color="orange", label="Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("output/path_planning.png", dpi=150)
plt.show()

print("路径规划结果已保存：output/path_planning.png")