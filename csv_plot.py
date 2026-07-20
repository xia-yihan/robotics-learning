import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = "data/robot_log.csv"

# 模拟机器人的运行数据
time = np.arange(0, 10, 0.2)
position = 2 * time + np.sin(time)
target = 2 * time

# 写入 CSV 文件
with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "position", "target"])

    for t, p, target_p in zip(time, position, target):
        writer.writerow([t, p, target_p])

print("CSV 日志已生成：", csv_path)

# 从 CSV 文件读取数据
time_data = []
position_data = []
target_data = []

with open(csv_path, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        time_data.append(float(row["time"]))
        position_data.append(float(row["position"]))
        target_data.append(float(row["target"]))

# 绹制曲线
plt.plot(time_data, position_data, label="Actual position")
plt.plot(time_data, target_data, "--", label="Target position")

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Robot Position Log")
plt.legend()
plt.grid(True)

plt.savefig("output/robot_log_plot.png", dpi=150)
plt.show()

print("图表已保存：output/robot_log_plot.png")