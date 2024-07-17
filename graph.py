import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x_data = [f"{i}" for i in range(0, 24)]
X_raw = np.load('./Data/R4_difference_position_estimation.npy')
y_data = 100 - (np.sum(X_raw, axis=0) / (10 ** 7) * 100) - 50

zipped = sorted(zip(y_data, x_data))
arr1_sorted, arr2_sorted = zip(*zipped)
y_data = list(arr1_sorted)[::-1]
x_data = list(arr2_sorted)[::-1]

# 正确显示中文和负号
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False

# 画图
fig, ax = plt.subplots()

# 画柱状图
for i in range(len(x_data)):
    # 设置颜色渐变
    color = (0, 0, 1, 1 - i / len(x_data))  # RGBA颜色，A(alpha)表示透明度
    ax.bar(x_data[i], y_data[i], color=color)

# 画折线图
ax.plot(x_data, y_data, color='r', marker='o',markersize=3)

# 设置图片名称
# plt.title("Contribution Rank of Leakage Differential Bit for 3-round BipBip")
# 设置x轴标签名
plt.xlabel("Bit Position")
# 设置y轴标签名
plt.ylabel("Contribution")
# 设置y轴的显示范围和刻度
plt.ylim(0, 1)  # 设置y轴的范围为0到1
plt.yticks(np.arange(0, 1, 0.1))  # 从0到1，间隔为0.1

# 保存为PDF文件
plt.savefig("./Fig.5b.pdf")

plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
#
# # Visualize input data
# input_data = np.random.rand(24, 2)
# ax.imshow(input_data, aspect='auto', cmap='viridis', extent=[0, 2, 0, 24])
# for i in range(24):
#     ax.text(0.5, i + 0.5, f"{input_data[i, 0]:.2f}", ha='center', va='center', color='white')
#     ax.text(1.5, i + 0.5, f"{input_data[i, 1]:.2f}", ha='center', va='center', color='white')
#
# # Indicate convolution kernel
# rect = plt.Rectangle((0, 23), 2, 1, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# ax.text(1, 24.5, 'Convolution\nKernel', ha='center', va='center', color='red')
#
# # Visualize output data
# output_data = np.zeros((24, 2))
# for i in range(24):
#     output_data[i, :] = np.sum(input_data[i, :]) / 2  # Simplified convolution operation
#
# ax.imshow(output_data, aspect='auto', cmap='viridis', extent=[3, 5, 0, 24])
# for i in range(24):
#     ax.text(4, i + 0.5, f"{output_data[i, 0]:.2f}", ha='center', va='center', color='white')
#
# ax.set_title('Convolution Operation Visualization')
# ax.set_xlim([-1, 6])
# ax.set_ylim([24, -1])
# ax.axis('off')
# plt.show()
