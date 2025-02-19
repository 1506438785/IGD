import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D

filename = "fxd00100.npy-point_cloud_0.npy"

# 假设你的文件路径是 pre_data/data_label/label
file_path = "generated_point_clouds/"

# 使用 os.path.join 来拼接路径
full_path = os.path.join(file_path, filename)

full_path = 'generated_point_clouds/aj00104.npy-point_cloud_0.npy'
# 读取.npy文件
point_cloud_data = np.load(full_path)


# 提取点的坐标
x = point_cloud_data[:, 0]
y = point_cloud_data[:, 1]
z = point_cloud_data[:, 2]

# 创建图像并定义子图
fig = plt.figure(figsize=(12, 6))

# 添加3D子图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z, s=3, c='b', marker='o')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Point Cloud')

# 添加2D XZ视图子图
ax2 = fig.add_subplot(122)
ax2.scatter(x, z, s=1, c='b', marker='o')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')
ax2.set_title('XZ View')

# 显示图像
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()

