import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from models import Generator, Discriminator
matplotlib.use('TkAgg')


# 超参数
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator(input_dim=2048*3*2).to(device)

generator.load_state_dict(torch.load('gbest_model.pth'))
discriminator.load_state_dict(torch.load('dbest_model.pth'))

generator.eval()
discriminator.eval()


# 把文件名作为一个变量
filename = "gd00105.npy"
file_path = "models_labels_npy"

# 使用 os.path.join 来拼接路径
full_path = os.path.join(file_path, filename)

# 然后再读取文件
test_labels = np.load(full_path)
# 生成测试数据
test_labels_tensor = torch.tensor(test_labels).float().unsqueeze(0).to(device)
# 生成噪声
batch_size = test_labels_tensor.size(0)
z = torch.randn(batch_size, latent_dim).to(device)

# 生成点云
with torch.no_grad():
    gen_point_clouds = generator(z, test_labels_tensor)
gen_point_clouds = gen_point_clouds.cpu().numpy()

num_samples = 1  # 生成10个点云数据

# 保存生成的点云数据
output_dir = 'generated_point_clouds'
os.makedirs(output_dir, exist_ok=True)

for i, point_cloud in enumerate(gen_point_clouds):
    file_path = os.path.join(output_dir, f'{filename}-point_cloud_{i}.npy')
    np.save(file_path, point_cloud)
    print(f'Saved {file_path}')


point_cloud_data = np.load(file_path)


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


