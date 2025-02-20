import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import torch.optim as optim
from datetime import datetime
from models import Generator, Discriminator, combined_loss

folder_path = 'models_pointcloud_npy'
label_path = 'models_labels_npy'
# 超参数
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
epochs = 10000

class PointCloudDataset(Dataset):
    def __init__(self, file_paths, label_paths):
        self.file_paths = file_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        data_tensor = torch.tensor(data).float()

        label = np.load(self.label_paths[idx])
        label_tensor = torch.tensor(label).float()

        return data_tensor, label_tensor

# Function to get npy file paths
def get_npy_files(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
    return file_paths



# Get file paths
file_paths = get_npy_files(folder_path)
label_paths = get_npy_files(label_path)

# Initialize the dataset
dataset = PointCloudDataset(file_paths, label_paths)

# Calculate split sizes (80% train, 10% validation, 10% test)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=27, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=27, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=27, shuffle=False)


# Now you can train the model using `train_loader`, `val_loader`, and `test_loader`


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator(input_dim=2048*3*2).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# 获取当前日期和时间
current_time = datetime.now().strftime('%Y%m%d_%H%M')

# 创建新的文件夹
checkpoint_dir = f'checkpoints_{current_time}'
os.makedirs(checkpoint_dir, exist_ok=True)

min_loss = float('inf')
dbest_save_path = os.path.join(checkpoint_dir, 'discriminator_best.pth')
gbest_save_path = os.path.join(checkpoint_dir, 'generator_best.pth')
dlast_save_path = os.path.join(checkpoint_dir, 'discriminator_last.pth')
glast_save_path = os.path.join(checkpoint_dir, 'generator_last.pth')

log_file_path = os.path.join(checkpoint_dir, 'training_log.log')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_path, filemode='w')
logger = logging.getLogger()

for epoch in range(epochs):

    for i, (real_point_clouds, labels) in enumerate(train_loader):
        batch_size = real_point_clouds.size(0)
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        real_point_clouds = real_point_clouds.to(device)
        labels = labels.to(device)

        # 训练生成器
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_point_clouds = generator(z, labels)
        g_loss = combined_loss(real_point_clouds, gen_point_clouds)

        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = combined_loss(real_point_clouds, real_point_clouds)
        fake_loss = combined_loss(real_point_clouds, gen_point_clouds.detach())
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        # 在每个epoch结束后进行验证

    with torch.no_grad():  # 在验证过程中不需要计算梯度
        val_loss = 0
        for i, (real_point_clouds, labels) in enumerate(val_loader):
            real_point_clouds = real_point_clouds.to(device)
            labels = labels.to(device)

            z = torch.randn(real_point_clouds.size(0), latent_dim).to(device)
            gen_point_clouds = generator(z, labels)
            val_loss += combined_loss(real_point_clouds, gen_point_clouds)

        val_loss /= len(val_loader)  # 计算平均验证损失

        print(f"Epoch [{epoch}/{epochs}], Validation Loss: {val_loss.item()}")



    if epoch % 500 == 0:
        dbest_save_path = os.path.join(checkpoint_dir, f'dbest_model_{epoch}.pth')
        gbest_save_path = os.path.join(checkpoint_dir, f'gbest_model_{epoch}.pth')
        torch.save(discriminator.state_dict(), dbest_save_path)
        torch.save(generator.state_dict(), gbest_save_path)
    if epoch % 10 == 0:
        log_message = (
            f"Epoch [{epoch}/{epochs}], Training D loss: {d_loss.item()}, G loss: {g_loss.item()}, Validation Loss: {val_loss.item()}")
        print(log_message)
        logger.info(log_message)

with torch.no_grad():  # 在验证过程中不需要计算梯度
    test_loss = 0
    for i, (real_point_clouds, labels) in enumerate(test_loader):
        real_point_clouds = real_point_clouds.to(device)
        labels = labels.to(device)

        z = torch.randn(real_point_clouds.size(0), latent_dim).to(device)
        gen_point_clouds = generator(z, labels)
        test_loss += combined_loss(real_point_clouds, gen_point_clouds)

    test_loss /= len(test_loader)  # 计算平均验证损失
    log_message = f"Test Loss: {test_loss.item()}"
    print(log_message)
    logger.info(log_message)
