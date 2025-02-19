
import torch

import torch.nn as nn

from scipy.stats import wasserstein_distance

class Discriminator(nn.Module):
    def __init__(self, input_dim=2048 * 3 * 2):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.apply(self.weights_init)

    def forward(self, point_cloud, labels):
        print(point_cloud.shape, labels.shape)
        point_cloud_flat = point_cloud.view(point_cloud.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        d_in = torch.cat((point_cloud_flat, labels_flat), -1)
        validity = self.model(d_in)
        return validity

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=2048 * 3):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + 2048 * 3, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        self.apply(self.weights_init)

    def forward(self, noise, labels):
        labels_flat = labels.view(labels.size(0), -1)
        gen_input = torch.cat((noise, labels_flat), -1)
        point_cloud = self.model(gen_input)
        point_cloud = point_cloud.view(point_cloud.size(0), 2048, 3)
        return point_cloud

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

def chamfer_distance(pc1, pc2):
    dist1 = torch.cdist(pc1, pc2, p=2)
    dist2 = torch.cdist(pc2, pc1, p=2)
    chamfer_dist = (dist1.min(dim=2)[0].mean() + dist2.min(dim=2)[0].mean()) / 2
    return chamfer_dist * 100

def emd_loss(real_point_cloud, gen_point_cloud):
    emd = 0
    for i in range(real_point_cloud.size(0)):
        emd += wasserstein_distance(real_point_cloud[i].cpu().detach().numpy().flatten(),
                                    gen_point_cloud[i].cpu().detach().numpy().flatten())
    emd = torch.tensor(emd / real_point_cloud.size(0), requires_grad=True).to(real_point_cloud.device)
    return emd * 100

def combined_loss(real_point_cloud, gen_point_cloud, alpha=0.5):
    cd = chamfer_distance(real_point_cloud, gen_point_cloud)
    emd = emd_loss(real_point_cloud, gen_point_cloud)
    return alpha * cd + (1 - alpha) * emd
