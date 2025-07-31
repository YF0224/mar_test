
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------- Residual Block ----------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, 1, 0)
        )

    def forward(self, x):
        return x + self.block(x)

# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_channels=128, res_nums=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels // 4, 4, 2, 1)  # 32->16
        self.conv2 = nn.Conv2d(hidden_channels // 4, hidden_channels // 2, 4, 2, 1)  # 16->8
        self.conv3 = nn.Conv2d(hidden_channels // 2, hidden_channels, 3, 1, 1)  # 8->8
        self.residual = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(res_nums)])
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.conv_mu = nn.Conv2d(hidden_channels, latent_dim, 3, 1, 1)
        self.conv_logvar = nn.Conv2d(hidden_channels, latent_dim, 3, 1, 1)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        for block in self.residual:
            x = block(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar

# ---------------- Decoder ----------------
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels, hidden_channels=128, res_nums=2):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, hidden_channels, 3, 1, 1)
        self.residual = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(res_nums)])
        self.convT1 = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1)  # 8->16
        self.convT2 = nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, 2, 1)  # 16->32
        self.conv2 = nn.Conv2d(hidden_channels // 4, out_channels, 3, 1, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, z):
        z = self.leakyrelu(self.conv1(z))
        for block in self.residual:
            z = block(z)
        z = self.leakyrelu(self.convT1(z))
        z = self.leakyrelu(self.convT2(z))
        z = torch.tanh(self.conv2(z))
        return z

# ---------------- VAE ----------------
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=16, hidden_channels=128, res_nums=2):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, hidden_channels, res_nums)
        self.decoder = Decoder(latent_dim, in_channels, hidden_channels, res_nums)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ---------------- 损失函数 ----------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld
