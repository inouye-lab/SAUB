import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

pl.seed_everything(0)


# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_latent_noise_scale=50, embedding_dim=2,
                 is_add_latent_noise=False):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, hidden_dim) if is_add_latent_noise else nn.Linear(latent_dim,
                                                                                                    hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        if is_add_latent_noise:
            self.latent_noise_embedding = nn.Embedding(num_latent_noise_scale, embedding_dim)

        self.is_add_latent_noise = is_add_latent_noise

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, latent_noise_idx=None):
        if self.is_add_latent_noise:
            if latent_noise_idx is not None:
                emb = self.latent_noise_embedding(latent_noise_idx)
            else:
                emb = self.latent_noise_embedding(torch.zeros(z.shape[0], device=z.device).type(torch.long))
            z = torch.hstack((z, emb))
        return self.decoder(z)

    def forward(self, x, latent_noise_idx=None, noise=None, latent_noise_scale=None):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        if self.is_add_latent_noise and (latent_noise_idx is not None):
            z += noise
            logvar = torch.log(torch.exp(logvar) + (latent_noise_scale ** 2).view(-1, 1))
        return self.decode(z, latent_noise_idx), z, mean, logvar


# Loss function
def vae_loss(recon_x, x, mean, logvar, beta=0.01, score=None, DSM=None):
    if isinstance(x, list) and isinstance(recon_x, list):
        for i in range(len(recon_x)):
            if i == 0:
                # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
            else:
                recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
    else:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='none').mean()
    kld_encoder_posterior = 0.5 * torch.mean(- 1 - logvar)
    # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
    kld_prior = 0.5 * torch.mean(mean.pow(2) + logvar.exp())
    # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
    kld_loss = kld_encoder_posterior + kld_prior
    if score is not None and DSM is None:
        kld_loss = kld_encoder_posterior - score
        kld_prior = - score
    elif DSM is not None:
        kld_loss = kld_encoder_posterior + DSM
    return recon_loss + beta * kld_loss, recon_loss, beta * kld_encoder_posterior, beta * kld_prior


def vae_loss(recon_x, x, mean, logvar, beta=0.01, score=None, DSM=None, weighting=None):
    if isinstance(x, list) and isinstance(recon_x, list):
        for i in range(len(recon_x)):
            if weighting is not None:
                if i == 0:
                    # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                    recon_loss = (weighting[:weighting.shape[0] // 2] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                               reduction='none')).mean()
                    # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                else:
                    recon_loss += (weighting[weighting.shape[0] // 2:] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                                reduction='none')).mean()
                    # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
            else:
                if i == 0:
                    # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                    recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                    # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                else:
                    recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                    # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
    else:
        recon_loss = (weighting * nn.functional.mse_loss(recon_x, x, reduction='none')).mean()
    if weighting is not None:
        kld_encoder_posterior = 0.5 * torch.mean(weighting * (- 1 - logvar))
        # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
        kld_prior = 0.5 * torch.mean(weighting * (mean.pow(2) + logvar.exp()))
        # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
        kld_loss = kld_encoder_posterior + kld_prior
    else:
        kld_encoder_posterior = 0.5 * torch.mean(- 1 - logvar)
        # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
        kld_prior = 0.5 * torch.mean(mean.pow(2) + logvar.exp())
        # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
        kld_loss = kld_encoder_posterior + kld_prior
    if score is not None and DSM is None:
        kld_loss = kld_encoder_posterior - score
        kld_prior = - score
    elif DSM is not None:
        kld_loss = kld_encoder_posterior + DSM
    return recon_loss + beta * kld_loss, recon_loss, beta * kld_encoder_posterior, beta * kld_prior


def vae_loss(recon_x, x, mean, logvar, beta=0.01, score=None, DSM=None):
    if isinstance(x, list) and isinstance(recon_x, list):
        for i in range(len(recon_x)):
            if i == 0:
                # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
            else:
                recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
    else:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='none').mean()
    kld_encoder_posterior = 0.5 * torch.mean(- 1 - logvar)
    # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
    kld_prior = 0.5 * torch.mean(mean.pow(2) + logvar.exp())
    # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
    kld_loss = kld_encoder_posterior + kld_prior
    if score is not None and DSM is None:
        kld_loss = kld_encoder_posterior - score
        kld_prior = - score
    elif DSM is not None:
        kld_loss = kld_encoder_posterior + DSM
    return recon_loss + beta * kld_loss, recon_loss, beta * kld_encoder_posterior, beta * kld_prior


def vae_loss_lambda(recon_x, x, mean, logvar, score=None, DSM=None, weighting=None):
    if isinstance(x, list) and isinstance(recon_x, list):
        for i in range(len(recon_x)):
            if weighting is not None:
                if i == 0:
                    # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                    recon_loss = (weighting[:weighting.shape[0] // 2] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                               reduction='none')).mean()
                    # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                else:
                    recon_loss += (weighting[weighting.shape[0] // 2:] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                                reduction='none')).mean()
                    # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
            else:
                if i == 0:
                    # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                    recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                    # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                else:
                    recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                    # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
    else:
        recon_loss = (weighting * nn.functional.mse_loss(recon_x, x, reduction='none')).mean()
    if weighting is not None:
        kld_encoder_posterior = 0.5 * torch.mean(weighting * (- 1 - logvar))
        # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
        kld_prior = 0.5 * torch.mean(weighting * (mean.pow(2) + logvar.exp()))
        # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
        kld_loss = kld_encoder_posterior + kld_prior
    else:
        kld_encoder_posterior = 0.5 * torch.mean(- 1 - logvar)
        # kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
        kld_prior = 0.5 * torch.mean(mean.pow(2) + logvar.exp())
        # kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
        kld_loss = kld_encoder_posterior + kld_prior
    if score is not None and DSM is None:
        kld_loss = kld_encoder_posterior - score
        kld_prior = - score
    elif DSM is not None:
        kld_loss = kld_encoder_posterior + DSM
    return recon_loss + kld_loss, recon_loss, kld_encoder_posterior, kld_prior


def compute_recon_loss(recon_x, x, weighting=None):
    if isinstance(x, list) and isinstance(recon_x, list):
        for i in range(len(recon_x)):
            if weighting is not None:
                if i == 0:
                    # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                    recon_loss = (weighting[:weighting.shape[0] // 2] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                               reduction='none')).mean()
                    # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                else:
                    recon_loss += (weighting[weighting.shape[0] // 2:] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                                reduction='none')).mean()
                    # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
            else:
                if i == 0:
                    # recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')/ratio
                    recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                    # print(nn.functional.mse_loss(recon_x[i], x[i], reduction='none').shape)
                else:
                    recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                    # recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
    else:
        recon_loss = (weighting * nn.functional.mse_loss(recon_x, x, reduction='none')).mean()

    return recon_loss


def compute_recon_loss(recon_x, x, weighting=None):
    if isinstance(x, list) and isinstance(recon_x, list):
        for i in range(len(recon_x)):
            if weighting is not None:
                if i == 0:
                    recon_loss = (weighting[:weighting.shape[0] // 2] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                               reduction='none')).mean()
                else:
                    recon_loss += (weighting[weighting.shape[0] // 2:] * nn.functional.mse_loss(recon_x[i], x[i],
                                                                                                reduction='none')).mean()
            else:
                if i == 0:
                    recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
                else:
                    recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='none').mean()
    else:
        recon_loss = (weighting * nn.functional.mse_loss(recon_x, x, reduction='none')).mean()

    # Debug check
    if recon_loss.grad_fn is None:
        raise RuntimeError('recon_loss is not connected to the computational graph')

    return recon_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define a Residual Block
class ResidualBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockVAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


# Define the Encoder with Skip Connections and Residual Blocks
class ResConvEncoder(nn.Module):
    def __init__(self, latent_size=64, type_of_dataset='mnist'):
        super(ResConvEncoder, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # (batch_size, 32, 14, 14)
        self.res1 = ResidualBlockVAE(16, 16)
        self.conv2 = nn.Conv2d(16, 64, 4, 2, 1)  # (batch_size, 64, 7, 7)
        self.res2 = ResidualBlockVAE(64, 64)
        self.type_of_dataset = type_of_dataset
        if type_of_dataset == 'svhn':
            self.res_SVHN = ResidualBlockVAE(64, 64)
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 4, 2, 1)
        elif type_of_dataset == 'mnist':
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 3, 2, 1)  # (batch_size, 128, 4, 4)
        elif type_of_dataset == 'usps':
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 3, 1, 1)  # (batch_size, 128, 4, 4)
        self.res3 = ResidualBlockVAE(2 * latent_size, 2 * latent_size)
        self.conv4 = nn.Conv2d(2 * latent_size, 2 * latent_size, 4)  # (batch_size, 128, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        if self.type_of_dataset == 'svhn':
            x = self.res_SVHN(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)

        mu, logvar = self.conv4(x).view(x.shape[0], -1).chunk(2, dim=-1)  # (batch_size, latent_size)

        mu = mu.view((-1, self.latent_size, 1, 1))
        logvar = logvar.view((-1, self.latent_size, 1, 1))

        return mu, logvar


# Define the Decoder with Skip Connections and Residual Blocks
class ResConvDecoder(nn.Module):
    def __init__(self, latent_size=64, type_of_dataset='mnist'):
        super(ResConvDecoder, self).__init__()
        self.latent_size = latent_size
        self.res1 = ResidualBlockVAE(latent_size, latent_size)
        self.conv1 = nn.ConvTranspose2d(latent_size, 64, 4)  # (batch_size, 64, 4, 4)
        self.res2 = ResidualBlockVAE(64, 64)
        self.conv2 = nn.ConvTranspose2d(64, 16, 4, 2, 1)  # (batch_size, 32, 8, 8)
        self.res3 = ResidualBlockVAE(16, 16)
        self.type_of_dataset = type_of_dataset
        if type_of_dataset == 'svhn':
            self.res_SVHN = ResidualBlockVAE(64, 64)
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 0)
        elif type_of_dataset == 'mnist':
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 2)
        elif type_of_dataset == 'usps':
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 2, 1)

    def forward(self, x):
        x = self.res1(x)
        x = F.relu(self.conv1(x))
        x = self.res2(x)
        if self.type_of_dataset == 'svhn':
            x = self.res_SVHN(x)
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for binary image
        return x


# Define the VAE with Skip Connections and Residual Blocks
class ResConvVAE(nn.Module):
    def __init__(self, latent_height=8, latent_width=8, type_of_dataset='mnist'):
        super(ResConvVAE, self).__init__()
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.encoder = ResConvEncoder(latent_size=latent_height * latent_width, type_of_dataset=type_of_dataset)
        self.decoder = ResConvDecoder(latent_size=latent_height * latent_width, type_of_dataset=type_of_dataset)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z.view((z.shape[0], -1, 1, 1)))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z.view((z.shape[0], 1, self.latent_height, self.latent_width)), mu.view(
            (z.shape[0], -1)), logvar.view((z.shape[0], -1))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define a Residual Block
class ResidualBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockVAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


# Define the Encoder with Skip Connections and Residual Blocks
class ResConvEncoder(nn.Module):
    def __init__(self, latent_size=64, type_of_dataset='mnist', is_2d=False, is_3d=False):
        super(ResConvEncoder, self).__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # (batch_size, 32, 14, 14)
        self.res1 = ResidualBlockVAE(16, 16)
        self.conv2 = nn.Conv2d(16, 64, 4, 2, 1)  # (batch_size, 64, 7, 7)
        self.res2 = ResidualBlockVAE(64, 64)
        self.is_2d = is_2d
        self.is_3d = is_3d
        self.type_of_dataset = type_of_dataset
        if is_2d and not is_3d:
            self.lin = nn.Linear(128, 4)
            self.bn2 = nn.BatchNorm1d(2, affine=False)
        elif not is_2d and is_3d:
            self.lin = nn.Linear(128, 6)
            self.bn2 = nn.BatchNorm1d(3, affine=False)
        if type_of_dataset == 'svhn':
            self.res_SVHN1 = ResidualBlockVAE(64, 64)
            self.res_SVHN2 = ResidualBlockVAE(64, 64)
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 4, 2, 1)
            self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)
        elif type_of_dataset == 'mnist':
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 3, 2, 1)  # (batch_size, 128, 4, 4)
        elif type_of_dataset == 'usps':
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 3, 1, 1)  # (batch_size, 128, 4, 4)
        self.res3 = ResidualBlockVAE(2 * latent_size, 2 * latent_size)
        self.conv4 = nn.Conv2d(2 * latent_size, 2 * latent_size, 4)  # (batch_size, 128, 1, 1)
        self.bn_mu = nn.BatchNorm1d(latent_size, affine=False)
        self.bn_var = nn.BatchNorm1d(latent_size, affine=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        if self.type_of_dataset == 'svhn':
            x = self.res_SVHN1(x)
            x = self.res_SVHN2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        x = self.conv4(x).view(x.shape[0], -1)  # (batch_size, latent_size)

        if self.is_2d or self.is_3d:
            x = self.lin(F.relu(x).view(x.shape[0], -1))
            mu, logvar = x.chunk(2, dim=-1)
            mu = self.bn2(mu)
            logvar = 7 * (torch.sigmoid(logvar) - 0.5)
            # logvar = - F.leaky_relu(logvar, 0.1)
        else:
            mu, logvar = x.chunk(2, dim=-1)
            # logvar = 10 * F.tanh(logvar)
            logvar = 7 * (torch.sigmoid(logvar) - 0.5)
            # logvar = F.leaky_relu(logvar, 0.1) # For stability reasons
            print("Here!!!!!", mu.shape)
            mu = self.bn_mu(mu.view(mu.shape[0], -1))
            logvar = logvar.view(logvar.shape[0], -1)

        if not self.is_2d and not self.is_3d:
            mu = mu.view((-1, self.latent_size, 1, 1))
            logvar = logvar.view((-1, self.latent_size, 1, 1))

        return mu, logvar


# Define the Decoder with Skip Connections and Residual Blocks
class ResConvDecoder(nn.Module):
    def __init__(self, latent_size=64, type_of_dataset='mnist', is_2d=False, is_3d=False):
        super(ResConvDecoder, self).__init__()
        self.latent_size = latent_size
        self.res1 = ResidualBlockVAE(latent_size, latent_size)
        self.conv1 = nn.ConvTranspose2d(latent_size, 64, 4)  # (batch_size, 64, 4, 4)
        self.res2 = ResidualBlockVAE(64, 64)
        self.conv2 = nn.ConvTranspose2d(64, 16, 4, 2, 1)  # (batch_size, 32, 8, 8)
        self.res3 = ResidualBlockVAE(16, 16)
        self.is_2d = is_2d
        self.is_3d = is_3d
        self.type_of_dataset = type_of_dataset
        if is_2d and not is_3d:
            self.lin = nn.Linear(2, 64)
        elif not is_2d and is_3d:
            self.lin = nn.Linear(3, 64)
        if type_of_dataset == 'svhn':
            self.res_SVHN1 = ResidualBlockVAE(64, 64)
            self.res_SVHN2 = ResidualBlockVAE(64, 64)
            self.conv3 = nn.ConvTranspose2d(16, 3, 4, 4, 0)
        elif type_of_dataset == 'mnist':
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 2)
        elif type_of_dataset == 'usps':
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 2, 1)

    def forward(self, x):
        if self.is_2d or self.is_3d:
            x = self.lin(x)
            x = x.view(-1, 64, 1, 1)
        x = self.res1(x)
        x = F.relu(self.conv1(x))
        x = self.res2(x)
        if self.type_of_dataset == 'svhn':
            x = self.res_SVHN1(x)
            x = self.res_SVHN2(x)
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for binary image

        return x


# Define the VAE with Skip Connections and Residual Blocks
class ResConvVAE(nn.Module):
    def __init__(self, latent_height=8, latent_width=8, type_of_dataset='mnist', is_2d=False, is_3d=False):
        super(ResConvVAE, self).__init__()
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.encoder = ResConvEncoder(latent_size=latent_height * latent_width, type_of_dataset=type_of_dataset,
                                      is_2d=is_2d, is_3d=is_3d)
        self.decoder = ResConvDecoder(latent_size=latent_height * latent_width, type_of_dataset=type_of_dataset,
                                      is_2d=is_2d, is_3d=is_3d)

        self.is_2d = is_2d
        self.is_3d = is_3d

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        if self.is_2d or self.is_3d:
            return self.decoder(z.view((z.shape[0], -1)))
        return self.decoder(z.view((z.shape[0], -1, 1, 1)))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z, mu.view((z.shape[0], -1)), logvar.view((z.shape[0], -1))

    def init_weights_fixed(self, seed=42):
        """
        Initialize all the weights of the model to the same small random values.
        """
        torch.manual_seed(seed)  # Set a fixed seed for reproducibility

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(weights_init)


import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.swish = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity  # Skip connection
        return self.swish(out)

# UNet with advanced techniques
class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, num_latent_noise_scale=50, is_add_latent_noise=False, multiplier=4, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), is_warm_init=False):
        super(UNet, self).__init__()
        self.num_timesteps = num_timesteps
        self.device = device
        self.is_add_latent_noise = is_add_latent_noise

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(multiplier*in_dim + embedding_dim, multiplier*in_dim) if not is_add_latent_noise else nn.Linear(multiplier*in_dim + embedding_dim + embedding_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            nn.Linear(multiplier*in_dim, out_dim)
        )

        # Define time step embedding layer for decoder
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)
        if self.is_add_latent_noise:
            self.latent_noise_embedding = nn.Embedding(num_latent_noise_scale, embedding_dim)
        if is_warm_init:
            self.warm_init()

    def forward(self, x, timestep, latent_noise_idx=None, enc_sigma=None):
        # Encoder
        if enc_sigma is not None:
            encoded_enc_sigma = self.encoder(enc_sigma)
        else:
            encoded_enc_sigma = 0
        x = self.encoder(x) + encoded_enc_sigma

        # Decoder
        if self.is_add_latent_noise:
            if latent_noise_idx is None:
                latent_noise_idx = torch.zeros(x.shape[0], device=x.device).type(torch.long)
            x = self.decoder(torch.hstack((x, self.embedding(timestep), self.latent_noise_embedding(latent_noise_idx))))
        else:
            x = self.decoder(torch.hstack((x, self.embedding(timestep))))

        return x

    def warm_init(self):
        # Custom initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = self._clone_model_params()

    def _clone_model_params(self):
        shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shadow[name] = param.data.clone()
        return shadow

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()


class UNet_simple(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, num_latent_noise_scale=50, is_add_latent_noise=False, multiplier=4, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), is_warm_init=False):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.device = device
        self.is_add_latent_noise = is_add_latent_noise

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(multiplier*in_dim + embedding_dim, multiplier*in_dim) if not is_add_latent_noise else nn.Linear(multiplier*in_dim + embedding_dim + embedding_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Linear(multiplier*in_dim, out_dim)
        )

        # Define time step embedding layer for decoder
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)
        if self.is_add_latent_noise:
            self.latent_noise_embedding = nn.Embedding(num_latent_noise_scale, embedding_dim)
        if is_warm_init:
            self.warm_init()

    def forward(self, x, timestep, latent_noise_idx=None, enc_sigma=None):
        # Encoder
        if enc_sigma is not None:
            encoded_enc_sigma = self.encoder(enc_sigma)
        else:
            encoded_enc_sigma = 0
        x = self.encoder(x) + encoded_enc_sigma

        # Decoder
        if self.is_add_latent_noise:
            if latent_noise_idx is None:
                latent_noise_idx = torch.zeros(x.shape[0], device=x.device).type(torch.long)
            x = self.decoder(torch.hstack((x, self.embedding(timestep), self.latent_noise_embedding(latent_noise_idx))))
        else:
            x = self.decoder(torch.hstack((x, self.embedding(timestep))))

        return x

    def warm_init(self):
        # Custom initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

class Score_fn(nn.Module):
    def __init__(self, model, ema=None, ema_decay=0.99, sigma_min=0.01, sigma_max=50, num_timesteps=1000,
                 is_add_latent_noise=False, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """Construct a score function model.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          num_timestep: number of discretization steps
        """
        super(Score_fn, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigma = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), num_timesteps)).to(device)
        self.num_timesteps = num_timesteps
        self.model = model
        self.device = device
        self.loss_dict = {}
        self.total_loss = 0
        self.loss_counter = 0
        self.is_add_latent_noise = is_add_latent_noise
        if ema is not None:
            self.ema = ema(model, decay=ema_decay)

        # Learnable parameter for residual score function and assures value between [0,1]
        self.lbda = nn.ParameterList([nn.Parameter(torch.tensor([0.0]))])

    def to_device(self):
        self.model = self.model.to(self.device)

    # Compute denoising score matching loss
    def compute_DSM_loss(self, x, t, latent_noise_idx=None, enc_mu=None, enc_sigma=None, alpha=None,
                         turn_off_enc_sigma=False, learn_lbda=False, is_mixing=False, is_residual=False,
                         is_vanilla=False, is_LSGM=False, divide_by_sigma=False):
        sigmas = self.discrete_sigma[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x, device=self.device) * sigmas
        perturbed_data = x + noise
        if is_mixing:
            if self.is_add_latent_noise:
                score = self.get_mixing_score_fn(perturbed_data, t, latent_noise_idx=latent_noise_idx, alpha=alpha,
                                                 is_residual=is_residual, is_vanilla=is_vanilla,
                                                 divide_by_sigma=divide_by_sigma)
            else:
                score = self.get_mixing_score_fn(perturbed_data, t, alpha=alpha, is_residual=is_residual,
                                                 is_vanilla=is_vanilla, divide_by_sigma=divide_by_sigma)
        elif is_residual:
            enc_eps = x - enc_mu
            if self.is_add_latent_noise:
                score = self.get_residual_score_fn(perturbed_data, t, latent_noise_idx, enc_eps, enc_sigma,
                                                   turn_off_enc_sigma, learn_lbda, is_vanilla=is_vanilla,
                                                   divide_by_sigma=divide_by_sigma)
            else:
                score = self.get_residual_score_fn(perturbed_data, t, enc_eps, enc_sigma, turn_off_enc_sigma,
                                                   learn_lbda, is_vanilla=is_vanilla, divide_by_sigma=divide_by_sigma)
        else:
            score = self.get_score_fn(perturbed_data, t)
        target = -noise / (sigmas ** 2)
        losses = torch.square(score - target)
        losses = 1 / 2. * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas.squeeze() ** 2
        if is_LSGM:
            return torch.sum(losses)
        else:
            return torch.mean(losses)

    # Get score function
    def get_score_fn(self, x, t, latent_noise_idx=None, detach=False):
        if detach:
            self.model.eval()
            if self.is_add_latent_noise:
                return (self.model(x, t, latent_noise_idx=latent_noise_idx) / self.discrete_sigma[t.long()].view(
                    x.shape[0], *([1] * len(x.shape[1:])))).detach()
            else:
                return (self.model(x, t) / self.discrete_sigma[t.long()].view(x.shape[0],
                                                                              *([1] * len(x.shape[1:])))).detach()
        else:
            if self.is_add_latent_noise:
                return self.model(x, t, latent_noise_idx=latent_noise_idx) / self.discrete_sigma[t.long()].view(
                    x.shape[0], *([1] * len(x.shape[1:])))
            else:
                return self.model(x, t) / self.discrete_sigma[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))

    # Our implementation of residual score function
    def get_residual_score_fn(self, x, t, enc_eps, enc_sigma, detach=False, turn_off_enc_sigma=False, learn_lbda=False):

        # turn on eval for detach
        if detach:
            self.model.eval()

        # Computes learnable score
        learnable_score = self.model(x, t) / self.discrete_sigma[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))

        # Learns lbda hyperparameter
        if learn_lbda:
            learnable_score = self.lbda * learnable_score

        # Makes the variance equal 1 when turned off and variance equal to the encoder variance
        if turn_off_enc_sigma:
            residual_score = - enc_eps
        else:
            residual_score = - enc_eps / (enc_sigma ** 2)
        if detach:
            self.model.train()
            return (learnable_score + residual_score).detach()
        else:
            return learnable_score + residual_score

    # Training LSGM Mixing Normal and Neural Score Functions based on this paper https://arxiv.org/pdf/2106.05931
    # if no alpha param is given assumed alpha is learned by the model. If it is residual behaves like Prof. Inouye's idea
    def get_mixing_score_fn(self, x, t, latent_noise_idx=None, alpha=None, is_residual=False, is_vanilla=False,
                            detach=False, divide_by_sigma=False):

        if detach:
            self.model.eval()

        # Converts lbda to alpha to match LGSM notation and bounds [0, 1]
        if alpha is None:
            # alpha = torch.relu(torch.tanh(self.lbda[0]))
            alpha = torch.sigmoid(self.lbda[0])
            # print(f"alpha: {alpha}")
        else:
            alpha = alpha.to(self.device)

        if divide_by_sigma:
            if self.is_add_latent_noise:
                learnable_score = alpha * self.model(x, t, latent_noise_idx) / self.discrete_sigma[t.long()].view(
                    x.shape[0], *([1] * len(x.shape[1:])))
            else:
                learnable_score = alpha * self.model(x, t) / self.discrete_sigma[t.long()].view(x.shape[0], *(
                            [1] * len(x.shape[1:])))
        else:
            if self.is_add_latent_noise:
                learnable_score = alpha * self.model(x, t, latent_noise_idx)
            else:
                learnable_score = alpha * self.model(x, t)

        # Turning on the residual flag is identical to Prof. Inouye's method
        if is_residual:
            residual_score = - x
        else:
            residual_score = - (1 - alpha) * x

        if detach:
            if is_vanilla:
                return learnable_score.detach()
            self.model.train()
            return (learnable_score + residual_score).detach()
        else:
            if is_vanilla:
                return learnable_score
            return learnable_score + residual_score

    def get_LSGM_loss(self, x, t=None, latent_noise_idx=None, is_mixing=False, is_residual=False, is_vanilla=False,
                      alpha=None):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device)
        if self.is_add_latent_noise:
            loss = self.compute_DSM_loss(x, t, latent_noise_idx=latent_noise_idx, is_mixing=is_mixing,
                                         is_residual=is_residual, alpha=alpha, is_vanilla=is_vanilla, is_LSGM=True,
                                         divide_by_sigma=True)
        else:
            loss = self.compute_DSM_loss(x, t, is_mixing=is_mixing, is_residual=is_residual, alpha=alpha,
                                         is_vanilla=is_vanilla, is_LSGM=True, divide_by_sigma=True)
        return loss

    # Update one batch and add shrink the max timestep for reducing the variance range of training (default is equal to defined num_timestep).
    # When verbose is true, gets the average loss up until last verbose and saves to loss dict
    def update_score_fn(self, x, optimizer, latent_noise_idx=None, alpha=None, max_timestep=None, t=None, verbose=False,
                        is_mixing=False, is_residual=False, is_vanilla=False, divide_by_sigma=False):
        # TODO: Add ema optimization
        if max_timestep is None or max_timestep > self.num_timesteps:
            max_timestep = self.num_timesteps

        if t is None:
            t = torch.randint(0, max_timestep, (x.shape[0],), device=self.device)

        if self.is_add_latent_noise:
            loss = self.compute_DSM_loss(x, t, latent_noise_idx, is_mixing=is_mixing, is_residual=is_residual,
                                         alpha=alpha, is_vanilla=is_vanilla, divide_by_sigma=False)
        else:
            loss = self.compute_DSM_loss(x, t, is_mixing=is_mixing, is_residual=is_residual, alpha=alpha,
                                         is_vanilla=is_vanilla, divide_by_sigma=False)

        self.total_loss += loss.item()
        self.loss_counter += 1.
        if verbose:
            avg_loss = self.total_loss / self.loss_counter
            self.reset_loss_count()
            self.update_loss_dict(avg_loss)
            # print(avg_loss)
            # print(f'alpha: {torch.sigmoid(self.lbda[0])}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA
        if hasattr(self, 'ema'):
            self.ema.update()

        return loss

    # Update for residual score model training
    def update_residual_score_fn(self, x, enc_mu, enc_sigma, optimizer, max_timestep=None, learn_lbda=False,
                                 turn_off_enc_sigma=False, t=None, verbose=False):
        if max_timestep is None or max_timestep > self.num_timesteps:
            max_timestep = self.num_timesteps

        if t is None:
            t = torch.randint(0, max_timestep, (x.shape[0],), device=self.device)

        loss = self.compute_DSM_loss(x, t, is_residual=True, enc_mu=enc_mu, enc_sigma=enc_sigma,
                                     turn_off_enc_sigma=turn_off_enc_sigma, learn_lbda=learn_lbda)

        self.total_loss += loss.item()
        self.loss_counter += 1.
        if verbose:
            avg_loss = self.total_loss / self.loss_counter
            self.reset_loss_count()
            self.update_loss_dict(avg_loss)
            print(avg_loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Update EMA
        if hasattr(self, 'ema'):
            self.ema.update()

    def add_EMA_training(self, ema, decay=0.99):
        self.ema = ema(self.model, decay)

    def update_param_with_EMA(self):
        if hasattr(self, 'ema'):
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.ema.shadow:
                    param.data.copy_(self.ema.shadow[name])
        else:
            raise AttributeError(
                "EMA model is not defined in the class. Please use add_EMA_training class function and retrain")

    # Draws a vector field of the score function
    def draw_gradient_field(self, xlim, ylim, t=0, x_num=20, y_num=20, file="./Score_Function", noise_label=1,
                            save=False, data=None, labels=None, n_samples=100, alpha=None, is_mixture=False,
                            is_residual=False, is_vanilla=False):
        x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], x_num), np.linspace(ylim[0], ylim[1], y_num))
        x_ = torch.from_numpy(x.reshape(-1, 1)).type(torch.float).to(self.device)
        y_ = torch.from_numpy(y.reshape(-1, 1)).type(torch.float).to(self.device)

        input = torch.hstack((x_, y_))

        if data is not None:
            if isinstance(data, torch.Tensor):
                data = data.detach()
                if data.is_cuda:
                    data = data.cpu().numpy()
            else:
                return data

            if labels is not None:
                data1, data2 = data.chunk(2)
                labels1, labels2 = labels.view((-1,)).chunk(2)
                data1_l1, data1_l2 = data1[labels1 == 0], data1[labels1 == 1]
                data2_l1, data2_l2 = data2[labels2 == 0], data2[labels1 == 1]
                plt.scatter(data1_l1[:n_samples, 0], data1_l1[:n_samples, 1], marker='x', label='D1_L1', c='b', s=20)
                plt.scatter(data1_l2[:n_samples, 0], data1_l2[:n_samples, 1], marker='o', label='D1_L2', c='b', s=20)
                plt.scatter(data2_l1[:n_samples, 0], data2_l1[:n_samples, 1], marker='+', label='D2_L1', c='g', s=20)
                plt.scatter(data2_l2[:n_samples, 0], data2_l2[:n_samples, 1], marker='o', label='D2_L2', c='g', s=20)
                plt.legend()
            else:
                plt.scatter(data[:, 0], data[:, 1])

        if is_mixture:
            score_fn = self.get_mixing_score_fn(input,
                                                torch.ones((x_num * y_num,), device=self.device).type(torch.long) * t,
                                                detach=True, alpha=alpha, is_vanilla=is_vanilla)
        elif is_residual:
            score_fn = self.get_mixing_score_fn(input,
                                                torch.ones((x_num * y_num,), device=self.device).type(torch.long) * t,
                                                detach=True, alpha=alpha, is_residual=True, is_vanilla=is_vanilla)
        else:
            score_fn = self.get_score_fn(input, torch.ones((x_num * y_num,), device=self.device).type(torch.long) * t,
                                         detach=True)

        score_fn_x = score_fn[:, 0].cpu().numpy().reshape(x_num, y_num)
        score_fn_y = score_fn[:, 1].cpu().numpy().reshape(x_num, y_num)
        plt.quiver(x, y, score_fn_x, score_fn_y, color='r')
        plt.title('Score Function')
        plt.grid()
        plt.show()
        if save:
            plt.savefig(f"{file}")

    # Resets the total loss and respective count of updates
    def reset_loss_count(self):
        self.total_loss = 0
        self.loss_counter = 0

    def update_loss_dict(self, loss):
        if not self.loss_dict:
            self.loss_dict.update({'DSMloss': [loss]})
        else:
            self.loss_dict['DSMloss'].append(loss)

    def get_loss_dict(self):
        return self.loss_dict


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from openTSNE import TSNE
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import umap
from torch.utils.data import Dataset, DataLoader


def normalize_tensor_per_dim(tensor):
    # Initialize an empty tensor to store the normalized values
    normalized_tensor = torch.empty_like(tensor)

    # Iterate over each dimension (here, each column)
    for i in range(tensor.shape[1]):
        min_val = tensor[:, i].min()
        max_val = tensor[:, i].max()
        normalized_tensor[:, i] = (tensor[:, i] - min_val) / (max_val - min_val)

    return normalized_tensor


def normalize_np_array(array):
    min_vals = np.min(array, axis=0)
    max_vals = np.max(array, axis=0)
    normalized_array = (array - min_vals) / (max_vals - min_vals)
    return normalized_array


class CustomDataset(Dataset):
    def __init__(self, img, emb, label):
        # self.img = torch.tensor(img, dtype=torch.float32)
        # self.emb = torch.tensor(emb, dtype=torch.float32)
        # self.label = torch.tensor(label, dtype=torch.long)
        self.img = img
        self.emb = emb
        self.label = label

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img, emb, label = self.img[idx], self.emb[idx], self.label[idx]
        return img, emb, label


def apply_umap_and_create_dataset(data, labels=None, n_dim=2, n_neighbors=10, min_dist=0.1, random_state=42,
                                  output_file="reduced_mnist.npy", batch_size=64):
    # Apply UMAP to reduce dimensions
    reducer = umap.UMAP(
        n_components=n_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='euclidean',
        verbose=True
    )

    # Fit UMAP and transform data
    reduced_data = reducer.fit_transform(data)

    # Concatenate labels if provided
    if labels is not None:
        reduced_data = np.hstack((reduced_data, labels.reshape(-1, 1)))

    # Save the reduced data to a NumPy file
    np.save(output_file, reduced_data)
    print(f"Dataset saved to {output_file}")

    # Create a custom PyTorch dataset
    dataset = CustomDataset(reduced_data, labels)

    return dataset


def plot_umap(data, labels=None, n_dim=2, n_neighbors=15, min_dist=0.1, markers=['.', '+'], is_normalize=False,
              n_samples=1500):
    """
    Applies UMAP to reduce the dimensions of the input data and plots the results.

    Parameters:
    - data: Input data (features)
    - labels: Labels for the input data (optional)
    - n_dim: Number of dimensions for the reduced data (default: 2)
    - n_neighbors: Number of neighbors to consider for UMAP (default: 15)
    - min_dist: Minimum distance between points in the embedded space (default: 0.1)
    - random_state: Random seed for reproducibility (default: 42)

    Returns:
    - reduced_data: Reduced data after applying UMAP
    """
    # Apply UMAP to reduce dimensions
    reducer = umap.UMAP(
        n_components=n_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        verbose=False
    )
    plt.figure(figsize=(10, 8))
    if isinstance(data, list):
        data_concat = np.concatenate(data, axis=0)
        reducer.fit(data_concat)
        for i in range(len(data)):
            if labels is not None and markers is not None:
                data_chunk = reducer.transform(data[i])
                if is_normalize:
                    data_chunk = normalize_np_array(data_chunk)
                plt.scatter(data_chunk[:n_samples, 0], data_chunk[:n_samples, 1], c=labels[i][:n_samples],
                            marker=markers[i], cmap='tab10', s=20)
                plt.colorbar()
            # else:
                # plt.scatter(reduced_data[:n_samples, 0], reduced_data[:n_samples, 1], s=10)
    else:
        # Fit UMAP and transform data
        reduced_data = reducer.fit_transform(data)

        if labels is not None:
            plt.scatter(reduced_data[:n_samples, 0], reduced_data[:n_samples, 1], c=labels[:n_samples], cmap='tab10',
                        s=10)
            plt.colorbar()
        else:
            plt.scatter(reduced_data[:n_samples, 0], reduced_data[:n_samples, 1], s=10)
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()


# Function to load MNIST dataset
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    return next(iter(train_loader))


# Function to load SVHN dataset
def load_svhn():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    return next(iter(train_loader))


# Function to load USPS dataset
def load_usps():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    return next(iter(train_loader))


# Function to plot t-SNE
def plot_tsne(data, labels, title):
    data_gpu = cp.asarray(data)
    tsne = TSNE(n_components=2, n_jobs=-1, verbose=True)
    data_2d = tsne.fit(data_gpu)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()


def apply_tsne(data, labels=None, n_dim=2, perplexity=30, learning_rate=200, n_iter=300):
    # Apply t-SNE to reduce dimensions
    data_gpu = cp.asarray(data)
    tsne = TSNE(n_components=n_dim, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                neighbors="exact", negative_gradient_method="bh", n_jobs=-1, verbose=True, random_state=42)
    reduced_data = tsne.fit(data_gpu)

    # Concatenate labels if provided
    if labels is not None:
        reduced_data = np.concatenate((reduced_data, labels.reshape(-1, 1)), axis=1)

    return reduced_data


def apply_tsne_and_save(data, labels=None, n_dim=2, perplexity=30, learning_rate=200, n_iter=300,
                        output_file="reduced_data.npy"):
    # Convert data to CuPy array for GPU computation
    data_gpu = cp.asarray(data)

    # Apply t-SNE to reduce dimensions
    tsne = TSNE(n_components=n_dim, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                neighbors="exact", negative_gradient_method="bh", n_jobs=-1, verbose=True, random_state=42)
    reduced_data_gpu = tsne.fit(data_gpu)

    # Convert reduced data back to NumPy array
    reduced_data = cp.asnumpy(reduced_data_gpu)

    # Concatenate labels if provided
    if labels is not None:
        reduced_data = np.concatenate((reduced_data, labels.reshape(-1, 1)), axis=1)

    # Save the reduced data to a NumPy file
    np.save(output_file, reduced_data)
    print(f"Dataset saved to {output_file}")


def plot_epochs_values(epochs_values, title='', xlabel='', ylabel='', legend_names=[], colors=None, linestyles=None):
    """
    Plot line charts for epochs values with customizable colors and line styles.

    Parameters:
    - epochs_values (list of lists): List of lists where each sublist contains values for epochs.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - legend_names (list of str): List of legend names corresponding to each sublist.
    - colors (list of str, optional): List of colors for each sublist. Defaults to None (uses default colors).
    - linestyles (list of str, optional): List of line styles for each sublist. Defaults to None (uses solid lines).

    Returns:
    - None (displays the plot).
    """
    if len(epochs_values) != len(legend_names):
        raise ValueError("Number of legend names must match the number of lists in epochs_values")

    plt.figure(figsize=(10, 6))  # Adjust figure size if needed

    for idx, (values, legend_name) in enumerate(zip(epochs_values, legend_names)):
        color = colors[idx] if colors else None
        linestyle = linestyles[idx] if linestyles else '-'
        plt.plot(range(1, len(values) + 1), values, label=legend_name, color=color, linestyle=linestyle)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def rbf_kernel(x, y):
    """Compute the RBF kernel between two sets of samples with a default bandwidth."""
    gamma = 1.0 / x.shape[1]  # Default bandwidth
    xx = torch.sum(x ** 2, dim=1).view(-1, 1)
    yy = torch.sum(y ** 2, dim=1).view(1, -1)
    dists = xx + yy - 2 * torch.matmul(x, y.t())
    return torch.exp(-gamma * dists)


def calculate_mmd_loss(X, Y, kernel=rbf_kernel):
    """Compute the MMD loss between two sets of samples."""
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    XX = kernel(X, X)
    YY = kernel(Y, Y)
    XY = kernel(X, Y)

    # Calculate MMD using the empirical estimates
    mmd_loss = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd_loss


def load_embeddings_to_dataset(file_path):
    # Load the saved UMAP embeddings from file
    data = torch.load(file_path)

    # print(len(data))

    # Separate data and labels
    img = data[0]  # Exclude the last column which is assumed to be labels
    emb = data[1]
    label = data[2]

    dataset = CustomDataset(img=img, emb=emb, label=label)

    return dataset


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from openTSNE import TSNE
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import umap
from torch.utils.data import Dataset, DataLoader


def normalize_tensor_per_dim(tensor):
    # Initialize an empty tensor to store the normalized values
    normalized_tensor = torch.empty_like(tensor)

    # Iterate over each dimension (here, each column)
    for i in range(tensor.shape[1]):
        min_val = tensor[:, i].min()
        max_val = tensor[:, i].max()
        normalized_tensor[:, i] = (tensor[:, i] - min_val) / (max_val - min_val)

    return normalized_tensor


def normalize_np_array(array):
    min_vals = np.min(array, axis=0)
    max_vals = np.max(array, axis=0)
    normalized_array = (array - min_vals) / (max_vals - min_vals)
    return normalized_array


def apply_umap_and_create_dataset(data, labels=None, n_dim=2, n_neighbors=10, min_dist=0.1, random_state=42,
                                  output_file="reduced_mnist.npy", batch_size=64):
    # Apply UMAP to reduce dimensions
    reducer = umap.UMAP(
        n_components=n_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='euclidean',
        verbose=True
    )

    # Fit UMAP and transform data
    reduced_data = reducer.fit_transform(data)

    # Concatenate labels if provided
    if labels is not None:
        reduced_data = np.hstack((reduced_data, labels.reshape(-1, 1)))

    # Save the reduced data to a NumPy file
    np.save(output_file, reduced_data)
    print(f"Dataset saved to {output_file}")

    # Create a custom PyTorch dataset
    dataset = CustomDataset(reduced_data, labels)

    return dataset


def plot_umap(data, labels=None, n_dim=2, n_neighbors=15, min_dist=0.1, markers=['.', '+'], is_normalize=False,
              n_samples=3000):
    """
    Applies UMAP to reduce the dimensions of the input data and plots the results.

    Parameters:
    - data: Input data (features)
    - labels: Labels for the input data (optional)
    - n_dim: Number of dimensions for the reduced data (default: 2)
    - n_neighbors: Number of neighbors to consider for UMAP (default: 15)
    - min_dist: Minimum distance between points in the embedded space (default: 0.1)
    - random_state: Random seed for reproducibility (default: 42)

    Returns:
    - reduced_data: Reduced data after applying UMAP
    """
    # Apply UMAP to reduce dimensions
    reducer = umap.UMAP(
        n_components=n_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        verbose=False
    )
    plt.figure(figsize=(10, 8))
    if isinstance(data, list):
        data_concat = np.concatenate(data, axis=0)
        # print(data_concat.shape)
        reducer.fit(data_concat)
        for i in range(len(data)):
            if labels is not None and markers is not None:
                data_chunk = reducer.transform(data[i])
                if is_normalize:
                    data_chunk = normalize_np_array(data_chunk)
                plt.scatter(data_chunk[:n_samples, 0], data_chunk[:n_samples, 1], c=labels[i][:n_samples],
                            marker=markers[i], cmap='tab10', s=20)
                plt.colorbar()
            # else:
            #     plt.scatter(reduced_data[:n_samples, 0], reduced_data[:n_samples, 1], s=10)
    else:
        # Fit UMAP and transform data
        reduced_data = reducer.fit_transform(data)

        if labels is not None:
            plt.scatter(reduced_data[:n_samples, 0], reduced_data[:n_samples, 1], c=labels[:n_samples], cmap='tab10',
                        s=10)
            plt.colorbar()
        else:
            plt.scatter(reduced_data[:n_samples, 0], reduced_data[:n_samples, 1], s=10)
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()


# Function to load MNIST dataset
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    return next(iter(train_loader))


# Function to load SVHN dataset
def load_svhn():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    return next(iter(train_loader))


# Function to load USPS dataset
def load_usps():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    return next(iter(train_loader))


# Function to plot t-SNE
def plot_tsne(data, labels, title):
    data_gpu = cp.asarray(data)
    tsne = TSNE(n_components=2, n_jobs=-1, verbose=True)
    data_2d = tsne.fit(data_gpu)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()


def apply_tsne(data, labels=None, n_dim=2, perplexity=30, learning_rate=200, n_iter=300):
    # Apply t-SNE to reduce dimensions
    data_gpu = cp.asarray(data)
    tsne = TSNE(n_components=n_dim, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                neighbors="exact", negative_gradient_method="bh", n_jobs=-1, verbose=True, random_state=42)
    reduced_data = tsne.fit(data_gpu)

    # Concatenate labels if provided
    if labels is not None:
        reduced_data = np.concatenate((reduced_data, labels.reshape(-1, 1)), axis=1)

    return reduced_data


def apply_tsne_and_save(data, labels=None, n_dim=2, perplexity=30, learning_rate=200, n_iter=300,
                        output_file="reduced_data.npy"):
    # Convert data to CuPy array for GPU computation
    data_gpu = cp.asarray(data)

    # Apply t-SNE to reduce dimensions
    tsne = TSNE(n_components=n_dim, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                neighbors="exact", negative_gradient_method="bh", n_jobs=-1, verbose=True, random_state=42)
    reduced_data_gpu = tsne.fit(data_gpu)

    # Convert reduced data back to NumPy array
    reduced_data = cp.asnumpy(reduced_data_gpu)

    # Concatenate labels if provided
    if labels is not None:
        reduced_data = np.concatenate((reduced_data, labels.reshape(-1, 1)), axis=1)

    # Save the reduced data to a NumPy file
    np.save(output_file, reduced_data)
    print(f"Dataset saved to {output_file}")


def load_umap_embeddings_to_dataset(file_path, is_invert=False):
    # Load the saved UMAP embeddings from file
    umap_data = np.load(file_path)

    # Separate data and labels
    data = umap_data[:, :-1]  # Exclude the last column which is assumed to be labels
    if is_invert:
        data = 1 - data
    labels = umap_data[:, -1]  # Last column as labels

    dataset = CustomDataset(data, labels)

    return dataset


def plot_epochs_values(epochs_values, title='', xlabel='', ylabel='', legend_names=[], colors=None, linestyles=None):
    """
    Plot line charts for epochs values with customizable colors and line styles.

    Parameters:
    - epochs_values (list of lists): List of lists where each sublist contains values for epochs.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - legend_names (list of str): List of legend names corresponding to each sublist.
    - colors (list of str, optional): List of colors for each sublist. Defaults to None (uses default colors).
    - linestyles (list of str, optional): List of line styles for each sublist. Defaults to None (uses solid lines).

    Returns:
    - None (displays the plot).
    """
    if len(epochs_values) != len(legend_names):
        raise ValueError("Number of legend names must match the number of lists in epochs_values")

    plt.figure(figsize=(10, 6))  # Adjust figure size if needed

    for idx, (values, legend_name) in enumerate(zip(epochs_values, legend_names)):
        color = colors[idx] if colors else None
        linestyle = linestyles[idx] if linestyles else '-'
        plt.plot(range(1, len(values) + 1), values, label=legend_name, color=color, linestyle=linestyle)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def rbf_kernel(x, y):
    """Compute the RBF kernel between two sets of samples with a default bandwidth."""
    gamma = 1.0 / x.shape[1]  # Default bandwidth
    xx = torch.sum(x ** 2, dim=1).view(-1, 1)
    yy = torch.sum(y ** 2, dim=1).view(1, -1)
    dists = xx + yy - 2 * torch.matmul(x, y.t())
    return torch.exp(-gamma * dists)


def calculate_mmd_loss(X, Y, kernel=rbf_kernel):
    """Compute the MMD loss between two sets of samples."""
    X = X.view(X.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)
    XX = kernel(X, X)
    YY = kernel(Y, Y)
    XY = kernel(X, Y)

    # Calculate MMD using the empirical estimates
    mmd_loss = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd_loss


import math


def gp_weight_fn(x, weight=0.25):
    """
    Applies a modified exponential decay function to the input tensor x,
    preserving the sign of the original values.

    Parameters:
    x (torch.Tensor): The input tensor.
    weight (float): The weighting factor to control the decay.

    Returns:
    torch.Tensor: The weighted tensor with the original sign preserved.
    """
    return x ** weight


def compute_gp_loss(x, z, dist_func_x, dist_func_z):
    batch_size = len(x)
    loss = 0
    for idx in range(batch_size - 1):
        p_dist_x = dist_func_x(x[idx], x[idx + 1:]).squeeze()
        p_dist_z = dist_func_z(z[idx], z[idx + 1:]).squeeze()
        loss += ((p_dist_x - p_dist_z) ** 2).sum()
    return loss / (batch_size - 1)


# def pairwise_distances(x, y=None):
#     '''
#     Input: x is a Nxd matrix
#     y is an optional Mxd matirx
#     Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
#     if y is not given then use 'y=x'.
#     i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
#     '''
#     # Normalize for each dimension
#     x_norm = (x**2).sum(1).view(-1, 1)
#     # x_norm = (x**2).sum(1).view(-1, 1) / ratio
#     if y is not None:
#         y_t = torch.transpose(y, 0, 1)
#         y_norm = (y**2).sum(1).view(1, -1)
#     else:
#         y_t = torch.transpose(x, 0, 1)
#         y_norm = x_norm.view(1, -1)
#         dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
#         # Ensure diagonal is zero if x=y
#     if y is None:
#         dist = dist - torch.diag(dist.diag())

#     dist = torch.clamp(dist, 0.0, np.inf)

#     dist = nn.functional.normalize(dist, p=1, dim=1)
#     return dist/dist.max()

def mask_out(tensor, threshold):
    """
    Zeros out elements in a tensor that are larger than a given threshold, using a mask.

    Args:
      tensor: The input tensor.
      threshold: The threshold value.
      mask: A tensor of the same shape as 'tensor' containing 0s and 1s.

    Returns:
      A new tensor with values larger than the threshold zeroed out.
    """

    # Create a boolean mask where True indicates values larger than the threshold
    larger_than_threshold_mask = tensor > threshold

    # Combine the masks using element-wise multiplication
    mask = (~larger_than_threshold_mask)

    # Apply the combined mask to the original tensor
    result = tensor * mask

    return result


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
            i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    # Compute the squared norms of each row in x
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        # Compute the squared norms of each row in y
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        # When y is None, use x for y
        y = x
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    # Compute the pairwise distance matrix
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    # Ensure diagonal is zero if x=y (i.e., self-distances are zero)
    if y is x:
        dist = dist - torch.diag(dist.diag())
        # print(torch.diag(dist))
        # print(torch.diag(dist.diag()))

    # Clamp to ensure no negative distances (due to floating point errors)
    dist = torch.clamp(dist, 0.0, np.inf)

    # dist = dist / dist.sum()
    # dist = nn.functional.normalize(dist, p=1)
    return dist / dist.max()


def cos_distances(x):
    x_norm = x / torch.norm(x).clamp(min=0.0)
    sim_matrix = 1 - torch.mm(x_norm, x_norm.transpose(0, 1))
    return sim_matrix / sim_matrix.max()


def calculate_gp_loss(X_list, Z_list, threshold=0.6, is_weight=False):
    loss = 0
    for i, (X, Z) in enumerate(zip(X_list, Z_list)):
        # ground_truth = mask_out(cos_distances(X), threshold=threshold)
        # ground_truth = mask_out(pairwise_distances(X), threshold=threshold)
        ground_truth = pairwise_distances(X)
        if is_weight:
            weight = (1 - torch.sin(0.5 * torch.tensor(math.pi) * ground_truth))
            # weight = (torch.sin(0.5 * torch.tensor(math.pi) * ground_truth))
        else:
            weight = 1
        loss += torch.mean(weight * torch.abs(ground_truth - pairwise_distances(Z)))
    return loss


def pca(X, k):
    """
    Perform PCA on the input data X and return the top k principal components.

    Args:
    X (torch.Tensor): Input data tensor with shape (n_samples, n_features).
    k (int): Number of principal components to return.

    Returns:
    torch.Tensor: Transformed data with shape (n_samples, k).
    """
    # Ensure input is a tensor
    X = torch.tensor(X, dtype=torch.float32)

    # Step 1: Center the data
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Step 2: Compute the covariance matrix
    covariance_matrix = torch.mm(X_centered.t(), X_centered) / (X_centered.shape[0] - 1)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvectors by eigenvalues in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_k_eigenvectors = eigenvectors[:, sorted_indices[:k]]

    # Step 5: Transform the data
    X_reduced = torch.mm(X_centered, top_k_eigenvectors)
    # X_reduced = X[:, sorted_indices[:k]]

    return X_reduced


def plt_scatter_pca_alignment(X, k=2):
    x1_reduced, x2_reduced = pca(X, k).chunk(2)
    x1_reduced, x2_reduced = x1_reduced.cpu().numpy(), x2_reduced.cpu().numpy()
    plt.scatter(x1_reduced[:, 0], x1_reduced[:, 1], c='r', marker='x')
    plt.scatter(x2_reduced[:, 0], x2_reduced[:, 1], c='b', marker='+')
    plt.show()


def extract_top_k_features(data, k, is_pca=False, is_both=False):
    """
    Perform PCA on the input data and extract the most important features based on the top k principal components.

    Args:
        data (torch.Tensor): Input data matrix of shape (n_samples, n_features)
        k (int): Number of top principal components to consider

    Returns:
        torch.Tensor: The data reduced to the most important features
        torch.Tensor: The indices of the most important features
    """
    # Center the data by subtracting the mean of each feature
    data_mean = torch.mean(data, dim=0)
    data_centered = data - data_mean

    # Perform SVD on the centered data
    U, S, V = torch.svd(data_centered)

    # The top k principal components are the first k columns of V
    top_k_components = V[:, :k]

    # Compute the importance of each feature by the magnitude of the loadings
    feature_top_k = torch.sum(top_k_components ** 2, dim=1)

    # Get the indices of the most important features
    top_k_features_indices = torch.argsort(feature_top_k, descending=True)

    # Select the data indexed by the important features
    top_k_data = data[:, top_k_features_indices[:k]]
    top_k_pca = torch.matmul(data_centered, top_k_components)

    if is_pca:
        return top_k_pca
    elif is_both:
        return top_k_data, top_k_pca
    else:
        return top_k_data


def plt_scatter_alignment(X, k=2, is_pca=False, is_both=False):
    if is_both:
        x_reduced, x_reduced_pca = extract_top_k_features(X, k, is_both=True)
        x1_reduced, x2_reduced = x_reduced.chunk(2)
        x1_reduced_pca, x2_reduced_pca = x_reduced_pca.chunk(2)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].set_title('Top K Features')
        axs[1].set_title('Top K PCA')
    elif is_pca:
        x1_reduced, x2_reduced = extract_top_k_features(X, k, is_pca=True).chunk(2)
        plt.set_title('Top K PCA')
    else:
        x1_reduced, x2_reduced = extract_top_k_features(X, k).chunk(2)
        plt.set_title('Top K Features')

    if is_both:
        x1_reduced, x2_reduced = x1_reduced.detach().cpu().numpy(), x2_reduced.detach().cpu().numpy()
        x1_reduced_pca, x2_reduced_pca = x1_reduced_pca.detach().cpu().numpy(), x2_reduced_pca.detach().cpu().numpy()
        axs[0].scatter(x1_reduced[:, 0], x1_reduced[:, 1], c='r', marker='x')
        axs[0].scatter(x2_reduced[:, 0], x2_reduced[:, 1], c='b', marker='+')
        axs[1].scatter(x1_reduced_pca[:, 0], x1_reduced_pca[:, 1], c='r', marker='x')
        axs[1].scatter(x2_reduced_pca[:, 0], x2_reduced_pca[:, 1], c='b', marker='+')
    else:
        x1_reduced, x2_reduced = x1_reduced.detach().cpu().numpy(), x2_reduced.detach().cpu().numpy()
        plt.scatter(x1_reduced[:, 0], x1_reduced[:, 1], c='r', marker='x')
        plt.scatter(x2_reduced[:, 0], x2_reduced[:, 1], c='b', marker='+')

    plt.show()


# import makegrid package
from torchvision.utils import make_grid
# Function to display reconstructed images
def display_reconstructed_images(epoch, vae_model, data, n_samples=10, dim=[1, 28, 28], is_flip=False):
    vae_model.eval()
    with torch.no_grad():
        data = data[:n_samples]
        recon_x, z, _, _ = vae_model(data)
        recon_x = recon_x[:n_samples]
        comparison = torch.cat([data.view(-1, dim[0], dim[1], dim[2]), recon_x.view(-1, dim[0], dim[1], dim[2])])
        comparison = make_grid(comparison, nrow=data.size(0))
        comparison = comparison.cpu().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(15, 5))
        plt.imshow(comparison, cmap='gray')
        plt.axis('off')
        plt.title(f'Reconstructed Images at Epoch {epoch}')
        plt.show()


def display_reconstructed_and_flip_images(epoch, vae_model, flip_vae_model, data, n_samples=10, dim=[1, 28, 28],
                                          flip_dim=[3, 32, 32], is_mnist=True, is_both=True):
    vae_model.eval()
    with torch.no_grad():
        data = data[:n_samples]
        recon_x, z, _, _ = vae_model(data)
        recon_x_flip = flip_vae_model.decode(z)
        data = data[:n_samples]
        recon_x = recon_x[:n_samples]
        recon_x_flip = recon_x_flip[:n_samples]

        data = data.view(n_samples, dim[0], dim[1], dim[2])
        recon_x = recon_x.view(n_samples, dim[0], dim[1], dim[2])
        recon_x_flip = recon_x_flip.view(n_samples, flip_dim[0], flip_dim[1], flip_dim[2])
        z = z[:n_samples]
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3 / 2, 4.5))
        if is_mnist:
            main_color = 'gray'
            flip_color = None
        elif is_both:
            main_color = 'gray'
            flip_color = 'gray'
        else:
            flip_color = 'gray'
            main_color = None

        for i in range(n_samples):
            axes[0, i].imshow(np.transpose(data[i].detach().cpu().numpy(), (1, 2, 0)), cmap=main_color)
            axes[0, i].axis('off')

            axes[1, i].imshow(np.transpose(recon_x[i].detach().cpu().numpy(), (1, 2, 0)), cmap=main_color)
            axes[1, i].axis('off')

            axes[2, i].imshow(np.transpose(recon_x_flip[i].detach().cpu().numpy(), (1, 2, 0)), cmap=flip_color)
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10, input_height=8, input_width=8):
        super(CNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the feature map after the convolution and pooling layers
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        def pool2d_size_out(size, kernel_size=2, stride=2, padding=0):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        conv1_out_height = conv2d_size_out(input_height)
        conv1_out_width = conv2d_size_out(input_width)

        pool1_out_height = pool2d_size_out(conv1_out_height)
        pool1_out_width = pool2d_size_out(conv1_out_width)

        conv2_out_height = conv2d_size_out(pool1_out_height)
        conv2_out_width = conv2d_size_out(pool1_out_width)

        pool2_out_height = pool2d_size_out(conv2_out_height)
        pool2_out_width = pool2d_size_out(conv2_out_width)

        conv3_out_height = conv2d_size_out(pool2_out_height)
        conv3_out_width = conv2d_size_out(pool2_out_width)

        pool3_out_height = pool2d_size_out(conv3_out_height)
        pool3_out_width = pool2d_size_out(conv3_out_width)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * pool3_out_height * pool3_out_width, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Download and load the training dataset
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test dataset
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Define a simple ResNet-like architecture for the classifier
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # Convolutional layer (sees 1x16x16 image tensor)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 2 * 2, 256)  # assuming the input is (1, 16, 16)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 2 * 2)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleLinearClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_classes=10):
        super(SimpleLinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10, hidden_dims=None, dropout_rate=0.5, activation='relu'):
        """
        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            hidden_dims (list of int, optional): List of hidden layer dimensions. Default is [512, 256, 128].
            dropout_rate (float, optional): Dropout rate. Default is 0.5.
            activation (str, optional): Activation function to use ('relu', 'leaky_relu', 'sigmoid', 'tanh'). Default is 'relu'.
        """
        super(CustomClassifier, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.layers = nn.ModuleList()
        current_dim = input_dim

        # Create hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            if activation == 'relu':
                self.layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                self.layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def compute_accuracy(dataloader, vae, model, device, is_compute_cp_loss=True):
    """
    Compute the accuracy of a classifier model on a given DataLoader.

    Args:
    dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
    model (torch.nn.Module): Trained model to evaluate.
    device (torch.device): Device to perform computation on ('cpu' or 'cuda').

    Returns:
    float: The accuracy of the model on the provided dataset.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    total_cp_loss = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data in dataloader:
            inputs, labels = data[0], data[-1]
            inputs, labels = inputs.to(device), labels.to(device)

            _, latent_inputs, _, _ = vae(inputs)
            # outputs = model(latent_inputs.view((latent_inputs.shape[0], 1, 16, 16)))

            # _, latent_inputs, _, _ = vae(inputs.view((inputs.shape[0], -1)))
            # outputs = model(latent_inputs.view((latent_inputs.shape[0], 1, 8, 8)))
            outputs = model(latent_inputs.view(latent_inputs.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
            total_cp_loss += F.cross_entropy(outputs, labels, reduction='sum')

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if is_compute_cp_loss:
        print(f'cp loss: {total_cp_loss}', end='|   ')
    accuracy = correct / total
    return accuracy

def conditional_errors(preds, labels, attrs):
    """
    Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 0, error | A = 1.
    """
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean( preds == labels)
    idx = attrs == 0
    error_0 = 1 - np.mean(preds[idx] == labels[idx])
    error_1 = 1 - np.mean(preds[~idx] == labels[~idx])
    return cls_error, error_0, error_1

import pandas as pd
from notebooks.dataset_fair import AdultDataset, COMPAS

def get_dataset_compas(root):
    compas = pd.read_csv(f"{root}/propublica.csv").values
    num_insts = compas.shape[0]
    # Random shuffle the dataset.
    indices = np.arange(num_insts)
    np.random.shuffle(indices)
    compas = compas[indices]
    # Partition the dataset into train and test split.
    ratio = 0.7
    num_train = int(num_insts * ratio)
    compas_train = COMPAS(compas[:num_train, :])
    compas_test = COMPAS(compas[num_train:, :])
    return compas_train, compas_test


def get_loaders_compas(compas_train, compas_test, batch_size):
    train_loader = DataLoader(compas_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(compas_test, batch_size=batch_size, shuffle=False)
    return train_loader ,test_loader


def get_dataset_adult(root, target="income", private="sex"):
    adult_train = AdultDataset(root_dir=root, phase='train', tar_attr=target, priv_attr=private)
    adult_test = AdultDataset(root_dir=root, phase='test', tar_attr=target, priv_attr=private)
    return adult_train, adult_test

def get_loaders_adult(adult_train, adult_val, adult_test, batch_size):
    train_loader = DataLoader(adult_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(adult_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(adult_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def visualize_fair_losses(total_loss_list, gp_loss_list, vaub_loss_list, cls_loss_list,
                          recon_loss_list, kld_encoder_posterior_list, score_list,
                          dsm_loss_list, z1_var_list, z2_var_list,
                          z1_mean_list, z2_mean_list):
    # Losses plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Visualization of Losses', fontsize=16)

    axs[0, 0].plot(total_loss_list, label='Total Loss', color='tab:blue')
    axs[0, 0].set_title('Total Loss', fontsize=12)
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss Value')
    axs[0, 0].legend()

    axs[1, 0].plot(gp_loss_list, label='GP Loss', color='tab:green')
    axs[1, 0].set_title('GP Loss', fontsize=12)
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss Value')
    axs[1, 0].legend()

    axs[0, 1].plot(vaub_loss_list, label='VAUB Loss', color='tab:red')
    axs[0, 1].set_title('VAUB Loss', fontsize=12)
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss Value')
    axs[0, 1].legend()

    axs[1, 1].plot(cls_loss_list, label='Classification Loss', color='tab:orange')
    axs[1, 1].set_title('Classification Loss', fontsize=12)
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss Value')
    axs[1, 1].legend()

    # Apply tight layout for clear spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # VAUB Losses plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Detailed Visualization of VAUB Losses', fontsize=16)

    axs[0].plot(recon_loss_list, label='Reconstruction Loss', color='tab:purple')
    axs[0].set_title('Reconstruction Loss', fontsize=12)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()

    axs[1].plot(score_list, label='Score', color='tab:brown')
    axs[1].set_title('Score Value', fontsize=12)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Score')
    axs[1].legend()

    axs[2].plot(kld_encoder_posterior_list, label='KLD Encoder Posterior', color='tab:pink')
    axs[2].set_title('KLD Encoder Posterior', fontsize=12)
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('KLD Value')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Accuracy and DSM plot
    fig, axs = plt.subplots(1, 1, figsize=(15, 4))
    fig.suptitle('Detailed Visualization of DSM Loss and Accuracy', fontsize=16)

    axs.plot(dsm_loss_list, label='DSM Loss', color='tab:cyan')
    axs.set_title('DSM Loss', fontsize=12)
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss Value')
    axs.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Latent Space Mean and Variance plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Visualization of Latent Space Mean and Variance', fontsize=16)

    axs[0, 0].plot(z1_var_list, label='Z1 Variance', color='tab:red')
    axs[0, 0].set_title('Z1 Variance', fontsize=12)
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Variance')
    axs[0, 0].legend()

    axs[0, 1].plot(z2_var_list, label='Z2 Variance', color='tab:orange')
    axs[0, 1].set_title('Z2 Variance', fontsize=12)
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Variance')
    axs[0, 1].legend()

    axs[1, 0].plot(z1_mean_list, label='Z1 Mean', color='tab:green')
    axs[1, 0].set_title('Z1 Mean', fontsize=12)
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Mean')
    axs[1, 0].legend()

    axs[1, 1].plot(z2_mean_list, label='Z2 Mean', color='tab:blue')
    axs[1, 1].set_title('Z2 Mean', fontsize=12)
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Mean')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def display_umap_for_latent_multi(axes_total, epoch, z_list, label_list):

    # Combine the datasets
    data = np.vstack([z.cpu() for z in z_list])
    labels = np.concatenate([label.cpu() for label in label_list])
    domains = np.concatenate([[f'domain{i}']*len(z.cpu()) for i, z in enumerate(z_list)])

    # Fit and transform the data using UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    # Create a DataFrame for easier handling
    df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df['label'] = labels
    df['domain'] = domains

    # Define markers and colors
    color_list = list(mcolors.BASE_COLORS)
    colors = {f'domain{i}': color_list[i] for i in range(len(z_list))}

    # Plot the data
    for domain in colors:
        subset = df[(df['domain'] == domain)]
        axes_total[1].scatter(subset['UMAP1'], subset['UMAP2'], c=[colors[domain]], marker="s",
                        alpha=0.6,
                        edgecolors='w', linewidth=0.5, label=f'{domain}')

    # Create a combined legend
    axes_total[1].set_title(f'UMAP Visualization of Latent Space at Epoch {epoch} in domain color')

    # Define markers and colors
    markers = {'domain1': 'X', 'domain2': 's'}
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Plot the data
    for domain in markers:
        for label in unique_labels:
            subset = df[(df['domain'] == domain) & (df['label'] == label)]
            axes_total[0].scatter(subset['UMAP1'], subset['UMAP2'], c=[color_map[label]], marker=markers[domain], alpha=0.6,
                       edgecolors='w', linewidth=0.5, label=f'{domain}-{label}')


    axes_total[0].set_title(f'UMAP Visualization of Latent Space at Epoch {epoch}')


    return axes_total