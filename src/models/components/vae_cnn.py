import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a Residual Block
class ResidualBlockVAEold(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockVAEold, self).__init__()
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


class ResidualBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockVAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


# Define the Encoder with Skip Connections and Residual Blocks
class Encoder(nn.Module):
    def __init__(self, latent_size=64):
        super().__init__()

        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # (batch_size, 32, 16, 16)
        self.res1 = ResidualBlockVAE(16, 16)
        self.conv2 = nn.Conv2d(16, 64, 4, 2, 1)  # (batch_size, 64, 8, 8)
        self.res2 = ResidualBlockVAE(64, 64)
        self.conv3 = nn.Conv2d(64, 2*latent_size, 4, 2, 1)  # (batch_size, 128, 4, 4)
        self.res3 = ResidualBlockVAE(2*latent_size, 2*latent_size)
        self.conv4 = nn.Conv2d(2*latent_size, 2*latent_size, 4)  # (batch_size, 128, 1, 1)

    def forward(self, x):
        # print(f"layer 0 has range [{x.max():.2f},{x.min():.2f}]")
        x = F.relu(self.conv1(x))
        # print(f"layer 1 has range [{x.max():.2f},{x.min():.2f}]")
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)

        mu, logvar = self.conv4(x).view(x.shape[0], -1).chunk(2, dim=-1)  # (batch_size, latent_size)

        mu = mu.view((-1, self.latent_size, 1, 1))
        logvar = logvar.view((-1, self.latent_size, 1, 1))

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_size=64, is_svhn=False):
        super().__init__()

        self.latent_size = latent_size
        self.res1 = ResidualBlockVAE(latent_size, latent_size)
        self.conv1 = nn.ConvTranspose2d(latent_size, 64, 4)  # (batch_size, 64, 4, 4)
        self.res2 = ResidualBlockVAE(64, 64)
        self.conv2 = nn.ConvTranspose2d(64, 16, 5, 2, 1)  # (batch_size, 32, 8, 8)
        self.res3 = ResidualBlockVAE(16, 16)
        self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 2)

    def forward(self, x):
        x = self.res1(x)
        x = F.relu(self.conv1(x))
        x = self.res2(x)
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for binary image
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CNN_VAE(nn.Module):
    def __init__(self, latent_height=8, latent_width=8):
        super().__init__()
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.encoder = Encoder(latent_size=latent_height*latent_width)
        self.decoder = Decoder(latent_size=latent_height*latent_width)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z.view((z.shape[0], -1, 1, 1)))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # print(f"mu has max {mu.max()}")
        # print(f"logvar has max {logvar.max()}")
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z.view((z.shape[0], 1, self.latent_height, self.latent_width)), mu.view(
            (z.shape[0], -1)), logvar.view((z.shape[0], -1))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
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
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), z, mean, logvar

