import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim, elementwise_affine=True)
        self.swish = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim, elementwise_affine=True)

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
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, multiplier=4, is_warm_init=False):
        super(UNet, self).__init__()
        self.num_timesteps = num_timesteps

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
            nn.Linear(multiplier*in_dim + embedding_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            nn.Linear(multiplier*in_dim, out_dim)
        )

        # Define time step embedding layer for decoder
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)

        if is_warm_init:
            self.warm_init()

    def forward(self, x, timestep, enc_sigma=None):
        # Encoder
        if enc_sigma is not None:
            encoded_enc_sigma = self.encoder(enc_sigma)
        else:
            encoded_enc_sigma = 0
        x = self.encoder(x) + encoded_enc_sigma

        # Decoder
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


class UNet_noise(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, num_latent_noise_scale=50, is_add_latent_noise=False, multiplier=4, is_warm_init=False):
        super(UNet_noise, self).__init__()
        self.num_timesteps = num_timesteps
        self.is_add_latent_noise = is_add_latent_noise

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(multiplier*in_dim + embedding_dim, multiplier*in_dim) if not is_add_latent_noise else nn.Linear(multiplier*in_dim + embedding_dim + embedding_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            # nn.Dropout(0.1),
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