import torch
import torch.nn as nn
import torch.nn.functional as F


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


#### Old Encoder version without BatchNorm
# # Define the Encoder with Skip Connections and Residual Blocks
# class ResConvEncoder(nn.Module):
#     def __init__(self, latent_size=64, type_of_dataset='mnist'):
#         super(ResConvEncoder, self).__init__()
#         self.latent_size = latent_size
#         if type_of_dataset=='svhn':
#             self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)
#         else:
#             self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # (batch_size, 32, 14, 14)
#         self.res1 = ResidualBlockVAE(16, 16)
#         self.conv2 = nn.Conv2d(16, 64, 4, 2, 1)  # (batch_size, 64, 7, 7)
#         self.res2 = ResidualBlockVAE(64, 64)
#         if type_of_dataset=='svhn':
#             self.conv3 = nn.Conv2d(64, 2*latent_size, 4, 2, 1)
#         elif type_of_dataset=='mnist':
#             self.conv3 = nn.Conv2d(64, 2*latent_size, 3, 2, 1)  # (batch_size, 128, 4, 4)
#         elif type_of_dataset=='usps':
#             self.conv3 = nn.Conv2d(64, 2*latent_size, 3, 1, 1)  # (batch_size, 128, 4, 4)
#         self.res3 = ResidualBlockVAE(2*latent_size, 2*latent_size)
#         self.conv4 = nn.Conv2d(2*latent_size, 2*latent_size, 4)  # (batch_size, 128, 1, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.res1(x)
#         x = F.relu(self.conv2(x))
#         x = self.res2(x)
#         x = F.relu(self.conv3(x))
#         x = self.res3(x)

#         mu, logvar = self.conv4(x).view(x.shape[0], -1).chunk(2, dim=-1)  # (batch_size, latent_size)

#         mu = mu.view((-1, self.latent_size, 1, 1))
#         logvar = logvar.view((-1, self.latent_size, 1, 1))

#         return mu, logvar

####################### Changed #############################
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
        if is_2d and not is_3d:
            self.lin = nn.Linear(128, 4)
            self.bn1D = nn.BatchNorm1d(2, affine=False)
        elif not is_2d and is_3d:
            self.lin = nn.Linear(128, 6)
            self.bn2 = nn.BatchNorm1d(3, affine=False)
        if type_of_dataset == 'svhn':
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 4, 2, 1)
        elif type_of_dataset == 'mnist':
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 3, 2, 1)  # (batch_size, 128, 4, 4)
        elif type_of_dataset == 'usps':
            self.conv3 = nn.Conv2d(64, 2 * latent_size, 3, 1, 1)  # (batch_size, 128, 4, 4)
        self.res3 = ResidualBlockVAE(2 * latent_size, 2 * latent_size)
        self.conv4 = nn.Conv2d(2 * latent_size, 2 * latent_size, 4)  # (batch_size, 128, 1, 1)
        self.bn_mu = nn.BatchNorm1d(latent_size, affine=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)

        x = self.conv4(x).view(x.shape[0], -1)  # (batch_size, latent_size)

        if self.is_2d or self.is_3d:
            x = self.lin(F.relu(x).view(x.shape[0], -1))
            mu, logvar = x.chunk(2, dim=-1)
            mu = self.bn2(mu)
        else:
            mu, logvar = x.chunk(2, dim=-1)
            mu = self.bn_mu(mu.view(mu.shape[0], -1))
            logvar = logvar.view(logvar.shape[0], -1)

        if not self.is_2d and not self.is_3d:
            mu = mu.view((-1, self.latent_size, 1, 1))
            logvar = logvar.view((-1, self.latent_size, 1, 1))

        return mu, logvar


############################################################

#### Old version without 2D and 3D latent representations
# # Define the Decoder with Skip Connections and Residual Blocks
# class ResConvDecoder(nn.Module):
#     def __init__(self, latent_size=64, type_of_dataset='mnist'):
#         super(ResConvDecoder, self).__init__()
#         self.latent_size = latent_size
#         self.res1 = ResidualBlockVAE(latent_size, latent_size)
#         self.conv1 = nn.ConvTranspose2d(latent_size, 64, 4)  # (batch_size, 64, 4, 4)
#         self.res2 = ResidualBlockVAE(64, 64)
#         self.conv2 = nn.ConvTranspose2d(64, 16, 4, 2, 1)  # (batch_size, 32, 8, 8)
#         self.res3 = ResidualBlockVAE(16, 16)
#         if type_of_dataset=='svhn':
#             self.conv3 = nn.ConvTranspose2d(16, 3, 4, 4, 0)
#         elif type_of_dataset=='mnist':
#             self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 2)
#         elif type_of_dataset=='usps':
#             self.conv3 = nn.ConvTranspose2d(16, 1, 4, 2, 1)

#     def forward(self, x):
#         x = self.res1(x)
#         x = F.relu(self.conv1(x))
#         x = self.res2(x)
#         x = F.relu(self.conv2(x))
#         x = self.res3(x)
#         x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for binary image
#         return x

####################### Changed ####################################

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
        if is_2d and not is_3d:
            self.lin = nn.Linear(2, 64)
        elif not is_2d and is_3d:
            self.lin = nn.Linear(3, 64)
        if type_of_dataset == 'svhn':
            self.conv3 = nn.ConvTranspose2d(16, 1, 4, 4, 0)
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
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for binary image

        return x


####################################################################

#### Old version without 2D or 3D latent representations and without soft poram start
# # Define the VAE with Skip Connections and Residual Blocks
# class CNN_VAE(nn.Module):
#     def __init__(self, latent_height=8, latent_width=8, type_of_dataset='mnist'):
#         super().__init__()
#         self.latent_height = latent_height
#         self.latent_width = latent_width

#         self.encoder = ResConvEncoder(latent_size=latent_height*latent_width, type_of_dataset=type_of_dataset)
#         self.decoder = ResConvDecoder(latent_size=latent_height*latent_width, type_of_dataset=type_of_dataset)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         return self.decoder(z.view((z.shape[0], -1, 1, 1)))

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), z.view((z.shape[0], 1, self.latent_height, self.latent_width)), mu.view((z.shape[0], -1)), logvar.view((z.shape[0], -1))

################# Changed ####################
# Define the VAE with Skip Connections and Residual Blocks
class CNN_VAE(nn.Module):
    def __init__(self, latent_height=8, latent_width=8, type_of_dataset='mnist', is_2d=False, is_3d=False):
        super().__init__()
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


##################################################


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