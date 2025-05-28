import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Encoder (Feature Extractor)
class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.fc1 = nn.Linear(48 * 4 * 4, latent_dim) # Mean of the latent space
        self.fc2 = nn.Linear(48 * 4 * 4, latent_dim) # Log variance of the latent space

        # self.fc2 = nn.Linear(256, 20)  # Mean of the latent space
        # self.fc3 = nn.Linear(256, 20)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        return self.fc1(x), self.fc2(x)

class EncoderPnP(nn.Module):
    def __init__(self, latent_dim=20):
        super(EncoderPnP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        # self.fc1 = nn.Linear(48 * 4 * 4, latent_dim) # Mean of the latent space
        self.log_var = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=True)

        # self.fc2 = nn.Linear(256, 20)  # Mean of the latent space
        # self.fc3 = nn.Linear(256, 20)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        # print(x.shape, self.log_var.shape)
        # x = F.relu(self.fc1(x))
        log_var_batch = self.log_var.expand(x.size(0), -1)  # Expand log_var to match batch size
        return x, log_var_batch


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 48 * 4 * 4),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1),  # 4x4 -> 7x7
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Sigmoid()  # Output activation function for the reconstruction
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 48, 4, 4)
        x = self.deconv(x)
        return x


# Define the VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=20, is_pnp=False):
        super(VAE, self).__init__()
        self.is_pnp = is_pnp
        if is_pnp:
            self.encoder = EncoderPnP(latent_dim)
        else:
            self.encoder = Encoder(latent_dim)
        # self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

# # Loss function
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD
#
# # Data loader
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
#
# train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
# test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
#
# # Training the VAE
# vae = VAE().to(device)
# optimizer = optim.Adam(vae.parameters(), lr=1e-3)
#
# def train(epoch):
#     vae.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = vae(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
#     print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
#
# def test(epoch):
#     vae.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = vae(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n], recon_batch.view(64, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(), f'results/reconstruction_{epoch}.png', nrow=n)
#
#     test_loss /= len(test_loader.dataset)
#     print(f'====> Test set loss: {test_loss:.4f}')
#
# for epoch in range(1, 11):
#     train(epoch)
#     test(epoch)
