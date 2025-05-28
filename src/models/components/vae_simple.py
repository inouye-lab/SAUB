import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, latent_dim, is_batchnorm=False):
        super(Encoder, self).__init__()
        self.is_batchnorm = is_batchnorm

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.bn4 = nn.BatchNorm1d(latent_dim, affine=False)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        if self.is_batchnorm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.bn3(self.fc1(x)))
            mu = self.bn4(self.fc_mu(x))
            logvar = self.fc_logvar(x)
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64*8*8)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 8, 8)  # Reshape
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # Sigmoid for MNIST (binary image)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, is_batchnorm=False, is_pnp=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, is_batchnorm=is_batchnorm)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, z, mu, logvar

    def decode(self, z):
        return self.decoder(z)


    def init_weights_fixed(self, seed=42):
        """
        Initialize all the weights of the model to the same small random values.
        """
        torch.manual_seed(seed)  # Set a fixed seed for reproducibility

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.uniform_(m.weight, a=-0.2, b=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(weights_init)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')


if __name__ == '__main__':
    batch_size = 128
    latent_dim = 20
    epochs = 10
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)