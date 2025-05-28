import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_swiss_roll
import sklearn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import pytorch_lightning as pl
from itertools import chain

pl.seed_everything(1)


# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.bn_mean = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)
        mean = self.bn_mean(mean)
        logvar = torch.clamp(logvar, max=4)
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


# Loss function
def vae_loss(recon_x, x, mean, logvar, beta=0.01, score=None, DSM=None):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
    kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
    kld_loss = kld_encoder_posterior + kld_prior
    if score is not None and DSM is None:
        kld_loss = kld_encoder_posterior - score
    elif DSM is not None:
        kld_loss = kld_encoder_posterior + DSM
    return recon_loss + beta * kld_loss, recon_loss, kld_encoder_posterior, kld_prior

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), is_warm_init=False):
        super(UNet, self).__init__()
        self.num_timesteps = num_timesteps
        self.device = device

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(64 + embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
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
        for param in self.parameters():
            nn.init.uniform_(param, -0.002, 0.002)


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


class Score_fn(nn.Module):
    def __init__(self, model, ema=None, ema_decay=0.99, sigma_min=0.01, sigma_max=50, num_timesteps=1000,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
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
        if ema is not None:
            self.ema = ema(model, decay=ema_decay)

        # Learnable parameter for residual score function and assures value between [0,1]
        self.lbda = nn.ParameterList([nn.Parameter(torch.tensor([0.0]))])

    def to_device(self):
        self.model = self.model.to(self.device)

    # Compute denoising score matching loss
    def compute_DSM_loss(self, x, t, enc_mu=None, enc_sigma=None, alpha=None, turn_off_enc_sigma=False,
                         learn_lbda=False, is_mixing=False, is_residual=False, is_vanilla=False, is_LSGM=False,
                         divide_by_sigma=False):
        sigmas = self.discrete_sigma[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x, device=self.device) * sigmas
        perturbed_data = x + noise
        if is_mixing:
            score = self.get_mixing_score_fn(perturbed_data, t, alpha=alpha, is_residual=is_residual,
                                             is_vanilla=is_vanilla, divide_by_sigma=divide_by_sigma)
        elif is_residual:
            enc_eps = x - enc_mu
            score = self.get_residual_score_fn(perturbed_data, t, enc_eps, enc_sigma, turn_off_enc_sigma, learn_lbda,
                                               is_vanilla=is_vanilla, divide_by_sigma=divide_by_sigma)
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
    def get_score_fn(self, x, t, detach=False):
        if detach:
            self.model.eval()
            return (self.model(x, t) / self.discrete_sigma[t.long()].view(x.shape[0],
                                                                          *([1] * len(x.shape[1:])))).detach()
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
    def get_mixing_score_fn(self, x, t, alpha=None, is_residual=False, is_vanilla=False, detach=False,
                            divide_by_sigma=False):

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
            learnable_score = alpha * self.model(x, t) / self.discrete_sigma[t.long()].view(x.shape[0],
                                                                                            *([1] * len(x.shape[1:])))
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

    def get_LSGM_loss(self, x, t=None, is_mixing=False, is_residual=False, is_vanilla=False, alpha=None):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device)

        loss = self.compute_DSM_loss(x, t, is_mixing=is_mixing, is_residual=is_residual, alpha=alpha,
                                     is_vanilla=is_vanilla, is_LSGM=True, divide_by_sigma=True)
        return loss

    # Update one batch and add shrink the max timestep for reducing the variance range of training (default is equal to defined num_timestep).
    # When verbose is true, gets the average loss up until last verbose and saves to loss dict
    def update_score_fn(self, x, optimizer, alpha=None, max_timestep=None, t=None, verbose=False, is_mixing=False,
                        is_residual=False, is_vanilla=False, divide_by_sigma=False):
        # TODO: Add ema optimization
        if max_timestep is None or max_timestep > self.num_timesteps:
            max_timestep = self.num_timesteps

        if t is None:
            t = torch.randint(0, max_timestep, (x.shape[0],), device=self.device)

        loss = self.compute_DSM_loss(x, t, is_mixing=is_mixing, is_residual=is_residual, alpha=alpha,
                                     is_vanilla=is_vanilla, divide_by_sigma=False)

        self.total_loss += loss.item()
        self.loss_counter += 1.
        if verbose:
            avg_loss = self.total_loss / self.loss_counter
            self.reset_loss_count()
            self.update_loss_dict(avg_loss)
            print(avg_loss)
            print(f'alpha: {torch.sigmoid(self.lbda[0])}')

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Update EMA
        if hasattr(self, 'ema'):
            self.ema.update()

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
    def draw_gradient_field(self, ax, xlim, ylim, t=0, x_num=20, y_num=20, file="./Score_Function", noise_label=1,
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
                data2_l1, data2_l2 = data2[labels2 == 0], data2[labels2 == 1]
                ax.scatter(data1_l1[:n_samples, 0], data1_l1[:n_samples, 1], marker='+', label='D1_L1', c='b', s=20)
                ax.scatter(data1_l2[:n_samples, 0], data1_l2[:n_samples, 1], marker='o', label='D1_L2', c='b', s=20)
                ax.scatter(data2_l1[:n_samples, 0], data2_l1[:n_samples, 1], marker='+', label='D2_L1', c='g', s=20)
                ax.scatter(data2_l2[:n_samples, 0], data2_l2[:n_samples, 1], marker='o', label='D2_L2', c='g', s=20)
                ax.legend()
            else:
                ax.scatter(data[:, 0], data[:, 1])

        if is_mixture:
            score_fn = self.get_mixing_score_fn(input, torch.ones((x_num * y_num,), device=self.device).type(torch.long) * t,
                                                detach=True, alpha=alpha, is_vanilla=is_vanilla)
        elif is_residual:
            score_fn = self.get_mixing_score_fn(input, torch.ones((x_num * y_num,), device=self.device).type(torch.long) * t,
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

def get_lp_dist(p=2):
    return nn.PairwiseDistance(p=p, keepdim=True)
# def move_metric_x_to_device(self, metric_x, device):
# metric_x.to(device)
def compute_gp_loss(x, z, dist_func_x, dist_func_z):
    batch_size = len(x)
    loss = 0
    for idx in range(batch_size-1):
        p_dist_x = dist_func_x(x[idx], x[idx+1:]).squeeze()
        p_dist_z = dist_func_z(z[idx], z[idx+1:]).squeeze()
        loss += ((p_dist_x-p_dist_z)**2).mean()
    return loss/(batch_size-1)

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
    y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''

    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag())
    return torch.clamp(dist, 0.0, np.inf)

def calculate_gp_loss(X_list, Z_list):
    loss = 0
    for X, Z in zip(X_list, Z_list):
        loss += torch.mean(torch.abs(pairwise_distances(X)-pairwise_distances(Z)))
    return loss


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# Define a custom Dataset class
class DShapeDataset(Dataset):
    def __init__(self, offset_inner=(1.5, 0), offset_outer=(0, 0), noise=0.25, flip=False, n_points=1000):
        def generate_d_shape(radius, height, n_points, offset=(0, 0), flip=False, noise=0.0):
            """
            Generate points for a "D" shape with optional flipping and noise.

            Args:
            - radius (float): Radius of the semicircle.
            - height (float): Height of the straight line part.
            - n_points (int): Number of points to generate.
            - offset (tuple): (x, y) offset to translate the shape.
            - flip (bool): Whether to flip the "D" shape horizontally.
            - noise (float): Standard deviation of Gaussian noise to add to the points.

            Returns:
            - points (np.array): Generated points of shape (n_points, 2).
            """
            # Generate points for the semicircle
            t = np.linspace(-np.pi / 2, np.pi / 2, n_points // 2)
            semicircle_x = radius * np.cos(t)
            semicircle_y = radius * np.sin(t)

            # Generate points for the straight line
            line_x = np.zeros(n_points // 2)
            line_y = np.linspace(-radius, radius, n_points // 2)

            # Combine the points
            x = np.concatenate([semicircle_x, line_x])
            y = np.concatenate([semicircle_y, line_y])

            # Apply flipping if needed
            if flip:
                x = -x + 2 * radius

            # Apply translation
            x += offset[0]
            y += offset[1]

            points = np.vstack((x, y)).T

            # Add Gaussian noise
            if noise > 0:
                points += np.random.normal(scale=noise, size=points.shape)

            return points

        # Parameters for the shapes
        radius_outer = 5
        height_outer = 10
        radius_inner = 2
        height_inner = 4

        # Generate the outer "D" shape
        outer_d = generate_d_shape(radius_outer, height_outer, n_points, offset_outer, flip=flip, noise=noise)

        # Center the inner "D" shape inside the outer "D" and scale it down
        inner_d = generate_d_shape(radius_inner, height_inner, n_points, offset_inner, flip=flip, noise=noise)

        # Combine the data
        data = np.vstack((outer_d, inner_d))

        labels_outer = np.zeros(n_points, dtype=np.longlong)
        labels_inner = np.ones(n_points, dtype=np.longlong)
        labels = np.hstack((labels_outer, labels_inner))

        # Convert to PyTorch tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloader(n_points, batch_size, plot=True):

    domain1 = DShapeDataset(n_points=n_points//2)
    domain2 = DShapeDataset(flip=True, n_points=n_points//2, offset_inner=(9.5, 4), offset_outer=(5, 4))

    # Create a DataLoader
    dataloader_domain1 = DataLoader(domain1, batch_size=batch_size, shuffle=True)
    dataloader_domain2 = DataLoader(domain2, batch_size=batch_size, shuffle=True)
    dataloader_score1 = DataLoader(domain1, batch_size=batch_size, shuffle=True)
    dataloader_score2 = DataLoader(domain2, batch_size=batch_size, shuffle=True)

    if plot:
        # Plot the combined D-shapes to visualize
        data1, label1 = next(iter(dataloader_domain1))
        data2, label2 = next(iter(dataloader_domain2))

        plt.figure(figsize=(10, 8))
        plt.scatter(data1[:, 0], data1[:, 1], c=label1, cmap='viridis', s=5)
        plt.scatter(data2[:, 0], data2[:, 1], c=label2, cmap='plasma', s=5)
        plt.title('Nested D- Labels')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.show()

    return dataloader_domain1, dataloader_domain2, dataloader_score1, dataloader_score2


def train_vaub_gp(mode, device, is_vanilla, input_dim, latent_dim, alpha, loops, hidden_dim, timesteps, sigma_max,
                  sigma_min, lr_vae, lr_score, beta, gp_lambda, num_epochs, dataloader_domain1, dataloader_domain2,
                  dataloader_score1, dataloader_score2, num_visual=1, num_log=1, plot=True):

    vae1 = VAE(input_dim, hidden_dim, latent_dim).to(device)
    vae2 = VAE(input_dim, hidden_dim, latent_dim).to(device)
    score_model = Score_fn(UNet(in_dim=2, out_dim=2, num_timesteps=timesteps, is_warm_init=False), sigma_min=sigma_min, sigma_max=sigma_max, num_timesteps=timesteps, device=device).to(device)
    optimizer_vae = optim.Adam(chain(vae1.parameters(), vae2.parameters()), lr=lr_vae)
    optimizer_score = torch.optim.Adam(score_model.parameters(), lr=lr_score)

    total_loss_list = []
    recon_loss_list = []
    kl_loss_list = []
    gp_loss_list = []
    # Training
    for epoch in tqdm(range(num_epochs)):
        vae1.train()
        vae2.train()
        total_loss = 0
        total_recon_loss = 0
        total_kld_encoder_posterior = 0
        total_kld_prior = 0

        for i, (data1, data2) in enumerate(zip(dataloader_domain1, dataloader_domain2)):
            x1, label1 = data1
            x2, label2 = data2
            x1, x2 = x1.to(device), x2.to(device)
            optimizer_vae.zero_grad()

            recon_x1, z1, mean1, logvar1 = vae1(x1)
            recon_x2, z2, mean2, logvar2 = vae2(x2)
            x, recon_x, z, mean, logvar = torch.vstack((x1, x2)), torch.vstack((recon_x1, recon_x2)), torch.vstack((z1, z2)), torch.vstack((mean1, mean2)), torch.vstack((logvar1, logvar2))

            # DSM = score_model.get_LSGM_loss(z, is_mixing=True, is_residual=True, is_vanilla=is_vanilla)
            score = score_model.get_mixing_score_fn(z, 5*torch.ones(z.shape[0], device=device).type(torch.long), detach=True, is_residual=True, is_vanilla=is_vanilla, alpha=alpha) - 0.05 * z
            score = torch.matmul(score.unsqueeze(1), z.unsqueeze(-1)).sum()
            # score = -torch.sqrt(torch.matmul(score.unsqueeze(1), z.unsqueeze(-1)).sum()**2)

            if mode == 'Gaussian':
                loss, recon_loss, kld_encoder_posterior, kld_prior = vae_loss(recon_x, x, mean, logvar, beta, score=None, DSM=None)
            else:
                loss, recon_loss, kld_encoder_posterior, kld_prior = vae_loss(recon_x, x, mean, logvar, beta, score=score, DSM=None)

            # dist_func_x = get_lp_dist(p=2)
            # dist_func_z = get_lp_dist(p=2)
            # gp_loss = sum([compute_gp_loss(x, z, dist_func_x, dist_func_z) for x, z in zip([x1, x2], [z1, z2])])
            gp_loss = gp_lambda * calculate_gp_loss([x1, x2], [z1, z2])

            gp_loss_list.append(gp_loss.item())

            loss += gp_loss
            total_loss_list.append((loss).item())
            recon_loss_list.append((recon_loss).item())
            kl_loss_list.append((kld_encoder_posterior+kld_prior).item())

            loss.backward()
            optimizer_vae.step()

            # if epoch % 25 == 0 and i==0:
            #     print(f'score loss: {score}')
            #     print(f'LSGM loss: {DSM}')
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_encoder_posterior += kld_encoder_posterior.item()
            total_kld_prior += kld_prior.item()

            # Update Score Function
            for loop in range(loops):
                data1, data2 = next(iter(zip(dataloader_score1, dataloader_score2)))
                x1, label1 = data1
                x2, label2 = data2
                x1, x2 = x1.to(device), x2.to(device)
                recon_x1, z1, mean1, logvar1 = vae1(x1)
                recon_x2, z2, mean2, logvar2 = vae2(x2)
                x, recon_x, z, mean, logvar, labels = torch.vstack((x1, x2)), torch.vstack((recon_x1, recon_x2)), torch.vstack((z1, z2)), torch.vstack((mean1, mean2)), torch.vstack((logvar1, logvar2)), torch.vstack((label1, label2))
                # print(loop)
                if loop == (loops-1) and (epoch+1) % (num_epochs//num_visual) == 0 and i==0:
                    print(f"Epoch {epoch} DSM average loss:", end=' ')
                    recon_x1_z2 = vae1.decode(z2)
                    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                    ax[0].scatter(recon_x1_z2.detach().cpu()[:, 0], recon_x1_z2.detach().cpu()[:, 1], marker='.')
                    ax[0].set_title('X1 Given Z2')
                    ax[1].scatter(recon_x1.detach().cpu()[:, 0], recon_x1.detach().cpu()[:, 1], marker='.')
                    ax[1].set_title('X1 Reconstruction')
                    score_model.update_score_fn(z, optimizer=optimizer_score, max_timestep=None, verbose=True, is_mixing=True, is_residual=True, is_vanilla=is_vanilla, alpha=alpha)
                    score_model.draw_gradient_field(ax[2], (-0, 0), (-0, 0), t=0, x_num=40, y_num=40, data=z.detach().cpu(), labels=labels, save=True, is_residual=True, is_vanilla=is_vanilla, alpha=alpha, is_mixture=True)
                else:
                    score_model.update_score_fn(z, optimizer=optimizer_score, max_timestep=None, is_mixing=True, is_residual=True, is_vanilla=is_vanilla, alpha=alpha)
        # Print every 25 epochs
        if (epoch + 1) % (num_epochs//num_log) == 0:
            print(f'Epoch {epoch+1}, Total Loss: {total_loss:.2f}, Recon Loss: {total_recon_loss:.2f}, '
                  f'Encoder Posterior Loss: {total_kld_encoder_posterior:.2f}, Prior Loss: {total_kld_prior:.2f}, '
                  f'Gp loss: {gp_loss}')

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))  # Adjust figsize as needed

        # Plot Data and Set Titles
        axs[0].plot(total_loss_list, label='total lost')
        axs[0].set_title('total lost')

        axs[1].plot(recon_loss_list, label='recon list')
        axs[1].set_title('recon list')

        axs[2].plot(kl_loss_list, label='kl list')
        axs[2].set_title('kl list')

        axs[3].plot(gp_loss_list, label='gp list')
        axs[3].set_title('gp list')

        plt.tight_layout()
        plt.show()

    return (vae1, vae2), score_model, z, labels

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, make_scorer, roc_auc_score
from sklearn.svm import SVC
import numpy as np

def calculate_auroc(X, y):
    X = X.detach().cpu().numpy()  # Convert tensors to numpy arrays for sklearn compatibility
    y = y.view((-1,)).cpu().numpy()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the hyperparameter grid for tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],               # Regularization parameter
        'kernel': ['rbf'],                    # Keeping 'rbf' kernel as in original code
        'gamma': [1, 0.1, 0.01, 0.001],       # Kernel coefficient
    }

    # Create an SVC model
    svc = SVC(probability=True)

    # Define a custom scorer based on AUROC for cross-validation
    auroc_scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Use GridSearchCV to tune hyperparameters with k-fold cross-validation (k=5)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=auroc_scorer, cv=5, verbose=1, n_jobs=-1)

    # Fit the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best hyperparameters: {best_params}")

    # Evaluate the model on the test set
    y_scores = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    return roc_auc

