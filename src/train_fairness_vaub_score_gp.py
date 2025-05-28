import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import umap
import matplotlib.colors as mcolors
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim


import os, sys
# print(sys.path)
# print(f"current path: {os.getcwd()}")
sys.path.append(os.getcwd())
from notebooks.util import UNet, Score_fn, SimpleLinearClassifier, get_dataset_adult, vae_loss_lambda, conditional_errors, UNet_simple


class LinearEncoder(nn.Module):
    def __init__(self, input_features=1, latent_features=1, hidden_features=3, ):
        super(LinearEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_features),
            nn.ReLU(),
            # nn.Linear(in_features=hidden_features, out_features=hidden_features),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=latent_features * 2),
        )
        self.bn_mu = nn.BatchNorm1d(latent_features, affine=False)
        # self.bn_var = nn.BatchNorm1d(latent_features, affine=True)

    def forward(self, x):
        output = self.enc(x)
        mu, log_var = output.chunk(2, dim=-1)
        # log_var = 7 * (torch.sigmoid(log_var) - 0.5)
        log_var = torch.clamp(log_var, max=4)
        return self.bn_mu(mu), log_var


class LinearDecoder(nn.Module):
    def __init__(self, latent_features=1, output_features=1, hidden_features=3):
        super(LinearDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=output_features),
        )

    def forward(self, z):
        return self.dec(z)


class LinearVAE(nn.Module):
    def __init__(self, encoder, decoder):

        super(LinearVAE, self).__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, x):
        mu, log_var = self.enc(x)
        z = self.reparameterize(mu, log_var)
        return self.dec(z), z, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_loss

    def init_weights(self, scale=0.1):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-scale, b=scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(weights_init)


class LinearVAEshared(nn.Module):
    def __init__(self, input_features=1, latent_features=1, hidden_features=3):
        super().__init__()
        self.enc = LinearEncoder(input_features, latent_features, hidden_features)
        self.dec1 = LinearDecoder(latent_features, input_features, hidden_features)
        self.dec2 = LinearDecoder(latent_features, input_features, hidden_features)

    def forward(self, x, domain_idx):
        mu, log_var = self.enc(x)
        z = self.reparameterize(mu, log_var)
        if domain_idx == 0:
            x_recon = self.dec1(z)
        else:
            x_recon = self.dec2(z)
        return x_recon, z, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    #
    # def decode(self, z):
    #     return self.dec(z)

    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_loss

    def init_weights(self, scale=0.1):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-scale, b=scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(weights_init)


class UNet_simplest(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, multiplier=4):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, multiplier * in_dim),
            nn.SiLU(),
            nn.Linear(multiplier * in_dim, multiplier * in_dim),
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(multiplier * in_dim + embedding_dim, multiplier * in_dim),
            nn.SiLU(),
            nn.Linear(multiplier * in_dim, out_dim)
        )

        # Define time step embedding layer for decoder
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)

    def forward(self, x, timestep, latent_noise_idx=None):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(torch.hstack((x, self.embedding(timestep))))

        return x


def vae_loss_function(recon_x, x, mean, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence: 0.5 * sum(1 + logvar - mean^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Total loss
    return recon_loss, kl_loss


def visualize_lists(
        loss_1_list, loss_2_list,
        recon_loss_1_list, recon_loss_2_list,
        kl_loss_1_list, kl_loss_2_list,
        z1_var_list, z2_var_list,
        z1_mean_list, z2_mean_list
):
    # Create a figure with 2 rows and 5 columns for subplots
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.tight_layout(pad=4.0)

    # First row (loss_1, recon_loss_1, kl_loss_1, z1_var, z1_mean)
    axs[0, 0].plot(loss_1_list, label='loss_1', color='blue')
    axs[0, 0].set_title('Loss 1')
    axs[0, 0].legend()

    axs[0, 1].plot(recon_loss_1_list, label='recon_loss_1', color='blue')
    axs[0, 1].set_title('Recon Loss 1')
    axs[0, 1].legend()

    axs[0, 2].plot(kl_loss_1_list, label='kl_loss_1', color='blue')
    axs[0, 2].set_title('KL Loss 1')
    axs[0, 2].legend()

    axs[0, 3].plot(z1_var_list, label='z1_var', color='blue')
    axs[0, 3].set_title('Z Variance 1')
    axs[0, 3].legend()

    axs[0, 4].plot(z1_mean_list, label='z1_mean', color='blue')
    axs[0, 4].set_title('Z Mean 1')
    axs[0, 4].legend()

    # Second row (loss_2, recon_loss_2, kl_loss_2, z2_var, z2_mean)
    axs[1, 0].plot(loss_2_list, label='loss_2', color='green')
    axs[1, 0].set_title('Loss 2')
    axs[1, 0].legend()

    axs[1, 1].plot(recon_loss_2_list, label='recon_loss_2', color='green')
    axs[1, 1].set_title('Recon Loss 2')
    axs[1, 1].legend()

    axs[1, 2].plot(kl_loss_2_list, label='kl_loss_2', color='green')
    axs[1, 2].set_title('KL Loss 2')
    axs[1, 2].legend()

    axs[1, 3].plot(z2_var_list, label='z2_var', color='green')
    axs[1, 3].set_title('Z Variance 2')
    axs[1, 3].legend()

    axs[1, 4].plot(z2_mean_list, label='z2_mean', color='green')
    axs[1, 4].set_title('Z Mean 2')
    axs[1, 4].legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()


def calculate_gp_loss(X_list, Z_list):
    loss = 0

    def pairwise_distances(x):
        return torch.cdist(x, x)

    for i, (X, Z) in enumerate(zip(X_list, Z_list)):
        # ground_truth = mask_out(cos_distances(X), threshold=threshold)
        # ground_truth = mask_out(pairwise_distances(X), threshold=threshold)
        dist_x = pairwise_distances(X)
        dist_x_normalized = dist_x / dist_x.max()
        dist_z = pairwise_distances(Z)
        dist_z_normalized = dist_z / dist_z.max()
        loss += torch.mean(torch.abs(dist_x_normalized - dist_z_normalized))

    return loss

def display_umap_for_latent_multi(axes_total, epoch, z_list, label_list, num_samples=200):
    # Combine the datasets
    data = np.vstack([z[:num_samples].cpu() for z in z_list])
    labels = np.concatenate([label[:num_samples].cpu() for label in label_list])
    domains = np.concatenate([[f'domain{i}'] * len(z[:num_samples].cpu()) for i, z in enumerate(z_list)])

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
        # print(subset.shape)
        axes_total[1].scatter(subset['UMAP1'], subset['UMAP2'], c=[colors[domain]], marker="s",
                              alpha=0.6,
                              edgecolors='w', linewidth=0.5, label=f'{domain}')

    # Create a combined legend
    axes_total[1].set_title(f'UMAP Visualization of Latent Space at Epoch {epoch} in domain color')
    axes_total[1].legend(title='Domain')

    # Define markers and colors
    unique_labels = np.unique(labels)
    # print(unique_labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Plot the data
    for label in unique_labels:
        subset = df[(df['label'] == label)]
        axes_total[0].scatter(subset['UMAP1'], subset['UMAP2'], c=[color_map[label]], marker='o', alpha=0.6,
                              edgecolors='w', linewidth=0.5, label=f'{label}')

    axes_total[0].set_title(f'UMAP Visualization of Latent Space at Epoch {epoch}')
    axes_total[0].legend(title='Label')

    return axes_total

def main(args):

    import wandb

    run = wandb.init(project='fairness-vaub-gp-fixed',
                     config=args,
                     dir="./vaub-gp-pl-hydra/logs/fairness-vaub-gp")
    run.name = args.name

    # Parameters
    sigma_max = args.sigma_max
    recon_lambda = args.recon_lambda
    vaub_lambda = args.vaub_lambda
    gp_lambda = args.gp_lambda
    classifier_lambda = args.classifier_lambda
    loops = args.loops
    init_scale = args.init_scale
    device = args.device
    num_epochs = args.num_epochs
    lr_vae = args.lr_vae
    lr_cls = 1e-4
    lr_score = 1e-4

    # Constants
    batch_size = 2048
    n_print_per_epoch = 20
    timesteps = 100
    input_features = 114
    hidden_features = 50
    latent_features = 8
    multiplier_unet = 1
    sigma_min = 0.01
    data_dir = "vaub-gp-pl-hydra/data"

    train_dataset, test_dataset = get_dataset_adult(data_dir)
    # split train dataset into train and validation with 0.9/0.1
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9),
    #                                                                            len(train_dataset) - int(
    #                                                                                len(train_dataset) * 0.9)])
    # train_loader, val_loader, test_loader = get_loaders_adult(train_dataset, val_dataset, test_dataset, batch_size)
    # train_loader_score, val_loader_score, test_loader_score = get_loaders_adult(train_dataset, val_dataset, test_dataset, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader_score = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Setup Model
    enc = LinearEncoder(input_features, latent_features, hidden_features)
    dec1 = LinearDecoder(latent_features, input_features, hidden_features)
    dec2 = LinearDecoder(latent_features, input_features, hidden_features)

    fair_vae1 = LinearVAE(encoder=enc, decoder=dec1).to(device)
    fair_vae2 = LinearVAE(encoder=enc, decoder=dec2).to(device)

    score_model = Score_fn(
        UNet_simple(in_dim=latent_features, out_dim=latent_features, num_timesteps=timesteps, is_add_latent_noise=False,
             num_latent_noise_scale=1, is_warm_init=False, multiplier=multiplier_unet), sigma_min=sigma_min,
        sigma_max=sigma_max, num_timesteps=timesteps, is_add_latent_noise=False, device=device).to(device)

    fair_vae1.init_weights(scale=init_scale)
    fair_vae2.init_weights(scale=init_scale)

    classifier = SimpleLinearClassifier(input_dim=latent_features, hidden_features=32, num_classes=2).to(device)

    optimizer_vae1 = optim.AdamW(fair_vae1.parameters(), lr=lr_vae, weight_decay=1e-2)
    optimizer_vae2 = optim.AdamW(fair_vae2.parameters(), lr=lr_vae, weight_decay=1e-2)
    optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=lr_cls, weight_decay=1e-2)
    optimizer_score = torch.optim.AdamW(score_model.parameters(), lr=lr_score, weight_decay=1e-2)


    # Training
    for epoch in tqdm(range(num_epochs)):

        for i, batch in enumerate(train_loader):

            fair_vae1.train()
            fair_vae2.train()

            optimizer_vae1.zero_grad()
            optimizer_vae2.zero_grad()
            optimizer_classifier.zero_grad()

            xs, ys, attrs = batch
            x1, x2 = xs[attrs == 0].to(device), xs[attrs == 1].to(device)
            label1, label2 = ys[attrs == 0].to(device), ys[attrs == 1].to(device)

            # x1, x2 = x1.to(device), x2.to(device)  # Reshape

            recon_x1, z1, mean1, logvar1 = fair_vae1(x1)
            recon_x2, z2, mean2, logvar2 = fair_vae2(x2)

            x_list, recon_x_list = [x1, x2], [recon_x1, recon_x2]
            z, mean, logvar, labels = torch.vstack((z1, z2)), torch.vstack((mean1, mean2)), torch.vstack(
                (logvar1, logvar2)), torch.hstack((label1, label2))

            score = score_model.get_mixing_score_fn(z, 5 * torch.ones(z.shape[0], device=device).type(torch.long), None,
                                                    detach=True, is_residual=True, is_vanilla=True,
                                                    alpha=None) - 0.05 * z
            score_estimation_trick = (torch.matmul(score.unsqueeze(1), z.unsqueeze(-1))).sum() / (
                        z.shape[0] * z.shape[1])

            vaub_loss, recon_loss, kld_encoder_posterior, kld_prior = vae_loss_lambda(recon_x_list, x_list, mean,
                                                                                      logvar,
                                                                                      score=score_estimation_trick,
                                                                                      DSM=None, weighting=None)
            vaub_loss = vaub_lambda * vaub_loss
            recon_loss = recon_lambda * recon_loss

            gp_loss = gp_lambda * calculate_gp_loss([x1, x2], [z1, z2])


            output = classifier(mean.detach())
            classifier_loss = classifier_lambda * F.cross_entropy(output, labels, reduction='none').mean()

            loss = vaub_loss + recon_loss + gp_loss

            loss.backward(retain_graph=True)
            classifier_loss.backward()

            optimizer_vae1.step()
            optimizer_vae2.step()
            optimizer_classifier.step()

            run.log({
                "Loss/loss": loss.item(),
                "Loss/vaub_loss": vaub_loss.item(),
                "Loss/recon_loss": recon_loss.item(),
                "Loss/gp_loss": gp_loss.item(),
                "Loss/classifier_loss": classifier_loss.item(),
                "Loss_detail/kld_encoder_posterior": kld_encoder_posterior.item(),
                "Loss_detail/score": score.mean().item(),
                "Latent/z1_var": torch.exp(logvar1.detach().cpu()).mean().item(),
                "Latent/z2_var": torch.exp(logvar2.detach().cpu()).mean().item(),
                "Latent/z1_mean": mean1.detach().cpu().mean().item(),
                "Latent/z2_mean": mean2.detach().cpu().mean().item(),
            })

            # Update Score Function
            for loop in range(loops):
                xs, ys, attrs = next(iter(train_loader_score))
                x1, x2 = xs[attrs == 0].to(device), xs[attrs == 1].to(device)
                label1, label2 = ys[attrs == 0].to(device), ys[attrs == 1].to(device)

                # x1, x2 = x1.to(device), x2.to(device)  # Reshape
                fair_vae1.eval()
                fair_vae2.eval()

                recon_x1, z1, mean1, logvar1 = fair_vae1(x1)
                recon_x2, z2, mean2, logvar2 = fair_vae2(x2)

                z = torch.vstack((z1.detach(), z2.detach()))

                dsm_loss = score_model.update_score_fn(z, latent_noise_idx=None, optimizer=optimizer_score,
                                                       max_timestep=None, verbose=True, is_mixing=True,
                                                       is_residual=True, is_vanilla=True, alpha=None)
                run.log({
                    "Loss_detail/DSM": dsm_loss.item(),
                })

        # Print every 25 epochs
        if (epoch + 1) % n_print_per_epoch == 0 or epoch == 0:
            fig_total, axes_total = plt.subplots(1, 2, figsize=(20, 10))
            plt_umap = display_umap_for_latent_multi(
                axes_total=axes_total,
                epoch=epoch,
                z_list=[z1.detach(), z2.detach()],
                label_list=[label1, label2],
            )
            run.log({"UMAP": wandb.Image(fig_total)})

            target_insts = torch.from_numpy(train_dataset.X).float().to(device)
            target_labels = np.argmax(train_dataset.Y, axis=1)
            target_attrs = np.argmax(train_dataset.A, axis=1)
            test_idx = target_attrs == 0
            conditional_idx = target_labels == 0

            fair_vae1.eval()
            _, _, mean1, _ = fair_vae1(target_insts)
            preds_labels = torch.max(classifier(mean1), 1)[1].cpu().numpy()
            cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)

            # print(f"Epoch {epoch}/{num_epochs}: Loss {loss.item():.4f}  " +
            #       f"\nOverall predicted error = {cls_error:.2f}, Err|A=0 = {error_0:.2f}, Err|A=1 = {error_1:.2f}")

            pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
            cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
            cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
            cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
            cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
            cls_error, _, _ = conditional_errors(preds_labels, target_labels, target_attrs)

            run.log({
                "Fairness metric/Overall Error": cls_error,
                "Fairness metric/Joint Error": error_0 + error_1,
                "Fairness metric/Error Gap": np.abs(error_0 - error_1),
                "Fairness metric/DP Gap": np.abs(pred_0 - pred_1),
                "Fairness metric/Equalized Odds Y = 0": np.abs(cond_00 - cond_10),
                "Fairness metric/Equalized Odds Y = 1": np.abs(cond_01 - cond_11),
            })

    target_insts = torch.from_numpy(test_dataset.X).float().to(device)
    target_labels = np.argmax(test_dataset.Y, axis=1)
    target_attrs = np.argmax(test_dataset.A, axis=1)
    test_idx = target_attrs == 0
    conditional_idx = target_labels == 0

    fair_vae1.eval()
    _, _, mean1, _ = fair_vae1(target_insts)
    preds_labels = torch.max(classifier(mean1), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)

    pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
    cls_error, _, _ = conditional_errors(preds_labels, target_labels, target_attrs)

    run.log({
        "Test fairness metric/Overall Error": cls_error,
        "Test fairness metric/Joint Error": error_0 + error_1,
        "Test fairness metric/Error Gap": np.abs(error_0 - error_1),
        "Test fairness metric/DP Gap": np.abs(pred_0 - pred_1),
        "Test fairness metric/Equalized Odds Y = 0": np.abs(cond_00 - cond_10),
        "Test fairness metric/Equalized Odds Y = 1": np.abs(cond_01 - cond_11),
    })

def main_alt(args):

    import wandb

    run = wandb.init(project='fairness-vaub-gp-fixed',
                     config=args,
                     dir="vaub-gp-pl-hydra/logs/fairness-vaub-gp")
    run.name = args.name

    # Parameters
    sigma_max = args.sigma_max
    recon_lambda = args.recon_lambda
    vaub_lambda = args.vaub_lambda
    gp_lambda = args.gp_lambda
    classifier_lambda = args.classifier_lambda
    loops = args.loops
    init_scale = args.init_scale
    device = args.device
    num_epochs = args.num_epochs
    lr_vae = args.lr_vae
    lr_cls = 1e-4
    lr_score = 1e-4

    # Constants
    batch_size = 2048
    n_print_per_epoch = 20
    timesteps = 100
    input_features = 114
    hidden_features = 64
    latent_features = 8
    multiplier_unet = 1
    sigma_min = 0.01
    data_dir = "vaub-gp-pl-hydra/data"

    train_dataset, test_dataset = get_dataset_adult(data_dir)
    # split train dataset into train and validation with 0.9/0.1
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9),
    #                                                                            len(train_dataset) - int(
    #                                                                                len(train_dataset) * 0.9)])
    # train_loader, val_loader, test_loader = get_loaders_adult(train_dataset, val_dataset, test_dataset, batch_size)
    # train_loader_score, val_loader_score, test_loader_score = get_loaders_adult(train_dataset, val_dataset, test_dataset, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader_score = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Setup Model
    fair_vae_shared = LinearVAEshared(input_features, latent_features, hidden_features).to(device)

    score_model = Score_fn(
        UNet_simple(in_dim=latent_features, out_dim=latent_features, num_timesteps=timesteps, is_add_latent_noise=False,
             num_latent_noise_scale=1, is_warm_init=False, multiplier=multiplier_unet), sigma_min=sigma_min,
        sigma_max=sigma_max, num_timesteps=timesteps, is_add_latent_noise=False, device=device).to(device)

    fair_vae_shared.init_weights(scale=init_scale)

    classifier = SimpleLinearClassifier(input_dim=latent_features).to(device)

    optimizer_vae_shared = optim.AdamW(fair_vae_shared.parameters(), lr=lr_vae, weight_decay=1e-2)
    optimizer_classifier = torch.optim.AdamW(classifier.parameters(), lr=lr_cls, weight_decay=1e-2)
    optimizer_score = torch.optim.AdamW(score_model.parameters(), lr=lr_score, weight_decay=1e-2)


    # Training
    for epoch in tqdm(range(num_epochs)):

        for i, batch in enumerate(train_loader):

            fair_vae_shared.train()

            optimizer_vae_shared.zero_grad()
            optimizer_classifier.zero_grad()

            xs, ys, attrs = batch
            x1, x2 = xs[attrs == 0].to(device), xs[attrs == 1].to(device)
            label1, label2 = ys[attrs == 0].to(device), ys[attrs == 1].to(device)

            # x1, x2 = x1.to(device), x2.to(device)  # Reshape

            recon_x1, z1, mean1, logvar1 = fair_vae_shared(x1, domain_idx=0)
            recon_x2, z2, mean2, logvar2 = fair_vae_shared(x2, domain_idx=1)

            x_list, recon_x_list = [x1, x2], [recon_x1, recon_x2]
            z, mean, logvar, labels = torch.vstack((z1, z2)), torch.vstack((mean1, mean2)), torch.vstack(
                (logvar1, logvar2)), torch.hstack((label1, label2))

            score = score_model.get_mixing_score_fn(z, 5 * torch.ones(z.shape[0], device=device).type(torch.long), None,
                                                    detach=True, is_residual=True, is_vanilla=True,
                                                    alpha=None) - 0.05 * z
            score_estimation_trick = (torch.matmul(score.unsqueeze(1), z.unsqueeze(-1))).sum() / (
                        z.shape[0] * z.shape[1])

            vaub_loss, recon_loss, kld_encoder_posterior, kld_prior = vae_loss_lambda(recon_x_list, x_list, mean,
                                                                                      logvar,
                                                                                      score=score_estimation_trick,
                                                                                      DSM=None, weighting=None)
            vaub_loss = vaub_lambda * vaub_loss
            recon_loss = recon_lambda * recon_loss

            gp_loss = gp_lambda * calculate_gp_loss([x1, x2], [z1, z2])

            variance_penalty = args.var_lambda * (torch.exp(logvar1).mean() + torch.exp(logvar2).mean())


            # print(args.detach_classifier)

            if args.detach_classifier:
                output = classifier(mean.detach())
                classifier_loss = F.cross_entropy(output, labels, reduction='none').mean()

                loss = vaub_loss + recon_loss + gp_loss + variance_penalty

                loss.backward(retain_graph=True)
                classifier_loss.backward()

                optimizer_vae_shared.step()
                optimizer_classifier.step()
            else:
                # print("Not detaching classifier")
                output = classifier(mean)
                classifier_loss = classifier_lambda * F.cross_entropy(output, labels, reduction='none').mean()

                loss = vaub_loss + recon_loss + gp_loss + classifier_loss + variance_penalty

                loss.backward()

                optimizer_vae_shared.step()
                optimizer_classifier.step()

            run.log({
                "Loss/loss": loss.item(),
                "Loss/vaub_loss": vaub_loss.item(),
                "Loss/recon_loss": recon_loss.item(),
                "Loss/gp_loss": gp_loss.item(),
                "Loss/classifier_loss": classifier_loss.item(),
                "Loss_detail/kld_encoder_posterior": kld_encoder_posterior.item(),
                "Loss_detail/score": score.mean().item(),
                "Latent/z1_var": torch.exp(logvar1.detach().cpu()).mean().item(),
                "Latent/z2_var": torch.exp(logvar2.detach().cpu()).mean().item(),
                "Latent/z1_mean": mean1.detach().cpu().mean().item(),
                "Latent/z2_mean": mean2.detach().cpu().mean().item(),
            })

            # Update Score Function
            for loop in range(loops):
                xs, ys, attrs = next(iter(train_loader_score))
                x1, x2 = xs[attrs == 0].to(device), xs[attrs == 1].to(device)
                label1, label2 = ys[attrs == 0].to(device), ys[attrs == 1].to(device)

                # x1, x2 = x1.to(device), x2.to(device)  # Reshape
                fair_vae_shared.eval()

                recon_x1, z1, mean1, logvar1 = fair_vae_shared(x1, domain_idx=0)
                recon_x2, z2, mean2, logvar2 = fair_vae_shared(x2, domain_idx=1)

                z = torch.vstack((z1.detach(), z2.detach()))

                dsm_loss = score_model.update_score_fn(z, latent_noise_idx=None, optimizer=optimizer_score,
                                                       max_timestep=None, verbose=True, is_mixing=True,
                                                       is_residual=True, is_vanilla=True, alpha=None)
                run.log({
                    "Loss_detail/DSM": dsm_loss.item(),
                })

        # Print every 25 epochs
        if (epoch + 1) % n_print_per_epoch == 0 or epoch == 0:
            fig_total, axes_total = plt.subplots(1, 2, figsize=(20, 10))
            plt_umap = display_umap_for_latent_multi(
                axes_total=axes_total,
                epoch=epoch,
                z_list=[z1.detach(), z2.detach()],
                label_list=[label1, label2],
            )
            run.log({"UMAP": wandb.Image(fig_total)})
            plt.close()

        target_insts = torch.from_numpy(train_dataset.X).float().to(device)
        target_labels = np.argmax(train_dataset.Y, axis=1)
        target_attrs = np.argmax(train_dataset.A, axis=1)
        test_idx = target_attrs == 0
        conditional_idx = target_labels == 0

        fair_vae_shared.eval()
        _, _, mean1, _ = fair_vae_shared(target_insts, domain_idx=0)
        preds_labels = torch.max(classifier(mean1), 1)[1].cpu().numpy()
        cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)

        # print(f"Epoch {epoch}/{num_epochs}: Loss {loss.item():.4f}  " +
        #       f"\nOverall predicted error = {cls_error:.2f}, Err|A=0 = {error_0:.2f}, Err|A=1 = {error_1:.2f}")

        pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
        cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
        cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
        cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
        cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
        cls_error, _, _ = conditional_errors(preds_labels, target_labels, target_attrs)

        run.log({
            "Fairness metric/Overall Error": cls_error,
            "Fairness metric/Joint Error": error_0 + error_1,
            "Fairness metric/Error Gap": np.abs(error_0 - error_1),
            "Fairness metric/DP Gap": np.abs(pred_0 - pred_1),
            "Fairness metric/Equalized Odds Y = 0": np.abs(cond_00 - cond_10),
            "Fairness metric/Equalized Odds Y = 1": np.abs(cond_01 - cond_11),
        })

        target_insts = torch.from_numpy(test_dataset.X).float().to(device)
        target_labels = np.argmax(test_dataset.Y, axis=1)
        target_attrs = np.argmax(test_dataset.A, axis=1)
        test_idx = target_attrs == 0
        conditional_idx = target_labels == 0

        fair_vae_shared.eval()
        _, _, mean1, _ = fair_vae_shared(target_insts, domain_idx=0)
        preds_labels = torch.max(classifier(mean1), 1)[1].cpu().numpy()
        cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)

        pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
        cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
        cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
        cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
        cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
        cls_error, _, _ = conditional_errors(preds_labels, target_labels, target_attrs)

        run.log({
            "Test fairness metric/Overall Error": cls_error,
            "Test fairness metric/Joint Error": error_0 + error_1,
            "Test fairness metric/Error Gap": np.abs(error_0 - error_1),
            "Test fairness metric/DP Gap": np.abs(pred_0 - pred_1),
            "Test fairness metric/Equalized Odds Y = 0": np.abs(cond_00 - cond_10),
            "Test fairness metric/Equalized Odds Y = 1": np.abs(cond_01 - cond_11),
        })


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Train VAE on Adult dataset with command-line parameters.')

    parser.add_argument('--sigma_max', type=float, default=1, help='Maximum sigma value (default: 1)')
    parser.add_argument('--recon_lambda', type=float, default=1, help='Reconstruction lambda (default: 1)')
    parser.add_argument('--vaub_lambda', type=float, default=1, help='VAUB lambda (default: 1)')
    parser.add_argument('--gp_lambda', type=float, default=0, help='GP lambda (default: 0)')
    parser.add_argument('--classifier_lambda', type=float, default=1, help='Classifier lambda (default: 1e-3)')
    parser.add_argument('--loops', type=int, default=1, help='Number of loops (default: 5)')
    parser.add_argument('--init_scale', type=float, default=0.2, help='Initialization scale (default: 0.2)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--name', type=str, default='test', help='Run name (default: fairness-vaub-gp)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr_vae', type=float, default=5e-4, help='Learning rate for VAE (default)')
    parser.add_argument('--detach_classifier', action='store_false', help='Detach classifier during training')
    parser.add_argument('--var_lambda', type=float, default=1e-1, help='Variance penalty lambda (default: 1e-3)')

    args = parser.parse_args()

    main_alt(args)
