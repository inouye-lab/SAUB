import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, model, ema=None, ema_decay=0.99, sigma_min=0.01, sigma_max=50, num_timesteps=1000):
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
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), num_timesteps))
        self.num_timesteps = num_timesteps
        self.model = model
        self.loss_dict = {}
        self.total_loss = 0
        self.loss_counter = 0
        if ema is not None:
            self.ema = ema(model, decay=ema_decay)

        # Learnable parameter for residual score function and assures value between [0,1]
        self.lbda = nn.ParameterList([nn.Parameter(torch.tensor([0.0]))])

    # Compute denoising score matching loss
    def compute_DSM_loss(self, x, t, enc_mu=None, enc_sigma=None, alpha=None, turn_off_enc_sigma=False,
                         learn_lbda=False, is_mixing=False, is_residual=False, is_vanilla=False, is_LSGM=False,
                         divide_by_sigma=False):
        sigmas = self.discrete_sigma.to(x.device)[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x, device=x.device) * sigmas
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
            return (self.model(x, t) / self.discrete_sigma.to(x.device)[t.long()].view(x.shape[0],
                                                                          *([1] * len(x.shape[1:])))).detach()
        else:
            return self.model(x, t) / self.discrete_sigma.to(x.device)[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))

    # Our implementation of residual score function
    def get_residual_score_fn(self, x, t, enc_eps, enc_sigma, detach=False, turn_off_enc_sigma=False, learn_lbda=False):

        # turn on eval for detach
        if detach:
            self.model.eval()

        # Computes learnable score
        learnable_score = self.model(x, t) / self.discrete_sigma.to(x.device)[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))

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
            alpha = alpha
        alpha = alpha.to(x.device)

        if divide_by_sigma:
            learnable_score = alpha * self.model(x, t) / self.discrete_sigma.to(x.device)[t.long()].view(x.shape[0],
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
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)

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
            t = torch.randint(0, max_timestep, (x.shape[0],), device=x.device)

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
            t = torch.randint(0, max_timestep, (x.shape[0],), device=x.device)

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
        loss.backward()
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
                            is_residual=False, is_vanilla=False, device='cpu'):
        x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], x_num), np.linspace(ylim[0], ylim[1], y_num))
        x_ = torch.from_numpy(x.reshape(-1, 1)).type(torch.float)
        y_ = torch.from_numpy(y.reshape(-1, 1)).type(torch.float)

        input = torch.hstack((x_, y_)).to(device)

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
            score_fn = self.get_mixing_score_fn(input, torch.ones((x_num * y_num,), device=input.device).type(torch.long) * t,
                                                detach=True, alpha=alpha, is_vanilla=is_vanilla)
        elif is_residual:
            score_fn = self.get_mixing_score_fn(input, torch.ones((x_num * y_num,), device=input.device).type(torch.long) * t,
                                                detach=True, alpha=alpha, is_residual=True, is_vanilla=is_vanilla)
        else:
            score_fn = self.get_score_fn(input, torch.ones((x_num * y_num,), device=input.device).type(torch.long) * t,
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


class Score_fn_noise(nn.Module):
    def __init__(self, model, ema=None, ema_decay=0.99, sigma_min=0.01, sigma_max=50, num_timesteps=1000,
                 is_add_latent_noise=False):
        """Construct a score function model.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          num_timestep: number of discretization steps
        """
        super(Score_fn_noise, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigma = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), num_timesteps))
        self.num_timesteps = num_timesteps
        self.model = model
        self.loss_dict = {}
        self.total_loss = 0
        self.loss_counter = 0
        self.is_add_latent_noise = is_add_latent_noise
        if ema is not None:
            self.ema = ema(model, decay=ema_decay)

        # Learnable parameter for residual score function and assures value between [0,1]
        self.lbda = nn.ParameterList([nn.Parameter(torch.tensor([0.0]))])
    #
    # def to_device(self):
    #     self.model = self.model.to(self.device)

    # Compute denoising score matching loss
    def compute_DSM_loss(self, x, t, latent_noise_idx=None, enc_mu=None, enc_sigma=None, alpha=None,
                         turn_off_enc_sigma=False, learn_lbda=False, is_mixing=False, is_residual=False,
                         is_vanilla=False, is_LSGM=False, divide_by_sigma=False):
        sigmas = self.discrete_sigma.to(x.device)[t.long()].view(x.shape[0], *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x, device=x.device) * sigmas
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
            alpha = alpha.to(x.device)

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
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
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

        self.model.train()

        # TODO: Add ema optimization
        if max_timestep is None or max_timestep > self.num_timesteps:
            max_timestep = self.num_timesteps

        if t is None:
            t = torch.randint(0, max_timestep, (x.shape[0],), device=x.device)

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
            t = torch.randint(0, max_timestep, (x.shape[0],), device=x.device)

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
                            is_residual=False, is_vanilla=False, device='cpu'):
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
