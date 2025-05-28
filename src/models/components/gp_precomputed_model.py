import sys

import torch
import torch.nn as nn
import lpips
from torchvision import transforms
import open_clip
import os

from typing import Union


class CLIPModule(nn.Module):

    def __init__(self, device, dist_mode='cosine'):
        super().__init__()

        self.device = device
        self.dist_mode = dist_mode

    def to_device(self, device):

        self.device = device

    def get_dist_mat(self, encoded_batch, block_index_list=None):

        encoded_batch = encoded_batch.to(self.device)

        def cosine_dist_matrix(batch1, batch2, dim=1, eps=1e-8):
            # Normalize the batches along the specified dimension
            batch1_norm = batch1 / torch.norm(batch1, dim=dim, keepdim=True).clamp(min=eps)
            batch2_norm = batch2 / torch.norm(batch2, dim=dim, keepdim=True).clamp(min=eps)

            # Calculate the cosine similarity matrix
            similarity_matrix = torch.mm(batch1_norm, batch2_norm.transpose(0, 1))
            return 1-similarity_matrix

        def L22_dist_matrix(batch1, batch2):
            return torch.cdist(batch1, batch2, p=2)**2

        def L2_dist_matrix(batch1, batch2):
            return torch.cdist(batch1, batch2, p=2)

        dist_matrix_array = []
        for block_index in block_index_list:
            image_features_1 = encoded_batch[block_index]

            if self.dist_mode == 'cosine':
                image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
                dist_matrix_array.append(cosine_dist_matrix(image_features_1, image_features_1, dim=-1))
            elif self.dist_mode == 'L22':
                dist_matrix_array.append(L22_dist_matrix(image_features_1, image_features_1))
            elif self.dist_mode == 'L2':
                dist_matrix_array.append(L2_dist_matrix(image_features_1, image_features_1))
            else:
                raise ValueError(f"Invalid distance mode in x domain {self.dist_mode}")

        return dist_matrix_array


class LPIPSModule_(nn.Module):

    def __init__(self, net_name, device):

        super().__init__()
        # self.loss_fn = lpips.LPIPS(net=net_name).to(device)

        self.device = device

    def to_device(self, device):

        self.device = device
        # self.loss_fn = self.loss_fn.to(device)

    def get_dist_mat(self, batch, block_index_list=None):

        batch = batch.to(self.device)
        if block_index_list is None:
            block_index_list = [range(len(batch))]

        def lpips_dist_matrix(batch1, batch2):
            # print(f"batch 1 is in device {batch1.device}")
            # print(f"batch 2 is in device {batch2.device}")
            similarity_matrix = []
            for batch in batch2:
                similarity_matrix.append(self.loss_fn.forward(batch, batch1).squeeze().unsqueeze(0).detach())
                # Calculate the cosine similarity matrix
            return torch.concat(similarity_matrix)

        with torch.no_grad(), torch.cuda.amp.autocast():
            if len(batch.size())==3:
                batch = batch.unsqueeze(1)

            dist_matrix_array = []
            for block_index in block_index_list:
                batch_block = batch[block_index]
                dist_matrix_array.append(lpips_dist_matrix(batch_block, batch_block))

        return dist_matrix_array


class GPModule(nn.Module):

    def __init__(self, device, diff="MSE", mode="max", dist_x_mode="cosine", **kwargs):
        super().__init__()

        self.device = device
        self.mode = mode
        self.kwargs = kwargs
        self.diff = diff

        self.dist_module = CLIPModule(
            device=device,
            dist_mode=dist_x_mode
        )

        self.device = device
        self.dist_x_mode = dist_x_mode

    def to_device(self, device):
        self.dist_module.to_device(device)

    def get_euc_dist_mat(self, z, block_index_list, p=2):

        dist_mat_list = []
        for block_index in block_index_list:
            if len(z[block_index].size()) > 2:
                z_block = z[block_index].view(len(z[block_index]), -1)
            else:
                z_block = z[block_index]

            # # Expand dimensions to create pairs of vectors
            # z1 = z_block.unsqueeze(1)  # (batch_size, 1, num_features)
            # z2 = z_block.unsqueeze(0)  # (1, batch_size, num_features)
            #
            # # Calculate squared differences between each pair
            # squared_diffs = (z1 - z2).pow(p)
            #
            # # Sum along the feature dimension (last dimension) to get squared distances
            # squared_dists = squared_diffs.mean(dim=-1)
            if self.dist_x_mode == 'cosine' or self.dist_x_mode == 'L22':
                squared_dists = torch.cdist(z_block, z_block, p=p)**2
            elif self.dist_x_mode == 'L2':
                squared_dists = torch.cdist(z_block, z_block, p=p)

            # Take the square root to get Euclidean distances
            # dist_mat_list.append(squared_dists**(1/p))
            dist_mat_list.append(squared_dists)


        return dist_mat_list

    def compute_gp_loss(self,
                        enc_list: list,
                        z_list: list,
                        block_size: Union[int, None] = None,
                        ) -> torch.Tensor:

        batch_size = len(enc_list[0])
        if block_size is None:
            block_size = batch_size

        block_index_list = torch.split(torch.randperm(batch_size),
                                       split_size_or_sections=block_size)
        block_index_list = [block.to(self.device) for block in block_index_list]
        tot_gp_loss = 0

        for domain_idx, (enc, z) in enumerate(zip(enc_list, z_list)):
            # print(x.device, z.device)
            dist_mat_x_list = self.dist_module.get_dist_mat(enc, block_index_list)
            # print(dist_mat_x_list[0])
            dist_mat_z_list = self.get_euc_dist_mat(z, block_index_list)

            for block_idx, (dist_mat_x, dist_mat_z) in enumerate(zip(dist_mat_x_list, dist_mat_z_list)):
                # print(dist_mat_x.size(), dist_mat_x.max())
                if self.mode == "max":
                    dist_mat_x = dist_mat_x / dist_mat_x.max()
                    dist_mat_z = dist_mat_z / dist_mat_z.max()
                elif self.mode == "mean":
                    dist_mat_x = dist_mat_x / dist_mat_x.mean()
                    dist_mat_z = dist_mat_z / dist_mat_z.mean()
                elif self.mode == "binarize":
                    threshold = self.kwargs.get(f"threshold_{domain_idx}")
                    dist_mat_x = dist_mat_x / dist_mat_x.max()
                    dist_mat_z = dist_mat_z / dist_mat_z.max()
                    dist_mat_x[dist_mat_x > threshold] = 1
                    dist_mat_x[dist_mat_x <= threshold] = 0
                    dist_mat_z = dist_mat_z / dist_mat_z.max()
                elif self.mode == "logistic":
                    coef = self.kwargs.get(f"coef_{domain_idx}")
                    intercept = self.kwargs.get(f"intercept_{domain_idx}")
                    dist_mat_x = dist_mat_x / dist_mat_x.max()
                    dist_mat_z = dist_mat_z / dist_mat_z.max()
                    # dist_mat_x = 1 / (1 + torch.exp(-coef*(dist_mat_x-intercept)))
                    dist_mat_x = 1 / (1 + torch.exp(-(coef * dist_mat_x + intercept)))
                else:
                    raise ValueError(f"Invalid mode for distance matrix for x space{self.mode}")

                if self.diff == "MSE":
                    tot_gp_loss += ((dist_mat_x-dist_mat_z)**2).mean()
                elif self.diff == "MAE":
                    tot_gp_loss += (dist_mat_x-dist_mat_z).abs().mean()
                else:
                    raise ValueError(f"Invalid difference metric for distance matrix{self.diff}")

        return tot_gp_loss
