import sys

import torch
import torch.nn as nn
import lpips
from torchvision import transforms
import open_clip
import os

from typing import Union

class CLIPModule(nn.Module):

    def __init__(self, model_name, pretrained_dataset, device):
        super().__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_dataset,
            cache_dir=os.path.join(os.getcwd(), "..", "data"),
            device=device
        )

        self.device = device
        self.model = model
        self.preprocess = preprocess

    def to_device(self, device):

        self.device = device
        self.model = self.model.to(device)

    def get_dist_mat(self, batch, block_index_list=None):

        batch = batch.to(self.device)
        def batch_preprocess(image_batch, preprocess):
            to_pil_image = transforms.ToPILImage()
            image_stack = torch.concat([preprocess(to_pil_image(img)).unsqueeze(0) for img in image_batch])
            return image_stack

        def cosine_dist_matrix(batch1, batch2, dim=1, eps=1e-8):
            # Normalize the batches along the specified dimension
            batch1_norm = batch1 / torch.norm(batch1, dim=dim, keepdim=True).clamp(min=eps)
            batch2_norm = batch2 / torch.norm(batch2, dim=dim, keepdim=True).clamp(min=eps)

            # Calculate the cosine similarity matrix
            similarity_matrix = torch.mm(batch1_norm, batch2_norm.transpose(0, 1))
            return 1-similarity_matrix

        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():

            dist_matrix_array = []
            for block_index in block_index_list:
                batch_block = batch[block_index]
                batch_processed = batch_preprocess(batch_block, self.preprocess).to(self.device)
                image_features_1 = self.model.encode_image(batch_processed).detach()
                image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
                dist_matrix_array.append(cosine_dist_matrix(image_features_1, image_features_1, dim=-1))

        return dist_matrix_array


class LPIPSModule(nn.Module):

    def __init__(self, net_name, device):

        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net_name).to(device)

        self.device = device

    def to_device(self, device):

        self.device = device
        self.loss_fn = self.loss_fn.to(device)

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

    def __init__(self, metric_name, device, **kwargs):
        super().__init__()

        # print(f"GP module on device {device}")

        if metric_name == "CLIP":
            self.dist_module = CLIPModule(
                model_name=kwargs.get("model_name"),
                pretrained_dataset=kwargs.get("pretrained_dataset"),
                device=device,
            )
        elif metric_name == "LPIPS":
            self.dist_module = LPIPSModule(
                net_name=kwargs.get("net_name"),
                device=device,
            )
        else:
            sys.exit("Wrong metric for GP module")

        self.device = device

    def to_device(self, device):
        self.dist_module.to_device(device)

    def get_euc_dist_mat(self, z, block_index_list, p=2):

        dist_mat_list = []
        for block_index in block_index_list:
            if len(z[block_index].size()) > 2:
                z_block = z[block_index].view(len(z[block_index]), -1)
            else:
                z_block = z[block_index]

            # Expand dimensions to create pairs of vectors
            z1 = z_block.unsqueeze(1)  # (batch_size, 1, num_features)
            z2 = z_block.unsqueeze(0)  # (1, batch_size, num_features)

            # Calculate squared differences between each pair
            squared_diffs = (z1 - z2).pow(p)

            # Sum along the feature dimension (last dimension) to get squared distances
            squared_dists = squared_diffs.mean(dim=-1)

            # Take the square root to get Euclidean distances
            # dist_mat_list.append(squared_dists**(1/p))
            dist_mat_list.append(squared_dists)

        return dist_mat_list

    def compute_gp_loss(self, x_list: list, z_list: list, block_size: Union[int, None] = None) -> torch.Tensor:

        batch_size = len(x_list[0])
        if block_size is None:
            block_size = batch_size

        block_index_list = torch.split(torch.randperm(batch_size),
                                       split_size_or_sections=block_size)
        block_index_list = [block.to(self.device) for block in block_index_list]
        tot_gp_loss = 0

        for x, z in zip(x_list, z_list):
            # print(x.device, z.device)
            dist_mat_x_list = self.dist_module.get_dist_mat(x, block_index_list)
            # print(dist_mat_x_list[0])
            dist_mat_z_list = self.get_euc_dist_mat(z, block_index_list)

            for dist_mat_x, dist_mat_z in zip(dist_mat_x_list, dist_mat_z_list):
                # print(dist_mat_x.size(), dist_mat_x.max())
                dist_mat_x /= dist_mat_x.max()

                tot_gp_loss += ((dist_mat_x-dist_mat_z)**2).sum()/(len(dist_mat_x)**2)

        return tot_gp_loss


