import torch
import torch.nn as nn
import numpy as np

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
        loss += ((p_dist_x-p_dist_z)**2).sum()
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
        loss += torch.sum(torch.abs(pairwise_distances(X)-pairwise_distances(Z)))
    return loss