import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def lpls_loss(y_predict, y_ture):
    u = y_predict[:, :, 0]
    sigma = y_predict[:, :, 1]
    sigma_min = torch.full_like(sigma, 1e-6)
    sigma = torch.maximum(sigma, sigma_min)
    loss = torch.sum(torch.log(2 * sigma) + torch.abs((y_ture - u) / (sigma)), dim=-1)
    loss = torch.sum(loss)
    if torch.isnan(loss):
        print('nan_train')
    return loss

def MLE_Gaussian(y_predict, y_true):
    # Gaussian likelihood
    u = y_predict[:, :, 0]
    sigma = y_predict[:, :, 1]
    sigma_min = torch.full_like(sigma, 1e-6)
    sigma = torch.maximum(sigma, sigma_min)
    taus = np.array([0.5])
    e = torch.sum(torch.log(sigma) + taus[0] * torch.square((y_true - u) / sigma))
    if torch.isnan(e):
        print('nan_train')
    return e

class Densegauss(nn.Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(Densegauss, self).__init__()
        self.n_in = n_input
        self.n_out = 2 * n_out_tasks
        self.n_tasks = n_out_tasks
        self.l1 = nn.Linear(self.n_in, self.n_out)

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            gamma, lognu = torch.split(x, self.n_tasks, dim=0)
        else:
            gamma, lognu = torch.split(x, self.n_tasks, dim=1)

        nu = F.softplus(lognu) + 1e-6

        return torch.stack([gamma, nu], dim=2).to(x.device)