# TODO: model
import random
import copy
import typing
import logging

import numpy as np
from numpy.random.mtrand import normal
import torch as T
from torch._C import dtype
from torch.distributions.normal import Normal
nn = T.nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl


# from rl.networks.mlp import MLPNet


LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20


class RLDataset(IterableDataset):
    pass


class DataModule(pl.LightningDataModule):
    pass




class SimpleModel(pl.LightningModule):
    def __init__(self, obs_dim, act_dim, rew_dim,
                 net_configs, device, seed) -> None:
        print('Initialize SimpleModel!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        self.net_configs = net_configs
        self.device = device
        net_arch = net_configs['arch']

        self.mu_log_sigma_net = MLPNet(obs_dim + act_dim, 0, net_configs, seed)
        self.mu = nn.Linear(net_arch[-1], obs_dim + rew_dim)
        self.log_sigma = nn.Linear(net_arch[-1], obs_dim + rew_dim)

        self.max_log_sigma = nn.Parameter( T.ones([1, obs_dim + rew_dim]) / 2, requires_grad=False)
        self.min_log_sigma = nn.Parameter( -T.ones([1, obs_dim + rew_dim]) * 10, requires_grad=False)
        self.reparam_noise = 1e-6

        super().__init__() # To automatically use 'def forward'


    def get_model_dist_params(self, ips):
        net_out = self.mu_log_sigma_net(ips)
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = self.max_log_sigma - (self.max_log_sigma - log_sigma)
        log_sigma = self.min_log_sigma + (log_sigma - self.min_log_sigma)
        sigma = T.exp(log_sigma)
        sigma_inv = T.exp(-log_sigma)
        return mu, log_sigma, sigma, sigma_inv


    def deterministic(self, mu):
        pass


    def forward(self, o, a,
                deterministic= False):
        ips = T.as_tensor(T.cat([o, a], dim=-1), dtype=T.float32).to(self.device)
        mu, log_sigma, sigma, sigma_inv = self.get_model_dist_params(
            T.as_tensor(ips, dtype=T.float32).to(self.device))

        if deterministic:
            predictions = self.deterministic(mu)
        else:
            normal_ditribution = Normal(mu, sigma)
            predictions = normal_ditribution.sample()

        return predictions, mu, log_sigma, sigma, sigma_inv


    def trainModel(self):
        pass


    """ Pytorch Lightning """
    def configure_optimizers(self):
        pass

    
    def training_step(self, batch, batch_idx):
        pass


    def get_progress_bar_dict(self):
        pass


    def compute_l2_loss(self):
        pass

