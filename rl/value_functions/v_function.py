import random
import numpy as np
import torch as T
nn = T.nn

from rl.networks.mlp import MLPNet


class VFunction(nn.Module):
    """
    V-Function
    """
    def __init__(self, obs_dim, act_dim, net_configs, seed):
        # print('init QFunction!')
        # if seed: random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        optimizer = 'T.optim.' + net_configs['optimizer']
        lr = net_configs['lr']

        super().__init__() # To automatically use forward

        self.v = MLPNet(obs_dim, 1, net_configs)

        self.optimizer = eval(optimizer)(self.parameters(), lr)


    def forward(self, o):
        return self.v(o)
