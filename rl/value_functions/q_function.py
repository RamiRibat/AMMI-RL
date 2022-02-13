import random
import numpy as np
import torch as T
nn = T.nn

from rl.networks.mlp import MLPNet


class SoftQFunction(nn.Module):
    """
    Soft Q-Function
    """
    def __init__(self, obs_dim, act_dim, net_configs, seed):
        # print('Initialize QFunction!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        optimizer = 'T.optim.' + net_configs['optimizer']
        lr = net_configs['lr']

        super().__init__() # To automatically use forward

        self.q1 = MLPNet(obs_dim + act_dim, 1, net_configs, seed)
        self.q2 = MLPNet(obs_dim + act_dim, 1, net_configs, seed)
        self.Qs = [self.q1, self.q2]
        
        self.optimizer = eval(optimizer)(self.parameters(), lr)


    def forward(self, o, a):
        q_inputs = T.cat([o, a], dim=-1)
        return tuple(Q(q_inputs) for Q in self.Qs)
      