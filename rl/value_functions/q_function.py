import random
import numpy as np
import torch as T
nn = T.nn

from rl.networks.mlp import MLPNet





def init_weights_(l):
	if isinstance(l, nn.Linear):
		nn.init.xavier_uniform_(l.weight, 1.0)
		nn.init.uniform_(l.bias, 0.0)





class QFunction(nn.Module):
    """
    Q-Function
    """
    def __init__(self, obs_dim, act_dim, net_configs, seed):
        # print('init QFunction!')
        # if seed: random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        optimizer = 'T.optim.' + net_configs['optimizer']
        lr = net_configs['lr']

        super(QFunction, self).__init__() # To automatically use forward

        self.q1 = MLPNet(obs_dim + act_dim, 1, net_configs)

        self.apply(init_weights_)

        self.to(device)

        self.optimizer = eval(optimizer)(self.parameters(), lr)



    def forward(self, o, a):
        q_inputs = T.cat([o, a], dim=-1)
        return self.q1(q_inputs)


class SoftQFunction(nn.Module):
    """
    Soft Q-Function
    """
    def __init__(self, obs_dim, act_dim, net_configs, seed):
        # print('init Soft QFunction!')
        # if seed: random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        optimizer = 'T.optim.' + net_configs['optimizer']
        lr = net_configs['lr']

        super(SoftQFunction, self).__init__() # To automatically use forward

        self.q1 = MLPNet(obs_dim + act_dim, 1, net_configs)
        self.q2 = MLPNet(obs_dim + act_dim, 1, net_configs)
        self.Qs = [self.q1, self.q2]

        self.apply(init_weights_)

        self.optimizer = eval(optimizer)(self.parameters(), lr)


    def forward(self, o, a):
        q_inputs = T.cat([o, a], dim=-1)
        return tuple(Q(q_inputs) for Q in self.Qs)
