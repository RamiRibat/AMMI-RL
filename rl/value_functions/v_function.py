import random
import numpy as np
import torch as T
nn = T.nn

from rl.networks.mlp import MLPNet

def init_weights_(l):
	if isinstance(l, nn.Linear):
		nn.init.xavier_uniform_(l.weight, 1.0)
		nn.init.uniform_(l.bias, 0.0)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer


class VFunction(nn.Module):
    """
    V-Function
    """
    def __init__(self, obs_dim, act_dim, net_configs, device, seed):
        # print('init VFunction!')

	    optimizer = 'T.optim.' + net_configs['optimizer']
	    lr = net_configs['lr']

	    super().__init__() # To automatically use forward

	    # self.v = MLPNet(obs_dim, 1, net_configs)
	    # self.apply(init_weights_)

	    self.v = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
							   layer_init(nn.Linear(64, 64)), nn.Tanh(),
							   layer_init(nn.Linear(64, 1), std=1.0))

	    self.to(device)

	    self.optimizer = eval(optimizer)(self.parameters(), lr, eps=1e-5)


    def forward(self, o):
        return self.v(T.as_tensor(o, dtype=T.float32))
