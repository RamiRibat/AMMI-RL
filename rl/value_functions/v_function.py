import random
import numpy as np
import torch as T
nn = T.nn

from rl.networks.mlp import MLPNet


# init1
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer

# init2
def init_weights_(l):
	if isinstance(l, nn.Linear):
		nn.init.xavier_uniform_(l.weight, 1.0)
		nn.init.uniform_(l.bias, 0.0)


# init3
def init_weights_3(l):
	if isinstance(l, nn.Linear):
		nn.init.orthogonal_(l.weight, np.sqrt(2))
		nn.init.constant_(l.bias, 0.0)



# init4
def init_weights_4(l):
    if isinstance(l, nn.Linear):
        nn.init.xavier_uniform_(l.weight, 1.0)
        nn.init.uniform_(l.bias, 0.1)


def init_weights_B(l, std=np.sqrt(2), bias=0.0): # init1
# def init_weights_(l, std=1.0, bias=0.0): # init2
	if isinstance(l, nn.Linear):
		# nn.init.xavier_uniform_(l.weight, std)
		# nn.init.uniform_(l.bias, bias)
		nn.init.orthogonal_(l.weight, std)
		nn.init.constant_(l.bias, bias)




class VFunction(nn.Module):
    """
    V-Function
    """
    def __init__(self, obs_dim, act_dim, net_configs, device, seed):
	    print('Initialize V-function!')

	    optimizer = 'T.optim.' + net_configs['optimizer']
	    lr = net_configs['lr']
	    # hid = 64
	    # hid = 128
	    # hid = 256

	    super().__init__() # To automatically use forward

	    self.device = device

	    self.v = MLPNet(obs_dim, 1, net_configs)
	    # self.apply(init_weights_4)

	    # self.v = nn.Sequential(
		# 	layer_init(nn.Linear(obs_dim, hid)),
		# 	nn.Tanh(),
		# 	layer_init(nn.Linear(hid, hid)),
		# 	nn.Tanh(),
		# 	layer_init(nn.Linear(hid, 1), std=1.0)
		# 				)

	    print('V-function: ', self)

	    self.to(device)

	    self.optimizer = eval(optimizer)(self.parameters(), lr)


    def forward(self, o):
        if isinstance(o, T.Tensor):
        	o = o.to(self.device)
        else:
        	o = o
        return self.v(o)
