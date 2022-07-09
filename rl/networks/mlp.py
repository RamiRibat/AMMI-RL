import numpy as np
import torch as T

nn = T.nn
F = nn.functional
Parameter = nn.Parameter
Linear = nn.Linear
Sequential = nn.Sequential



# class Swish(nn.Module):
#     def __init__(self):
#         super(Swish, self).__init__()
#
#     def forward(self, x):
#         x = x * F.sigmoid(x)
#         return x


class MLPNet(nn.Module):
    def __init__(self, ip_dim, op_dim, net_configs):
        # print('Initialize MLP Network!')
        super(MLPNet, self).__init__() # To automatically use forward
        # random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        net_arch = net_configs['arch']
        activation = 'nn.' + net_configs['activation']
        op_activation = 'nn.Identity' # net_config['output_activation']

        if len(net_arch) > 0:
            layers = [Linear(ip_dim, net_arch[0]), eval(activation)()]
            for l in range(len(net_arch)-1):
                layers.extend([Linear(net_arch[l], net_arch[l+1]), eval(activation)()])
            if op_dim > 0:
                last_dim = net_arch[-1]
                layers.extend([Linear(last_dim, op_dim), eval(op_activation)()])
        else:
            raise 'No network arch!'

        # print('layers: ', layers)
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)
