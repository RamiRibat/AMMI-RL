import random
import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn
F = T.nn.functional

from rl.networks.mlp import MLPNet


LOG_STD_MAX = 2
# LOG_STD_MIN = -5
LOG_STD_MIN = -20

epsilon = 1e-8

# # init1
# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     T.nn.init.orthogonal_(layer.weight, std)
#     T.nn.init.constant_(layer.bias, bias_const)
#     return layer
#
# def layer_init2(layer, std=0.0, bias_const=0.0):
#     T.nn.init.xavier_uniform_(layer.weight, std)
#     T.nn.init.uniform_(layer.bias, bias_const)
#     return layer

# init2
def init_weights_(l):
	if isinstance(l, nn.Linear):
		nn.init.xavier_uniform_(l.weight, 1.0)
		nn.init.uniform_(l.bias, 0.0)


# init2
def init_weights_ii(l):
	if isinstance(l, nn.Linear):
		nn.init.trunc_normal_(l.weight, mean=0.0, std=1e-4)
		nn.init.constant_(l.bias, 0.0)

def init_weights_iii(l):
	if isinstance(l, nn.Linear):
		nn.init.trunc_normal_(l.weight, mean=0.0, std=0.001)
		nn.init.constant_(l.bias, 0.0)




# Best for MB-PPO
class PPOPolicy(nn.Module):
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(PPOPolicy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		init_log_std = net_configs['init_log_std']

		self.mean = MLPNet(obs_dim, act_dim, net_configs)
		self.log_std = nn.Parameter(init_log_std * T.ones(act_dim, dtype=T.float32),
                                      requires_grad=net_configs['log_std_grad']) # (MF/MB)-PPO

		self.std_value = T.tensor([0.])

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			# self.apply(init_weights_)
			self.apply(init_weights_ii)
			# self.apply(init_weights_iii)

		self.act_dim = act_dim

		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr)

		print('Policy: ', self)


	def forward(self, obs, act=None,
                on_policy=True,
				reparameterize=False, # Default: True
				deterministic=False, # Default: False
				return_log_pi=True, # Default: False
				return_entropy=True, # Default: False
                return_pre_pi=False,
				):

		if isinstance(obs, T.Tensor):
			obs = (obs.to(self.device) - self.obs_bias) / (self.obs_scale + epsilon)
		else:
			obs = (obs - self.obs_bias.cpu().numpy()) / (self.obs_scale.cpu().numpy() + epsilon)

		mean, std = self.pi_mean_std(obs, on_policy)

		log_pi, entropy = None, None

		if deterministic:
			pre_pi = None
			with T.no_grad(): pi = T.tanh(mean)
		else:
			pre_pi, pi, log_pi, entropy = self.pi_prob(act,
                                               mean, std,
                                               on_policy,
                                               reparameterize,
                                               return_log_pi,
                                               return_entropy=True,
                                               return_pre_prob=return_pre_pi)
			pre_pi = (pre_pi * self.act_scale) + self.act_bias

		pi = (pi * self.act_scale) + self.act_bias

		return pre_pi, pi, log_pi, entropy


	def pi_mean_std(self, obs, on_policy=True):
		obs = T.as_tensor(obs, dtype=T.float32).to(self.device)

		mean = self.mean(obs)
		log_std = self.log_std
		std = T.exp(log_std)

		self.std_value = std

		return mean, std


	def pi_prob(self, act,
                mean, std,
                on_policy,
                reparameterize,
                return_log_prob,
                return_entropy,
                return_pre_prob=False):

		normal_ditribution = Normal(mean, std)

		if act is None:
			if reparameterize:
				sample = normal_ditribution.rsample()
			else:
				sample = normal_ditribution.sample()
			pre_prob, prob = sample, T.tanh(sample)
		else:
			pre_prob, prob = act, T.tanh(act)

		log_prob, entropy = None, None

		if return_log_prob:
			log_prob = normal_ditribution.log_prob(pre_prob)
			log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(axis=-1, keepdim=True)

		if return_entropy:
			entropy = normal_ditribution.entropy().sum(axis=-1, keepdims=True)

		return pre_prob, prob, log_prob, entropy


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(PPOPolicy, self).to(device)




# Best for MB-SAC
class SACPolicy(nn.Module): # B
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(SACPolicy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		# init_log_std = net_configs['init_log_std']

        # My suggestions:
		self.mean_and_log_std_bb = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		self.std_value = T.tensor([0.])

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			# self.apply(init_weights_)
			self.apply(init_weights_ii)

		self.act_dim = act_dim

		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr)

		print('Policy: ', self)


	def forward(self, obs, act=None,
                on_policy=True,
				reparameterize=False, # Default: True
				deterministic=False, # Default: False
				return_log_pi=True, # Default: False
				return_entropy=True, # Default: False
                return_pre_pi=False,
				):

		if isinstance(obs, T.Tensor):
			obs = (obs.to(self.device) - self.obs_bias) / (self.obs_scale + epsilon)
		else:
			obs = (obs - self.obs_bias.cpu().numpy()) / (self.obs_scale.cpu().numpy() + epsilon)

		mean, std = self.pi_mean_std(obs, on_policy)

		log_pi, entropy = None, None

		if deterministic:
			pre_pi = None
			with T.no_grad(): pi = T.tanh(mean)
		else:
			pre_pi, pi, log_pi, entropy = self.pi_prob(act,
                                               mean, std,
                                               on_policy,
                                               reparameterize,
                                               return_log_pi,
                                               return_entropy=True,
                                               return_pre_prob=return_pre_pi)
			pre_pi = (pre_pi * self.act_scale) + self.act_bias

		pi = (pi * self.act_scale) + self.act_bias

		return pre_pi, pi, log_pi, entropy


	def pi_mean_std(self, obs, on_policy=True):
		obs = T.as_tensor(obs, dtype=T.float32).to(self.device)

		net_out = self.mean_and_log_std_bb(obs)
		mean = self.mean(net_out)
		log_std = self.log_std(net_out)
		log_std = T.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		std = T.exp(log_std)
		self.std_value = std

		return mean, std


	def pi_prob(self, act,
                mean, std,
                on_policy,
                reparameterize,
                return_log_prob,
                return_entropy,
                return_pre_prob=False):

		normal_ditribution = Normal(mean, std)

		if act is None:
			if reparameterize:
				sample = normal_ditribution.rsample()
			else:
				sample = normal_ditribution.sample()
			pre_prob, prob = sample, T.tanh(sample)
		else:
			pre_prob, prob = act, T.tanh(act)

		log_prob, entropy = None, None

		if return_log_prob:
			log_prob = normal_ditribution.log_prob(pre_prob)
			log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(axis=-1, keepdim=True)

		if return_entropy:
			entropy = normal_ditribution.entropy().sum(axis=-1, keepdims=True)

		return pre_prob, prob, log_prob, entropy


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(SACPolicy, self).to(device)
