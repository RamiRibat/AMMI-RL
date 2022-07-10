import random
import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

from rl.networks.mlp import MLPNet
# from rl.control.distributions import TanhNormal


LOG_STD_MAX = 2
LOG_STD_MIN = -20

epsilon = 1e-8





def init_weights_(l):
	if isinstance(l, nn.Linear):
		nn.init.xavier_uniform_(l.weight, 1.0)
		nn.init.uniform_(l.bias, 0.0)


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     T.nn.init.orthogonal_(layer.weight, std)
#     T.nn.init.constant_(layer.bias, bias_const)
#     return layer


class PPOPolicy(nn.Module):

	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('init PPOPolicy!')
		super(PPOPolicy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']

		# My suggestions:
		# self.mean = MLPNet(obs_dim, act_dim, net_configs)
		# self.log_std = nn.Parameter(T.zeros(1, act_dim))
		# self.log_std = nn.Parameter(T.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32)))

		# self.apply(init_weights_)

		self.mean = nn.Sequential(
		    layer_init(nn.Linear(obs_dim, 64)),
		    nn.Tanh(),
		    layer_init(nn.Linear(64, 64)),
		    nn.Tanh(),
		    layer_init(nn.Linear(64, act_dim), std=0.01),
		)
		self.log_std = nn.Parameter(T.zeros(1, act_dim))

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr, eps=1e-5)


	def forward(self, obs, act=None,
				deterministic=False, # Default: False
				return_log_pi=True, # Default: False
				return_entropy=True, # Default: False
				):

		mean = self.mean(obs)
		std = T.exp(self.log_std)
		probs = Normal(mean, std)
		log_probs = None
		entropy = None

		if act is None: act = probs.sample()
		if return_log_pi: log_probs = probs.log_prob(act).sum(1, keepdims=True)
		if return_entropy: entropy = probs.entropy().sum(1, keepdims=True)
		return act, log_probs, entropy


	# def to(self, device):
	# 	self.obs_bias = self.obs_bias.to(device)
	# 	self.obs_scale = self.obs_scale.to(device)
	# 	self.act_bias = self.act_bias.to(device)
	# 	self.act_scale = self.act_scale.to(device)
	# 	return super(PPOPolicy, self).to(device)





class StochasticPolicy(nn.Module):

	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		# print('init Policy!')
		super(StochasticPolicy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']

		# My suggestions:
		self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std

		self.apply(init_weights_)

		self.obs_bias   = T.zeros(obs_dim)#.to(device)
		self.obs_scale  = T.ones(obs_dim)#.to(device)

		self.act_dim = act_dim
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr)


	def pi_mean_std(self, obs):
		net_out = self.mean_and_log_std_net(obs)
		mean = self.mean(net_out)
		log_std = self.log_std(net_out)
		log_std = T.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		std = T.exp(log_std)
		return mean, std


	def pi_prob(self, mean, std, reparameterize, return_log_prob):
		normal_ditribution = Normal(mean, std)

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		prob, log_prob, entropy = T.tanh(sample), None, None

		if return_log_prob:
			log_prob = normal_ditribution.log_prob(sample)
			log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(1, keepdim=True)

		return prob, log_prob, entropy


	def forward(self,
				obs, act=None,
				reparameterize=True, # Default: True
				deterministic=False, # Default: False
				return_log_pi=False # Default: False
				):
		# print('forward.reparameterize: ', reparameterize)

		if isinstance(obs, T.Tensor):
			obs = (obs.to(self.device) - self.obs_bias) / (self.obs_scale + epsilon)
		else:
			obs = (obs - self.obs_bias.cpu().numpy()) / (self.obs_scale.cpu().numpy() + epsilon)

		mean, std = self.pi_mean_std(
		T.as_tensor(obs, dtype=T.float32).to(self.device)
		)

		log_pi, entropy = None, None

		if deterministic: # Evaluation
			with T.no_grad(): pi = T.tanh(mean)

		else: # Stochastic | Interaction
			# print('reparameterize: ', reparameterize)
			pi, log_pi, entropy = self.pi_prob(mean, std, reparameterize, return_log_pi)

		pi = (pi * self.act_scale) + self.act_bias

		return pi, log_pi, entropy


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(StochasticPolicy, self).to(device)
