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





class StochasticPolicy(nn.Module):

	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		# print('init Policy!')
		# if seed: random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
		super(StochasticPolicy, self).__init__() # To automatically use 'def forward'

		self._device_ = device
		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']

		# My suggestions:
		self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		self.mu = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std

		self.apply(init_weights_)




		# self.obs_bias   = np.zeros(obs_dim)
		# self.obs_scale  = np.ones(obs_dim)

		self.obs_bias   = T.zeros(obs_dim)#.to(device)
		self.obs_scale  = T.ones(obs_dim)#.to(device)

		self.act_dim = act_dim
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		# Define optimizer
		self.optimizer = eval(optimizer)(self.parameters(), lr)


	def pi_mean_std(self, obs):
		net_out = self.mean_and_log_std_net(obs)
		mean = self.mu(net_out)
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

		prob, log_prob = T.tanh(sample), None

		if return_log_prob:
			log_prob = normal_ditribution.log_prob(sample)
			log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(1, keepdim=True)

		return prob, log_prob


	def forward(self,
				obs,
				reparameterize=True, # Default: True
				deterministic=False, # Default: False
				return_log_pi=False # Default: False
				):

		if isinstance(obs, T.Tensor):
			obs = (obs - self.obs_bias) / (self.obs_scale + epsilon)
		else:
			obs = (obs - self.obs_bias.cpu().numpy()) / (self.obs_scale.cpu().numpy() + epsilon)

		mean, std = self.pi_mean_std(
		T.as_tensor(obs, dtype=T.float32).to(self._device_)
		)

		log_pi = None

		if deterministic: # Evaluation
			with T.no_grad(): pi = T.tanh(mean)

		else: # Stochastic | Interaction
			pi, log_pi = self.pi_prob(mean, std, reparameterize, return_log_pi)

		pi = (pi * self.act_scale) + self.act_bias

		return pi, log_pi


	def step_np(self,
				obs,
				reparameterize=True, # Default: True
				deterministic=False, # Default: False
				return_log_pi=False # Default: False
				):
		pi, log_pi = self.forward(obs, reparameterize, deterministic, return_log_pi)
		pi = pi.detach().cpu().numpy()
		return pi, log_pi


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(StochasticPolicy, self).to(device)
