import random
import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

from rl.networks.mlp import MLPNet
from rl.control.distributions import TanhNormal


LOG_STD_MAX = 2
LOG_STD_MIN = -20

art_zero = 1e-8



class StochasticPolicy(nn.Module):

	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		# print('init Policy!')
		# if seed: random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

		self.device = device
		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']

		super().__init__() # To automatically use 'def forward'

		# My suggestions:
		self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		self.mu = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std

		# Define optimizer
		self.optimizer = eval(optimizer)(self.parameters(), lr)

		self.obs_bias   = np.zeros(obs_dim)
		self.obs_scale  = np.ones(obs_dim)

		self.act_dim = act_dim
		self.action_scale = T.FloatTensor( 0.5 * (act_up_lim - act_low_lim) ).to(device)
		self.action_bias =  T.FloatTensor( 0.5 * (act_up_lim + act_low_lim) ).to(device)


	def get_act_dist_params(self, obs):
		# print('get_act_dist_params')
		# print(f'obs={obs}')
		net_out = self.mean_and_log_std_net(obs)
		# print(f'net_out={net_out}')
		mean = self.mu(net_out)
		# print(f'mean={mean}')
		log_std = self.log_std(net_out)
		log_std = T.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
		# print(f'Policy log_std={log_std}')
		std = T.exp(log_std)
		# print(f'std={std}')
		return mean, std


	def prob(self, mean, std, reparameterize):
		pre_tanh_value = None
		# print(f'Policy prob: \nmean={mean}, \nstd={std}')
		tanh_normal = TanhNormal(mean, std, device=self.device)
		if reparameterize:
			pi, pre_tanh_value = tanh_normal.rsample()
		else:
			pi, pre_tanh_value = tanh_normal.sample()
		# print('pi b4: ', T.mean(pi, dim=0))
		pi = (pi * self.action_scale) + self.action_bias
		# print('pi af: ', T.mean(pi, dim=0))
		return pi, pre_tanh_value


	def logprob(self, pi, mean, std, pre_tanh_value):
		tanh_normal = TanhNormal(mean, std, device=self.device)
		log_pi = tanh_normal.log_prob(pi, pre_tanh_value=pre_tanh_value)
		return log_pi.view(-1,1)


	def deterministic(self, mean):
		with T.no_grad():
			pi = (T.tanh(mean) * self.action_scale) + self.action_bias
		return pi


	def forward(self,
				obs,
				reparameterize=True, # Default: True
				deterministic=False, # Default: False
				return_log_pi=False # Default: False
				):

		obs = (obs - self.obs_bias) / (self.obs_scale + art_zero)

		mean, std = self.get_act_dist_params(T.as_tensor(obs, dtype=T.float32).to(self.device))
		# print('STD: ', std)
		log_pi = None
		if deterministic:
			pi = self.deterministic(mean)
		else: # stochastic
			if return_log_pi:
				pi, pre_tanh_value = self.prob(mean, std, reparameterize)
				log_pi = self.logprob(pi, mean, std, pre_tanh_value)
				# print('LogPi:	', log_pi.shape)
			else:
				pi, pre_tanh_value = self.prob(mean, std, reparameterize)
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
