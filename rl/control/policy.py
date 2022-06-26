import random
import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

from rl.networks.mlp import MLPNet


LOG_STD_MAX = 2
LOG_STD_MIN = -20

# LOG_STD_MAX = 1.0
# LOG_STD_MIN = -2.5

epsilon = 1e-8

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


def init_weights_B(l, std=np.sqrt(2), bias=0.0): # init1
# def init_weights_(l, std=1.0, bias=0.0): # init2
	if isinstance(l, nn.Linear):
		# nn.init.xavier_uniform_(l.weight, std)
		# nn.init.uniform_(l.bias, bias)
		nn.init.orthogonal_(l.weight, std)
		nn.init.constant_(l.bias, bias)



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
		# hid1 = 64
		hid1 = 128
		# hid1 = 256

		# hid2 = 64
		hid2 = 128
		# hid2 = 256

		# self.mean = nn.Sequential(
		#     layer_init(nn.Linear(obs_dim, hid1)),
		#     nn.Tanh(),
		#     layer_init(nn.Linear(hid1, hid2)),
		#     nn.Tanh(),
		#     layer_init(nn.Linear(hid2, act_dim), std=0.01), # PPO-E: Major improvemet!
		# 	nn.Identity()
		# )
		# self.log_std = nn.Parameter(-0.5 * T.ones(act_dim, dtype=T.float32), requires_grad=False)

		self.mean = MLPNet(obs_dim, act_dim, net_configs)
		self.log_std = nn.Parameter(-0.5 * T.ones(act_dim, dtype=T.float32), requires_grad=False)
		# self.apply(init_weights_)

		# for param in list(self.parameters())[-2:]: param.data = 1e-2 * param.data
		# init_log_std = 0.
		# self.min_log_std = T.ones(act_dim)*(-2.5)
		# self.max_log_std = T.ones(act_dim)*(1.0)
		# self.log_std = nn.Parameter(T.ones(act_dim) * init_log_std, requires_grad=True)
		# self.log_std.data = T.max(self.log_std.data, self.min_log_std)
		# self.log_std.data = T.min(self.log_std.data, self.max_log_std)

		print('PPOPolicy: ', self)

		self.act_dim = act_dim

		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr, eps=1e-5) # PPO-E
		# self.optimizer = eval(optimizer)(self.parameters(), lr)


	def forward(self, obs, act=None,
                on_policy=True,
				reparameterize=False, # Default: True
				deterministic=False, # Default: False
				return_log_pi=True, # Default: False
				return_entropy=True, # Default: False
				):

		if isinstance(obs, T.Tensor):
			obs = obs.to(self.device)
		else:
			obs = obs

		mean = self.mean(obs)
		log_std = self.log_std
		# log_std = T.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		std = T.exp(log_std)

		probs = Normal(mean, std)

		log_probs = None
		entropy = None

		if act is None:
			act = probs.sample()
		if return_log_pi:
			log_probs = probs.log_prob(act).sum(axis=-1, keepdims=True)
		if return_entropy:
			entropy = probs.entropy().sum(-1, keepdims=True)

		if deterministic:
			act = mean

		return act, log_probs, entropy


	def forward_new(self, obs, act=None,
                on_policy=True,
				reparameterize=False, # Default: True
				deterministic=False, # Default: False
				return_log_pi=True, # Default: False
				return_entropy=True, # Default: False
				):

		if isinstance(obs, T.Tensor):
			obs = obs.to(self.device)
		else:
			obs = obs

		mean = self.mean(obs)
		log_std = self.log_std
		# log_std = T.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		std = T.exp(log_std)

		probs = Normal(mean, std)

		log_prob = None
		entropy = None

		if act is None:
			act = probs.sample()
		if return_log_pi:
			act_tanh = T.tanh(act)
			log_prob = probs.log_prob(act)
			log_prob -= T.log( self.act_scale * (1 - act_tanh.pow(2)) + epsilon )
			log_prob = log_prob.sum(-1, keepdim=True)
		if return_entropy:
			entropy = probs.entropy().sum(-1, keepdims=True)

		if deterministic:
			act = mean

		act = T.tanh(act)
		act = (act * self.act_scale) + self.act_bias

		return act, log_prob, entropy


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
		print('Initialize StochasticPolicy!')
		super(StochasticPolicy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		# hid1 = 64
		# hid1 = 128
		# hid1 = 256

		# hid2 = 64
		# hid2 = 128
		# hid2 = 256

		# My suggestions:
		self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		# self.apply(init_weights_)

		# self.mean_and_log_std_net = nn.Sequential(
		#     layer_init(nn.Linear(obs_dim, net_arch[0])),
		#     nn.Tanh(),
		#     layer_init(nn.Linear(net_arch[0], net_arch[1])),
		#     nn.Tanh()
		# )
		# self.mean = layer_init(nn.Linear(net_arch[-1], act_dim), std=0.01)
		# self.log_std = layer_init(nn.Linear(net_arch[-1], act_dim)) # Q: Best in MBPO
		# # self.log_std = layer_init(nn.Linear(net_arch[-1], act_dim), std=0.01) # R

		# self.mean_and_log_std_net = nn.Sequential(
		#     layer_init(nn.Linear(obs_dim, hid1)),
		#     nn.Tanh(),
		#     layer_init(nn.Linear(hid1, hid2)),
		#     nn.Tanh()
		# )
		# self.mean = layer_init(nn.Linear(hid2, act_dim), std=0.01)
		# self.log_std = layer_init(nn.Linear(hid2, act_dim))


		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)

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


	def pi_prob(self, mean, std, reparameterize, return_log_prob, return_entropy=True):
		normal_ditribution = Normal(mean, std)

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		prob, log_prob, entropy = T.tanh(sample), None, None

		if return_log_prob:
			log_prob = normal_ditribution.log_prob(sample)
			log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(-1, keepdim=True)
		if return_entropy:
			entropy = normal_ditribution.entropy().sum(-1, keepdims=True)

		return prob, log_prob, entropy


	def forward(self, obs, act=None,
			    on_policy=True,
				reparameterize=True, # Default: True
				deterministic=False, # Default: False
				return_log_pi=False, # Default: False
				return_entropy=True
				):

		if isinstance(obs, T.Tensor):
			obs = (obs.to(self.device) - self.obs_bias) / (self.obs_scale + epsilon)
		else:
			obs = (obs - self.obs_bias.cpu().numpy()) / (self.obs_scale.cpu().numpy() + epsilon)

		mean, std = self.pi_mean_std(T.as_tensor(obs, dtype=T.float32).to(self.device))

		log_pi = None
		entropy = None

		if deterministic: # Evaluation
			with T.no_grad(): pi = T.tanh(mean)
		else: # Stochastic | Interaction | Policy Evaluation/Improvement
			pi, log_pi, entropy = self.pi_prob(mean, std, reparameterize, return_log_pi, return_entropy=True)

		pi = (pi * self.act_scale) + self.act_bias

		return pi, log_pi, entropy


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(StochasticPolicy, self).to(device)






class OVOQPolicy(nn.Module):
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize OVOQ-Policy!')
		super(OVOQPolicy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']

		# We'll have a disturbed Initialization from log_std_q and log_std_v
		# hid = 128
		# self.mean_and_log_std_net = nn.Sequential(
		#     layer_init(nn.Linear(obs_dim, hid)),
		#     nn.Tanh(),
		#     layer_init(nn.Linear(hid, hid)),
		#     nn.Tanh()
		# )
		# self.mean = layer_init(nn.Linear(hid, act_dim), std=0.01)
		# self.log_std_q = layer_init(nn.Linear(hid, act_dim))
		# self.log_std_v = nn.Parameter(-0.5 * T.ones(act_dim, dtype=T.float32), requires_grad=False)


        # My suggestions:
		self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std_q = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		self.log_std_v = nn.Parameter(-0.5 * T.ones(act_dim, dtype=T.float32), requires_grad=False)
		# self.apply(init_weights_)


		self.act_dim = act_dim

		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr, eps=1e-5)


	def forward(self, obs, act=None,
                on_policy=True,
				reparameterize=False, # Default: True
				deterministic=False, # Default: False
				return_log_pi=True, # Default: False
				return_entropy=True, # Default: False
				):

		if isinstance(obs, T.Tensor):
			obs = (obs.to(self.device) - self.obs_bias) / (self.obs_scale + epsilon)
		else:
			obs = (obs - self.obs_bias.cpu().numpy()) / (self.obs_scale.cpu().numpy() + epsilon)

		mean, std = self.pi_mean_std(obs, on_policy)

		log_pi, entropy = None, None

		if deterministic:
			with T.no_grad():
				pi = mean if on_policy else T.tanh(mean)
		else:
			pi, log_pi, entropy = self.pi_prob(act, mean, std, on_policy, reparameterize, return_log_pi, return_entropy=True)

		if not on_policy:
			pi = (pi * self.act_scale) + self.act_bias

		return pi, log_pi, entropy


	def pi_mean_std(self, obs, on_policy=True):
		obs = T.as_tensor(obs, dtype=T.float32).to(self.device)

		net_out = self.mean_and_log_std_net(obs)
		mean = self.mean(net_out)

		if on_policy:
			log_std = self.log_std_v
		else:
			log_std = self.log_std_q(net_out)

		log_std = T.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		std = T.exp(log_std)
		return mean, std


	def pi_prob(self, act, mean, std, on_policy, reparameterize, return_log_prob, return_entropy):

		normal_ditribution = Normal(mean, std)

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		if on_policy:
			prob = sample
		else:
			prob = T.tanh(sample)

		log_prob, entropy = None, None

		if return_log_prob:
			if on_policy:
				if act is None: act = sample
				log_prob = normal_ditribution.log_prob(act).sum(axis=-1, keepdims=True)
			else:
				log_prob = normal_ditribution.log_prob(sample)
				log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
				log_prob = log_prob.sum(-1, keepdim=True)
		if return_entropy:
			entropy = normal_ditribution.entropy().sum(-1, keepdims=True)

		return prob, log_prob, entropy

	#
	# def to(self, device):
	# 	self.obs_bias = self.obs_bias.to(device)
	# 	self.obs_scale = self.obs_scale.to(device)
	# 	self.act_bias = self.act_bias.to(device)
	# 	self.act_scale = self.act_scale.to(device)
	# 	return super(OVOQPolicy, self).to(device)
