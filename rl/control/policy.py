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
		nn.init.trunc_normal_(l.weight, mean=0.0, std=0.1)
		nn.init.constant_(l.bias, 0.0)



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
		init_log_std = net_configs['init_log_std']

		self.mean = MLPNet(obs_dim, act_dim, net_configs)
		# self.log_std = nn.Parameter(-0.5 * T.ones(act_dim, dtype=T.float32), requires_grad=False) # org
		# self.log_std = nn.Parameter(0.5 * T.ones(act_dim, dtype=T.float32), requires_grad=False) # (MF/MB)-PPO
		# self.log_std = nn.Parameter(T.ones(act_dim, dtype=T.float32), requires_grad=False) # MBPPO-ReLU-21
		self.log_std = nn.Parameter(init_log_std * T.ones(act_dim, dtype=T.float32),
                                    requires_grad=net_configs['log_std_grad']) # (MF/MB)-PPO
		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_)

		self.std_value = T.tensor([0.])

		self.act_dim = act_dim

		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr)
		# self.optimizer = eval(optimizer)(self.parameters(), lr, eps=1e-5) # PPO-V

		print('PPO-Policy: ', self)
		print('PPO-Policy.log_std: ', self.log_std, '\n')


	def forward(self, obs, act=None,
                on_policy=True,
				reparameterize=False, # Default: True
				deterministic=False, # Default: False
				return_log_pi=True, # Default: False
				return_entropy=True, # Default: False
                return_pre_pi=False,
				):

		if isinstance(obs, T.Tensor):
			obs = obs.to(self.device)
		else:
			obs = obs

		mean = self.mean(obs)
		log_std = self.log_std
		# log_std = T.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		std = T.exp(log_std)
		self.std_value = std

		probs = Normal(mean, std)

		log_probs = None
		entropy = None

		# if act is None:
		# 	pre_act = probs.sample()
		# 	act = pre_act
		# 	# act = T.tanh(act)
		# 	# print(f'act={act} | tanh(act)={T.tanh(act)}')
		# else:
		# 	pre_act = act
		# 	# act = T.tanh(act)
		# if return_log_pi:
		# 	# log_probs = probs.log_prob(act).sum(axis=-1, keepdims=True)
		# 	log_probs = probs.log_prob(pre_act).sum(axis=-1, keepdims=True)
		# if return_entropy:
		# 	entropy = probs.entropy().sum(-1, keepdims=True)

		if deterministic:
			pi = mean
			pre_pi = pi
			# act = T.tanh(act)
			# print(f'act={act} | tanh(act)={T.tanh(act)}')
		else:
			# if act is None:
			pre_pi = probs.sample()
			pi = pre_pi
        	# act = T.tanh(act)
        	# print(f'act={act} | tanh(act)={T.tanh(act)}')
			if act is not None:
				pre_pi = act
				pi = pre_pi
            	# act = T.tanh(act)
            	# print(f'act={act} | tanh(act)={T.tanh(act)}')
			if return_log_pi:
            	# log_probs = probs.log_prob(pi).sum(axis=-1, keepdims=True)
				log_probs = probs.log_prob(pre_pi).sum(axis=-1, keepdims=True)
			if return_entropy:
				entropy = probs.entropy().sum(-1, keepdims=True)

		# return act, log_probs, entropy
		return pre_pi, pi, log_probs, entropy


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

		# My suggestions:
		self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_)

		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)

		self.act_dim = act_dim
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr)
		# self.optimizer = eval(optimizer)(self.parameters(), lr, eps=1e-5)

		print('SAC-Policy: ', self)


	def pi_mean_std(self, obs):
		net_out = self.mean_and_log_std_net(obs)
		mean = self.mean(net_out)
		log_std = self.log_std(net_out)
		log_std = T.tanh(log_std)
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
			    on_policy=False,
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
		# print(f'mean={mean}')

		log_pi, entropy = None, None

		if deterministic: # Evaluation
			with T.no_grad(): pi = T.tanh(mean)
		else: # Stochastic | Interaction | Policy Evaluation/Improvement
			pi, log_pi, entropy = self.pi_prob(mean, std,
                                    reparameterize=reparameterize,
                                    return_log_prob=return_log_pi,
                                    return_entropy=return_entropy)

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
		init_log_std = net_configs['init_log_std_v']

        # My suggestions:
		self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std_q = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		self.log_std_v = nn.Parameter(init_log_std * T.ones(act_dim, dtype=T.float32),
                                      requires_grad=net_configs['log_std_grad']) # (MF/MB)-PPO
		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_)


		self.act_dim = act_dim

		self.obs_bias   = T.zeros(obs_dim)
		self.obs_scale  = T.ones(obs_dim)
		self.act_bias =  T.FloatTensor( (act_up_lim + act_low_lim) / 2.0 )#.to(device)
		self.act_scale = T.FloatTensor( (act_up_lim - act_low_lim) / 2.0 )#.to(device)

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr)

		print('OVOQ-Policy: ', self)
		# print(f'Action | Bias={self.act_bias}, Scale={self.act_scale}')


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
			with T.no_grad(): pi = mean if on_policy else T.tanh(mean)
		else:
			pi, log_pi, entropy = self.pi_prob(act,
                                               mean, std,
                                               on_policy,
                                               reparameterize,
                                               return_log_pi,
                                               return_entropy=True)

		# if not on_policy:
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


	def pi_prob(self, act,
                mean, std,
                on_policy,
                reparameterize,
                return_log_prob,
                return_entropy):

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


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(OVOQPolicy, self).to(device)




class Policy(nn.Module):
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(Policy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		init_log_std = net_configs['init_log_std']

        # My suggestions:
		# self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		# self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		# self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std

		self.mean = MLPNet(obs_dim, act_dim, net_configs)
		# self.log_std = MLPNet(obs_dim, act_dim, net_configs)
		# self.log_std = nn.Linear(obs_dim, act_dim) # Last layer of Actor std
		# self.log_std = nn.Parameter(1. * T.ones(act_dim, dtype=T.float32),
        #                               requires_grad=False)
		# self.std = nn.Parameter(2.75 * T.ones(act_dim, dtype=T.float32),
        #                               requires_grad=net_configs['std_grad'])
		self.log_std = nn.Parameter(init_log_std * T.ones(act_dim, dtype=T.float32),
                                      requires_grad=net_configs['log_std_grad']) # (MF/MB)-PPO
		self.std_value = T.tensor([0.])

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_)

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

		# if isinstance(obs, T.Tensor):
		# 	obs = (obs.to(self.device) - self.obs_bias) / (self.obs_scale + epsilon)
		# else:
		# 	obs = (obs - self.obs_bias.cpu().numpy()) / (self.obs_scale.cpu().numpy() + epsilon)

		if isinstance(obs, T.Tensor):
			obs = obs.to(self.device)
		else:
			obs = obs

		mean, std = self.pi_mean_std(obs, on_policy)
		# print(f'mean={mean}')

		log_pi, entropy = None, None

		if deterministic:
			pre_pi = mean#None
			pi = mean
			# with T.no_grad(): pi = mean
			# with T.no_grad(): pi = T.tanh(mean)
		else:
			pre_pi, pi, log_pi, entropy = self.pi_prob(act,
                                               mean, std,
                                               on_policy,
                                               reparameterize,
                                               return_log_pi,
                                               return_entropy=True,
                                               return_pre_prob=return_pre_pi)
			# pre_pi = (pre_pi * self.act_scale) + self.act_bias

		# pi = (pi * self.act_scale) + self.act_bias

		return pre_pi, pi, log_pi, entropy


	def pi_mean_std(self, obs, on_policy=True):
		obs = T.as_tensor(obs, dtype=T.float32).to(self.device)

		# net_out = self.mean_and_log_std_net(obs)
		# mean = self.mean(net_out)
		# log_std = self.log_std(net_out)
		# log_std = T.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		# std = T.exp(log_std)

		mean = self.mean(obs)
		log_std = self.log_std
		std = T.exp(log_std)

		# mean = self.mean(obs)
		# # if not on_policy:
		# # 	self.std.requires_grad = False
		# # else:
		# # 	self.std.requires_grad = True
		# std = self.std

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

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		pre_prob, prob = sample, sample
		# pre_prob, prob = sample, T.tanh(sample)

		log_prob, entropy = None, None

		if return_log_prob:
			if act is not None: pre_prob, prob = act, act
			# if act is not None: pre_prob, prob = act, T.tanh(act)
			log_prob = normal_ditribution.log_prob(pre_prob)
			# log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(axis=-1, keepdim=True)

		if return_entropy:
			entropy = normal_ditribution.entropy().sum(axis=-1, keepdims=True)

		return pre_prob, prob, log_prob, entropy


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(Policy, self).to(device)



class PolicyB(nn.Module): # B
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(Policy, self).__init__() # To automatically use 'def forward'

		self.device = device

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		# init_log_std = net_configs['init_log_std']

        # My suggestions:
		self.mean_and_log_std_bb = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.log_std_a = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		# self.log_std.weight_decay = 0.00000
		# self.std_b = nn.Parameter(2.75 * T.ones(act_dim, dtype=T.float32),
        #                               requires_grad=net_configs['std_grad'])
		self.std_b = nn.Parameter(.5 * T.ones(act_dim, dtype=T.float32),
                                      requires_grad=net_configs['std_grad'])
		self.std = T.tensor([0.])

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_)

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
		# print(f'mean={mean}')
		# print(f'std={std}')

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
		log_std_a = self.log_std_a(net_out)
		log_std_a = T.clamp(log_std_a, min=LOG_STD_MIN, max=LOG_STD_MAX)
		std_a = T.exp(log_std_a)
		std = 0.25*std_a + 0.75*self.std_b
		self.std = std

		return mean, std


	def pi_prob(self, act,
                mean, std,
                on_policy,
                reparameterize,
                return_log_prob,
                return_entropy,
                return_pre_prob=False):

		normal_ditribution = Normal(mean, std)

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		pre_prob, prob = sample, T.tanh(sample)

		log_prob, entropy = None, None

		if return_log_prob:
			if act is not None: pre_prob, prob = act, T.tanh(act)
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
		return super(Policy, self).to(device)



class PolicyC(nn.Module): # B
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(Policy, self).__init__() # To automatically use 'def forward'

		self.device = device
		self.net_configs = net_configs

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		# init_log_std = net_configs['init_log_std']

        # My suggestions:
		self.mean_and_std_bb = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_iii)

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
		# print(f'mean={mean}')
		# print(f'std={std}')

		log_pi, entropy = None, None

		if deterministic:
			pre_pi = None
			with T.no_grad(): pi = mean
			# with T.no_grad(): pi = T.tanh(mean)
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

		net_out = self.mean_and_std_bb(obs)
		mean = self.mean(net_out)
		std = self.std(net_out)
		std = F.softplus(std)
		std = std * self.net_configs['init_std'] / F.softplus(T.tensor([0.]))
		std = std + T.tensor([self.net_configs['min_std']])
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

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		pre_prob, prob = sample, sample
		# pre_prob, prob = sample, T.tanh(sample)

		log_prob, entropy = None, None

		if return_log_prob:
			if act is not None: pre_prob, prob = act, act
			# if act is not None: pre_prob, prob = act, T.tanh(act)
			log_prob = normal_ditribution.log_prob(pre_prob)
			# log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(axis=-1, keepdim=True)

		if return_entropy:
			entropy = normal_ditribution.entropy().sum(axis=-1, keepdims=True)

		return pre_prob, prob, log_prob, entropy


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(Policy, self).to(device)



class PolicyD(nn.Module): # D
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(Policy, self).__init__() # To automatically use 'def forward'

		self.device = device
		self.net_configs = net_configs

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		# init_log_std = net_configs['init_log_std']

        # My suggestions:
		self.mean_and_std_bb = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		self.std_value = T.tensor([0.])

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_iii)

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
		# print(f'mean={mean}')
		# print(f'std={std}')

		log_pi, entropy = None, None

		if deterministic:
			pre_pi = None
			# with T.no_grad(): pi = mean
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

		net_out = self.mean_and_std_bb(obs)
		mean = self.mean(net_out)
		std = self.std(net_out)
		std = F.softplus(std)
		std = std * self.net_configs['init_std'] / F.softplus(T.tensor([0.]))
		std = std + T.tensor([self.net_configs['min_std']])
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

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		# pre_prob, prob = sample, sample
		pre_prob, prob = sample, T.tanh(sample)

		log_prob, entropy = None, None

		if return_log_prob:
			# if act is not None: pre_prob, prob = act, act
			if act is not None: pre_prob, prob = act, T.tanh(act)
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
		return super(Policy, self).to(device)




class PolicyE(nn.Module): # E
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(Policy, self).__init__() # To automatically use 'def forward'

		self.device = device
		self.net_configs = net_configs

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		# init_log_std = net_configs['init_log_std']

        # My suggestions:
		self.mean_and_std_bb = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		# self.log_std_a = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		# self.std_a = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		self.std_b = nn.Parameter(2.75 * T.ones(act_dim, dtype=T.float32),
                                  requires_grad=net_configs['std_grad'])
		self.std_value = T.tensor([0.])

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_)
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
		# print(f'mean={mean}')
		# print(f'std={std}')

		log_pi, entropy = None, None

		if deterministic:
			pre_pi = None
			with T.no_grad(): pi = mean
			# with T.no_grad(): pi = T.tanh(mean)
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

		net_out = self.mean_and_std_bb(obs)
		mean = self.mean(net_out)
		# std_a = self.std_a(net_out)
		# std_a = T.clamp(std_a, min=1e-6, max=7)
		# log_std_a = self.log_std_a(net_out)
		# log_std_a = T.clamp(log_std_a, min=LOG_STD_MIN, max=LOG_STD_MAX)
		# std_a = T.exp(log_std_a)
		std_b = self.std_b
		# std = std_a
		std = std_b
		# std = 0.3*std_a + 0.7*std_b
		# std = 0.5*std_a + 0.5*std_b
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

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		pre_prob, prob = sample, sample
		# pre_prob, prob = sample, T.tanh(sample)

		log_prob, entropy = None, None

		if return_log_prob:
			if act is not None: pre_prob, prob = act, act
			# if act is not None: pre_prob, prob = act, T.tanh(act)
			log_prob = normal_ditribution.log_prob(pre_prob)
			# log_prob -= T.log( self.act_scale * (1 - prob.pow(2)) + epsilon )
			log_prob = log_prob.sum(axis=-1, keepdim=True)

		if return_entropy:
			entropy = normal_ditribution.entropy().sum(axis=-1, keepdims=True)

		return pre_prob, prob, log_prob, entropy


	def to(self, device):
		self.obs_bias = self.obs_bias.to(device)
		self.obs_scale = self.obs_scale.to(device)
		self.act_bias = self.act_bias.to(device)
		self.act_scale = self.act_scale.to(device)
		return super(Policy, self).to(device)





class PolicyF(nn.Module): # F
	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('Initialize Policy!')
		super(Policy, self).__init__() # To automatically use 'def forward'

		self.device = device
		self.net_configs = net_configs

		net_arch = net_configs['arch']
		optimizer = 'T.optim.' + net_configs['optimizer']
		lr = net_configs['lr']
		# init_log_std = net_configs['init_log_std']

        # My suggestions:
		self.mean_and_std_bb = MLPNet(obs_dim, 0, net_configs)
		self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		self.std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		self.std_value = T.tensor([0.])

		if net_configs['initialize_weights']:
			print('Apply Initialization')
			self.apply(init_weights_iii)

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
		# print(f'mean={mean}')
		# print(f'std={std}')

		log_pi, entropy = None, None

		if deterministic:
			pre_pi = None
			# with T.no_grad(): pi = mean
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

		net_out = self.mean_and_std_bb(obs)
		mean = self.mean(net_out)
		std = self.std(net_out)
		std = T.clamp(std, min=2, max=3)
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

		if reparameterize:
			sample = normal_ditribution.rsample()
		else:
			sample = normal_ditribution.sample()

		# pre_prob, prob = sample, sample
		pre_prob, prob = sample, T.tanh(sample)

		log_prob, entropy = None, None

		if return_log_prob:
			# if act is not None: pre_prob, prob = act, act
			if act is not None: pre_prob, prob = act, T.tanh(act)
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
		return super(Policy, self).to(device)




# squash(regularized(logstd(x)) + log_std_constant)
