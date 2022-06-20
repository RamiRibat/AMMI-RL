import random
import numpy as np
import torch as T
# from torch.autograd import Variable
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer




class StochasticPolicy(nn.Module):

	def __init__(self, obs_dim, act_dim,
				act_up_lim, act_low_lim,
				net_configs, device, seed) -> None:
		print('init StochasticPolicy!')
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
		# if return_entropy:
		# 	entropy = normal_ditribution.entropy().sum(-1, keepdims=True)

		return prob, log_prob, entropy


	def forward(self,
				obs, act=None,
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






LOG_STD_MAX = 2
LOG_STD_MIN = -20



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
		# hid = 64
		hid = 256 # PPO-G
		# hid = 128 # PPO-H

		# My suggestions:
		# self.mean = MLPNet(obs_dim, act_dim, net_configs)
		# self.log_std = nn.Parameter(T.zeros(1, act_dim))
		# self.log_std = nn.Parameter(-0.5 * T.ones(act_dim, dtype=T.float32))
		# self.apply(init_weights_)

		# self.mean = nn.Sequential(
		#     layer_init(nn.Linear(obs_dim, 64)),
		#     nn.Tanh(),
		#     layer_init(nn.Linear(64, 64)),
		#     nn.Tanh(),
		#     layer_init(nn.Linear(64, act_dim), std=0.01),
		# )
		# self.log_std = nn.Parameter(T.zeros(1, act_dim))

		self.mean = nn.Sequential(
		    layer_init(nn.Linear(obs_dim, hid)),
		    nn.Tanh(),
		    layer_init(nn.Linear(hid, hid)),
		    nn.Tanh(),
		    layer_init(nn.Linear(hid, act_dim), std=0.01), # PPO-E: Major improvemet!
			nn.Identity()
		)
		# self.log_std = nn.Parameter(T.ones(act_dim, dtype=T.float32))
		self.log_std = nn.Parameter(-0.5 * T.ones(act_dim, dtype=T.float32), requires_grad=False)
		# self.log_std = nn.Parameter(T.zeros(act_dim, dtype=T.float32), requires_grad=True)



		# self.mean_and_log_std_net = MLPNet(obs_dim, 0, net_configs)
		# self.mean = nn.Linear(net_arch[-1], act_dim) # Last layer of Actoe mean
		# self.log_std = nn.Linear(net_arch[-1], act_dim) # Last layer of Actor std
		#
		# self.apply(init_weights_)

		# for param in list(self.parameters())[-2:]: param.data = 1e-2 * param.data
		# init_log_std = 0.
		# self.min_log_std = T.ones(act_dim)*(-2.5)
		# self.max_log_std = T.ones(act_dim)*(1.0)
		# self.log_std = nn.Parameter(T.ones(act_dim) * init_log_std, requires_grad=True)
		# self.log_std.data = T.max(self.log_std.data, self.min_log_std)
		# self.log_std.data = T.min(self.log_std.data, self.max_log_std)

		# print('PPOPolicy: ', self)

		self.act_dim = act_dim

		self.to(device)

		self.optimizer = eval(optimizer)(self.parameters(), lr, eps=1e-5) # PPO-E
		# self.optimizer = eval(optimizer)(self.parameters(), lr)


	def forward(self, obs, act=None,
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

		log_probs = None
		entropy = None

		if deterministic:
			# print('deter')
			act = mean
			# log_probs = self.get_log_probs(act, mean, log_std, std)
		else:
			act = mean + std * T.randn(self.act_dim)
			# print(f'Pi: mean={mean} | std={std}, | a={act}')
			log_probs = self.get_log_probs(act, mean, log_std, std)
			probs = Normal(mean, std)
			entropy = probs.entropy().sum(-1, keepdims=True)

		return act, log_probs, entropy


	def get_log_probs(self, act, mean, log_std, std):
		zs = (act - mean) / std
		log_probs = - 0.5 * (zs ** 2).sum(axis=-1, keepdims=True) - log_std.sum(axis=-1, keepdims=True) - 0.5 * self.act_dim * np.log(2 * np.pi)
		return log_probs



	def kl_old_new(self, obs, old_mean, old_log_std):
		new_mean = self.mean(obs)
		new_log_std = self.log_std
		kl_divergence = self.kl_divergence(new_mean, old_mean, new_log_std, old_log_std)
		return kl_divergence


	def mean_kl(self, obs):
		new_log_std = self.log_std
		old_log_std = self.log_std.detach().clone()
		new_mean = self.mean(obs)
		old_mean = new_mean.detach()
		return self.kl_divergence(new_mean, old_mean, new_log_std, old_log_std)


	def kl_divergence(self, new_mean, old_mean, new_log_std, old_log_std):
		new_std, old_std = T.exp(new_log_std), T.exp(old_log_std)
		Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
		Dr = 2 * new_std ** 2 + 1e-8
		sample_kl = (Nr / Dr + new_log_std - old_log_std).sum(axis=-1, keepdims=True)
		return T.mean(sample_kl)


	# def to(self, device):
	# 	self.obs_bias = self.obs_bias.to(device)
	# 	self.obs_scale = self.obs_scale.to(device)
	# 	self.act_bias = self.act_bias.to(device)
	# 	self.act_scale = self.act_scale.to(device)
	# 	return super(PPOPolicy, self).to(device)




import torch


LOG_STD_MAX = 1.0
LOG_STD_MIN = -2.5

epsilon = 1e-8


def tensorize(var, device='cpu'):
    """
    Convert input to torch.Tensor on desired device
    :param var: type either torch.Tensor or np.ndarray
    :param device: desired device for output (e.g. cpu, cuda)
    :return: torch.Tensor mapped to the device
    """
    if type(var) == torch.Tensor:
        return var.to(device)
    elif type(var) == np.ndarray:
        return torch.from_numpy(var).float().to(device)
    elif type(var) == float:
        return torch.tensor(var).float()
    else:
        print("Variable type not compatible with function.")
        return None



class NPGPolicy(T.nn.Module): # Source: github.com/aravindr93/mjrl
    def __init__(self,
				 # env_spec=None,
				 obs_dim, act_dim,
			 	 act_up_lim, act_low_lim,
			 	 net_configs,
				 device, seed,
                 # hidden_sizes=(256,256),
                 hidden_sizes=(64,64),
                 init_log_std=0.0,
                 # *args, **kwargs,
                 ):
        super(NPGPolicy, self).__init__()
        print('init NPG Policy')
        print('hidden_sizes: ', hidden_sizes)
        # check input specification
        # if env_spec is None:
        #     assert observation_dim is not None
        #     assert action_dim is not None
        # T.manual_seed(1)
        # np.random.seed(1)

        self.device = device
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.min_log_std_val = LOG_STD_MIN if type(LOG_STD_MIN)==np.ndarray else LOG_STD_MIN * np.ones(self.act_dim)
        self.max_log_std_val = LOG_STD_MAX if type(LOG_STD_MAX)==np.ndarray else LOG_STD_MAX * np.ones(self.act_dim)
        self.min_log_std = tensorize(self.min_log_std_val)
        self.max_log_std = tensorize(self.max_log_std_val)

        # Policy network
        # ------------------------
        self.layer_sizes = (self.obs_dim, ) + hidden_sizes + (self.act_dim, )
        self.nonlinearity = T.tanh
        self.fc_layers = T.nn.ModuleList([T.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                             for i in range(len(self.layer_sizes)-1)])
        for param in list(self.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = T.nn.Parameter(T.ones(self.act_dim) * init_log_std, requires_grad=True)
        self.log_std.data = T.max(self.log_std.data, self.min_log_std)
        self.log_std.data = T.min(self.log_std.data, self.max_log_std)
        self.trainable_params = list(self.parameters())

        # transform variables
        self.in_shift, self.in_scale = T.zeros(self.obs_dim), T.ones(self.obs_dim)
        self.out_shift, self.out_scale = T.zeros(self.act_dim), T.ones(self.act_dim)

        # Easy access variables
        # -------------------------
        self.log_std_val = self.log_std.to('cpu').data.numpy().ravel()
        # clamp log_std to [min_log_std, max_log_std]
        self.log_std_val = np.clip(self.log_std_val, self.min_log_std_val, self.max_log_std_val)
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = T.zeros(self.obs_dim)

        # Move parameters to device
        # ------------------------
        self.to(device)


    # Network forward
    # ============================================
    def forward(self, obs):
        if type(obs) == np.ndarray: obs = T.from_numpy(obs).float()
        assert type(obs) == T.Tensor

        obs = obs.to(self.device)
        out = (obs - self.in_shift) / (self.in_scale + 1e-6)

        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)

        out = self.fc_layers[-1](out) * self.out_scale + self.out_shift
        return out



    # Utility functions
    # ============================================
    def to(self, device):
        super().to(device)
        self.min_log_std, self.max_log_std = self.min_log_std.to(device), self.max_log_std.to(device)
        self.in_shift, self.in_scale = self.in_shift.to(device), self.in_scale.to(device)
        self.out_shift, self.out_scale = self.out_shift.to(device), self.out_scale.to(device)
        self.trainable_params = list(self.parameters())
        self.device = device


    def get_param_values(self,
						# *args, **kwargs
						):
        params = torch.cat([p.contiguous().view(-1).data for p in self.parameters()])
        return params.clone()


    def set_param_values(self, new_params,
						# *args, **kwargs
						):
        current_idx = 0
        for idx, param in enumerate(self.parameters()):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            # clip std at minimum value
            vals = torch.max(vals, self.min_log_std) if idx == 0 else vals
            vals = torch.min(vals, self.max_log_std) if idx == 0 else vals
            param.data = vals.to(self.device).clone()
            current_idx += self.param_sizes[idx]
        # update log_std_val for sampling
        self.log_std_val = np.float64(self.log_std.to('cpu').data.numpy().ravel())
        self.log_std_val = np.clip(self.log_std_val, self.min_log_std_val, self.max_log_std_val)
        self.trainable_params = list(self.parameters())


    def set_transformations(self, in_shift=None, in_scale=None,
                            out_shift=None, out_scale=None,
							# *args, **kwargs
							):
        in_shift = self.in_shift if in_shift is None else tensorize(in_shift)
        in_scale = self.in_scale if in_scale is None else tensorize(in_scale)
        out_shift = self.out_shift if out_shift is None else tensorize(out_shift)
        out_scale = self.out_scale if out_scale is None else tensorize(out_scale)
        self.in_shift, self.in_scale = in_shift.to(self.device), in_scale.to(self.device)
        self.out_shift, self.out_scale = out_shift.to(self.device), out_scale.to(self.device)


    # Main functions
    # ============================================
    def get_action(self, obs): # Real Interaction/Evaluation
        assert type(obs) == np.ndarray
        if self.device != 'cpu':
            print("Warning: get_action function should be used only for simulation.")
            print("Requires policy on CPU. Changing policy device to CPU.")
            self.to('cpu')
        o = np.float32(obs.reshape(1, -1))
        self.obs_var.data = T.from_numpy(o)
        mean = self.forward(self.obs_var).to('cpu').data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.act_dim)
        act = mean + noise
        return [act, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]


    def mean_LL(self, obs, act, log_std=None,
				# *args, **kwargs
				):
        if type(obs) == np.ndarray: obs = T.from_numpy(obs).float()
        if type(act) == np.ndarray: act = T.from_numpy(act).float()
        obs, act = obs.to(self.device), act.to(self.device)
        log_std = self.log_std if log_std is None else log_std
        mean = self.forward(obs)
        zs = (act - mean) / T.exp(self.log_std)
        # print('zs: ', zs)
        # print('log_std: ', log_std)
        # print('self.act_dim: ', self.act_dim)
        LL = - 0.5 * T.sum(zs ** 2) - T.sum(log_std) - 0.5 * self.act_dim * np.log(2 * np.pi)
        return mean, LL


    def log_likelihood(self, obs, act,
					   # *args, **kwargs,
					   ):
        mean, LL = self.mean_LL(obs, act)
        return LL.to('cpu').data.numpy()


    def mean_kl(self, obs,
				# *args, **kwargs
				):
        new_log_std = self.log_std
        old_log_std = self.log_std.detach().clone()
        new_mean = self.forward(obs)
        old_mean = new_mean.detach()
        return self.kl_divergence(new_mean, old_mean,
								  new_log_std, old_log_std,
								  # *args, **kwargs
								  )


    def kl_divergence(self,
					  new_mean, old_mean,
					  new_log_std, old_log_std,
						# *args, **kwargs
						):
        new_std, old_std = T.exp(new_log_std), T.exp(old_log_std)
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = T.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return T.mean(sample_kl)





#
#
# class MLPPolicy(torch.nn.Module): # Source: github.com/aravindr93/mjrl
#     def __init__(self,
# 				 # env_spec=None,
# 				 obs_dim, act_dim,
# 			 	 act_up_lim, act_low_lim,
# 			 	 net_configs,
# 				 device, seed,
#                  hidden_sizes=(64,64),
#                  init_log_std=0.0,
#                  # min_log_std=-3.0,
#                  # max_log_std=1.0,
#                  # seed=123,
#                  # device='cpu',
#                  # observation_dim=None,
#                  # action_dim=None,
#                  # *args, **kwargs,
#                  ):
#         super(MLP, self).__init__()
#         # check input specification
#         if env_spec is None:
#             assert observation_dim is not None
#             assert action_dim is not None
#
#         # self.obs_dim = env_spec.observation_dim if env_spec is not None else observation_dim   # number of states
#         # self.act_dim = env_spec.action_dim if env_spec is not None else action_dim                  # number of actions
#         self.device = device
#         # self.seed = seed
#
#         self.min_log_std_val = LOG_STD_MIN if type(LOG_STD_MIN)==np.ndarray else LOG_STD_MIN * np.ones(self.act_dim)
#         self.max_log_std_val = LOG_STD_MAX if type(LOG_STD_MAX)==np.ndarray else LOG_STD_MAX * np.ones(self.act_dim)
#         self.min_log_std = tensorize(self.min_log_std_val)
#         self.max_log_std = tensorize(self.max_log_std_val)
#
#         # # Set seed
#         # # ------------------------
#         # assert type(seed) == int
#         # torch.manual_seed(seed)
#         # np.random.seed(seed)
#
#         # Policy network
#         # ------------------------
#         self.layer_sizes = (self.obs_dim, ) + hidden_sizes + (self.act_dim, )
#         self.nonlinearity = torch.tanh
#         self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
#                                              for i in range(len(self.layer_sizes)-1)])
#         for param in list(self.parameters())[-2:]:  # only last layer
#            param.data = 1e-2 * param.data
#         self.log_std = torch.nn.Parameter(torch.ones(self.act_dim) * init_log_std, requires_grad=True)
#         self.log_std.data = torch.max(self.log_std.data, self.min_log_std)
#         self.log_std.data = torch.min(self.log_std.data, self.max_log_std)
#         self.trainable_params = list(self.parameters())
#
#         # transform variables
#         self.in_shift, self.in_scale = torch.zeros(self.obs_dim), torch.ones(self.obs_dim)
#         self.out_shift, self.out_scale = torch.zeros(self.act_dim), torch.ones(self.act_dim)
#
#         # Easy access variables
#         # -------------------------
#         self.log_std_val = self.log_std.to('cpu').data.numpy().ravel()
#         # clamp log_std to [min_log_std, max_log_std]
#         self.log_std_val = np.clip(self.log_std_val, self.min_log_std_val, self.max_log_std_val)
#         self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
#         self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
#         self.d = np.sum(self.param_sizes)  # total number of params
#
#         # Placeholders
#         # ------------------------
#         self.obs_var = torch.zeros(self.obs_dim)
#
#         # Move parameters to device
#         # ------------------------
#         self.to(device)
#
#
#     # Network forward
#     # ============================================
#     def forward(self, obs):
#         if type(obs) == np.ndarray: obs = torch.from_numpy(obs).float()
#         assert type(obs) == torch.Tensor
#
#         obs = obs.to(self.device)
#         out = (obs - self.in_shift) / (self.in_scale + 1e-6)
#
#         for i in range(len(self.fc_layers)-1):
#             out = self.fc_layers[i](out)
#             out = self.nonlinearity(out)
#
#         out = self.fc_layers[-1](out) * self.out_scale + self.out_shift
#         return out
#
#
#     # Utility functions
#     # ============================================
#     def to(self, device):
#         super().to(device)
#         self.min_log_std, self.max_log_std = self.min_log_std.to(device), self.max_log_std.to(device)
#         self.in_shift, self.in_scale = self.in_shift.to(device), self.in_scale.to(device)
#         self.out_shift, self.out_scale = self.out_shift.to(device), self.out_scale.to(device)
#         self.trainable_params = list(self.parameters())
#         self.device = device
#
#
#     def get_param_values(self,
# 						# *args, **kwargs
# 						):
#         params = torch.cat([p.contiguous().view(-1).data for p in self.parameters()])
#         return params.clone()
#
#
#     def set_param_values(self, new_params,
# 						# *args, **kwargs
# 						):
#         current_idx = 0
#         for idx, param in enumerate(self.parameters()):
#             vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
#             vals = vals.reshape(self.param_shapes[idx])
#             # clip std at minimum value
#             vals = torch.max(vals, self.min_log_std) if idx == 0 else vals
#             vals = torch.min(vals, self.max_log_std) if idx == 0 else vals
#             param.data = vals.to(self.device).clone()
#             current_idx += self.param_sizes[idx]
#         # update log_std_val for sampling
#         self.log_std_val = np.float64(self.log_std.to('cpu').data.numpy().ravel())
#         self.log_std_val = np.clip(self.log_std_val, self.min_log_std_val, self.max_log_std_val)
#         self.trainable_params = list(self.parameters())
#
#
#     def set_transformations(self, in_shift=None, in_scale=None,
#                             out_shift=None, out_scale=None,
# 							# *args, **kwargs
# 							):
#         in_shift = self.in_shift if in_shift is None else tensorize(in_shift)
#         in_scale = self.in_scale if in_scale is None else tensorize(in_scale)
#         out_shift = self.out_shift if out_shift is None else tensorize(out_shift)
#         out_scale = self.out_scale if out_scale is None else tensorize(out_scale)
#         self.in_shift, self.in_scale = in_shift.to(self.device), in_scale.to(self.device)
#         self.out_shift, self.out_scale = out_shift.to(self.device), out_scale.to(self.device)
#
#
#     # Main functions
#     # ============================================
#     def get_action(self, obs):
#         assert type(obs) == np.ndarray
#         if self.device != 'cpu':
#             print("Warning: get_action function should be used only for simulation.")
#             print("Requires policy on CPU. Changing policy device to CPU.")
#             self.to('cpu')
#         o = np.float32(obs.reshape(1, -1))
#         self.obs_var.data = torch.from_numpy(o)
#         mean = self.forward(self.obs_var).to('cpu').data.numpy().ravel()
#         noise = np.exp(self.log_std_val) * np.random.randn(self.act_dim)
#         act = mean + noise
#         return [act, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]
#
#
#     def mean_LL(self, obs, act, log_std=None,
# 				# *args, **kwargs
# 				):
#         if type(obs) == np.ndarray: obs = torch.from_numpy(obs).float()
#         if type(act) == np.ndarray: act = torch.from_numpy(act).float()
#         obs, act = obs.to(self.device), act.to(self.device)
#         log_std = self.log_std if log_std is None else log_std
#         mean = self.forward(obs)
#         zs = (act - mean) / torch.exp(self.log_std)
#         LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
#              - torch.sum(log_std) + \
#              - 0.5 * self.act_dim * np.log(2 * np.pi)
#         return mean, LL
#
#
#     def log_likelihood(self, obs, act, *args, **kwargs):
#         mean, LL = self.mean_LL(obs, act)
#         return LL.to('cpu').data.numpy()
#
#
#     def mean_kl(self, obs,
# 				# *args, **kwargs
# 				):
#         new_log_std = self.log_std
#         old_log_std = self.log_std.detach().clone()
#         new_mean = self.forward(obs)
#         old_mean = new_mean.detach()
#         return self.kl_divergence(new_mean, old_mean,
# 								  new_log_std, old_log_std,
# 								  # *args, **kwargs
# 								  )
#
#
#     def kl_divergence(self,
# 					  new_mean, old_mean,
# 					  new_log_std, old_log_std,
# 						# *args, **kwargs
# 						):
#         new_std, old_std = torch.exp(new_log_std), torch.exp(old_log_std)
#         Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
#         Dr = 2 * new_std ** 2 + 1e-8
#         sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
#         return torch.mean(sample_kl)
#
#
#
