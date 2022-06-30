import random
import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

from .mlp import MLPNet


LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20



class StochasticPolicy(nn.Module):
    """
    Stochastic Gaussian Policy, π(at|st; φ)
        A mapping form observation space to a gaussian ditribution over action space. 
    """
    def __init__(self, obs_dim, act_dim,
                 act_up_lim, act_low_lim,
                 net_configs, device, seed) -> None:
        # print('Initialize Policy!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        self.device = device
        net_arch = net_configs['arch']
        optimizer = 'T.optim.' + net_configs['optimizer']
        lr = net_configs['lr']

        self.mu_log_sigma_net = MLPNet(obs_dim, 0, net_configs, seed)
        self.mu = nn.Linear(net_arch[-1], act_dim)
        self.log_sigma = nn.Linear(net_arch[-1], act_dim)

        self.optimizer = eval(optimizer)(self.parameters(), lr)

        self.act_dim = act_dim
        self.act_scale = T.FloatTensor( 0.5 * (act_up_lim - act_low_lim) ).to(device)
        self.act_bias =  T.FloatTensor( 0.5 * (act_up_lim + act_low_lim) ).to(device)
        self.reparam_noise = 1e-6


        super().__init__() # To automatically use 'def forward'


    def get_mu_sigma(self, o):
        net_out = self.mu_log_sigma_net(o)
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = T.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        sigma = T.exp(log_sigma)
        return mu, sigma


    def prob(self, mu, sigma, reparameterize):
        pi_distribution = Normal(mu, sigma)

        if reparameterize:
            pre_tanh_pi = pi_distribution.rsample()
        else:
            pre_tanh_pi = pi_distribution.sample()

        pi = T.tanh(pre_tanh_pi)
        return pi, pre_tanh_pi
        # return (pi * self.act_scale) + self.act_bias


    def log_prob(self, pi, pre_tanh_pi, mu, sigma):
        pi_distribution = Normal(mu, sigma)
        log_pi = pi_distribution.log_prob(pre_tanh_pi)
        log_pi -= T.log(self.act_scale * (1- pi.pow(2)) + self.reparam_noise)
        log_pi = log_pi.sum(-1, keepdim=True)
        return log_pi


    def deterministic(self, mu):
        with T.no_grad():
            return (T.tanh(mu) * self.act_scale) + self.act_bias


    def forward(self, o,
                deterministic= False,
                return_log_pi=True,
                reparameterize=True):

        mu, sigma = self.get_mu_sigma(T.as_tensor(o, dtype=T.float32).to(self.device))

        log_pi = None

        if deterministic:
            pi = self.deterministic(mu)
        else:
            if return_log_pi:
                pi, pre_tanh_pi = self.prob(mu, sigma, reparameterize)
                log_pi = self.log_prob(pi, pre_tanh_pi, mu, sigma)
            else:
                pi, _ = self.prob(mu, sigma, reparameterize)
            pi = (pi * self.act_scale) + self.act_bias

        return pi, log_pi


    def step_np(self, o,
                deterministic= False,
                return_log_pi=True,
                reparameterize=True):
        pi, log_pi = self.forward(o, deterministic, return_log_pi, reparameterize)
        if self.device == 'cuda:0':
            pi = pi.cpu().detach().numpy()
        else:
            pi = pi.detach().numpy()
        return pi, log_pi

