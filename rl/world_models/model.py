# TODO: model
import random
import copy
import typing
import logging

import numpy as np
from numpy.random.mtrand import normal
import torch as T
from torch._C import dtype
from torch.distributions.normal import Normal
nn = T.nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


from rl.networks.mlp import MLPNet

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('lightning').setLevel(0)
T.multiprocessing.set_sharing_strategy('file_system')


LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20



class SimpleModel(LightningModule):

    def __init__(self, obs_dim, act_dim, rew_dim, configs) -> None:
        # print('init SimpleModel!')
        super(SimpleModel, self).__init__() # To automatically use 'def forward'
        # if seed:
        #     random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        self.configs = configs
        self._device_ = configs['experiment']['device']

        net_configs = configs['world_model']['network']
        net_arch = net_configs['arch']

        self.mu_log_sigma_net = MLPNet(obs_dim + act_dim, 0, net_configs)
        self.mu = nn.Linear(net_arch[-1], obs_dim + rew_dim)
        self.log_sigma = nn.Linear(net_arch[-1], obs_dim + rew_dim)

        self.max_log_sigma = nn.Parameter( T.ones([1, obs_dim + rew_dim]) / 2, requires_grad=False)
        self.min_log_sigma = nn.Parameter( -T.ones([1, obs_dim + rew_dim]) * 10, requires_grad=False)
        self.reparam_noise = 1e-6


    def get_model_dist_params(self, ips):
        net_out = self.mu_log_sigma_net(ips)
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = self.max_log_sigma - (self.max_log_sigma - log_sigma)
        log_sigma = self.min_log_sigma + (log_sigma - self.min_log_sigma)
        sigma = T.exp(log_sigma)
        sigma_inv = T.exp(-log_sigma)
        return mu, log_sigma, sigma, sigma_inv


    def deterministic(self, mu):
        pass


    def forward(self, o, a,
                deterministic= False):
        ips = T.as_tensor(T.cat([o, a], dim=-1), dtype=T.float32).to(self._device_)
        mu, log_sigma, sigma, sigma_inv = self.get_model_dist_params(
            T.as_tensor(ips, dtype=T.float32).to(self._device_))

        if deterministic:
            predictions = self.deterministic(mu)
        else:
            normal_ditribution = Normal(mu, sigma)
            predictions = normal_ditribution.sample()

        return predictions, mu, log_sigma, sigma, sigma_inv


    def train_Model(self, trainer, data_module, m):

        self.m = m

        batch_size = self.configs['world_model']['network']['batch_size']
        dropout = self.configs['world_model']['network']['dropout']
        # env_buffer.device = 'cpu'

        # data = DataModule(env_buffer, batch_size)
        if dropout != None: self.train()

        trainer.fit(self, data_module)

        if dropout != None: self.eval()

        return self.Jmean, self.J #, mEpochs


	### PyTorch Lightning ###
	# add: dropouts, regulaizers
    def configure_optimizers(self):
        opt = 'T.optim.' + self.configs['world_model']['network']['optimizer']
        lr = self.configs['world_model']['network']['lr']
        optimizer = eval(opt)(self.parameters(), lr=lr)
        return optimizer

	# def compute_model_loss(self, batch):
	# 	device = self.configs['Experiment']['device']
	# 	O, A, R, O_next, D = batch
	# 	D = T.as_tensor(D, dtype=T.bool).to(device)

	# 	Prediction, mean, log_std, std, inv_std = self(O, A) # dyn_delta, reward
	# 	mean_target = T.cat([O_next - O, R], dim=-1)


	# 	Jmean = T.mean(T.mean(T.square(mean - mean_target) * inv_std, dim=-1), dim=-1) # batch loss
	# 	Jstd = T.mean(T.mean(log_std, dim=-1), dim=-1)
	# 	pass

    def training_step(self, batch, batch_idx):
        device = self._device_
        O, A, R, O_next, D = batch
        D = T.as_tensor(D, dtype=T.bool).to(device)

        _, mean, log_std, std, inv_std = self(O, A) # dyn_delta, reward
        mean_target = T.cat([O_next - O, R], dim=-1)

        # 2 Compute obj function
        Jmean = T.mean(T.mean(T.square(mean - mean_target) * inv_std * ~D, dim=-1), dim=-1) # batch loss
        Jstd = T.mean(T.mean(log_std * ~D, dim=-1), dim=-1)
        Jwl2 = self.weight_l2_loss()
        J = Jmean + Jstd + Jwl2
        J += 0.01 * (T.sum(self.max_log_sigma) - T.sum(self.min_log_sigma))
        # print('J=', J)


        self.log(f'Model {self.m+1}, Jmean_train', Jmean.item(), prog_bar=True)
        self.Jmean = Jmean.item()
        self.J = J.item() # We no longer need it; bc it's auto optimized
        # print('self.J=', self.J)

        return J


    def validation_step(self, batch, batch_idx):
        device = self._device_
        O, A, R, O_next, D = batch
        D = T.as_tensor(D, dtype=T.bool).to(device)

        _, mean, log_std, std, inv_std = self(O, A) # dyn_delta, reward
        mean_target = T.cat([O_next - O, R], dim=-1)

        # 2 Compute obj function
        Jmean = T.mean(T.mean(T.square(mean - mean_target) * inv_std * ~D, dim=-1), dim=-1) # batch loss
        Jstd = T.mean(T.mean(log_std * ~D, dim=-1), dim=-1)
        Jwl2 = self.weight_l2_loss()
        J = Jmean + Jstd + Jwl2
        J += 0.01 * (T.sum(self.max_log_sigma) - T.sum(self.min_log_sigma))

        self.log("Jmean_val", Jmean.item(), prog_bar=True)


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items


    # def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]):
    def weight_l2_loss(self): # must have 4 hid-layers in the WorldModel
        l2_loss_coefs = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]
        weight_norms = []
        for name, weight in self.named_parameters():
        	if "weight" in name:
        		weight_norms.append(weight.norm(2))
        weight_norms = T.stack(weight_norms, dim=0)
        # print('l2_loss_coefs: ', T.tensor(l2_loss_coefs, device=weight_norms.device))
        # print('weight_norms: ', weight_norms)
        weight_decay = (T.tensor(l2_loss_coefs, device=weight_norms.device) * weight_norms).sum()
        return weight_decay
