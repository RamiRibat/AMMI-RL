# TODO: model
import random
import copy
import typing

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('lightning').setLevel(0)

import numpy as np
from numpy.random.mtrand import normal
import torch as T
from torch._C import dtype
from torch.distributions.normal import Normal
nn = T.nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
# T.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

from rl.networks.mlp import MLPNet



LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20


art_zero = 1e-8


class DynamicsModel(LightningModule):

    def __init__(self, obs_dim, act_dim, rew_dim, configs,
                    obs_bias=None, obs_scale=None,
                    act_bias=None, act_scale=None,
                    out_bias=None, out_scale=None) -> None:
        # print('init SimpleModel!')
        super(DynamicsModel, self).__init__() # To automatically use 'def forward'
        # if seed:
        #     random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        self.val = False

        self.configs = configs
        self._device_ = configs['experiment']['device']

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim + rew_dim
        self.normalization(obs_bias, obs_scale, act_bias, act_scale, out_bias, out_scale)

        net_configs = configs['world_model']['network']
        net_arch = net_configs['arch']

        self.mu_log_sigma_net = MLPNet(obs_dim + act_dim, 0, net_configs)
        self.mu = nn.Linear(net_arch[-1], obs_dim + rew_dim)
        self.log_sigma = nn.Linear(net_arch[-1], obs_dim + rew_dim)

        self.max_log_sigma = nn.Parameter( T.ones([1, obs_dim + rew_dim]) / 2, requires_grad=False)
        self.min_log_sigma = nn.Parameter( -T.ones([1, obs_dim + rew_dim]) * 10, requires_grad=False)
        self.reparam_noise = 1e-6

        self.normalize = True
        self.normalize_out = True

        self.loss = nn.MSELoss()



    def get_model_dist_params(self, ips):
        net_out = self.mu_log_sigma_net(ips)
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = self.max_log_sigma - (self.max_log_sigma - log_sigma)
        log_sigma = self.min_log_sigma + (log_sigma - self.min_log_sigma)
        sigma = T.exp(log_sigma)
        sigma_inv = T.exp(-log_sigma)
        return mu, log_sigma, sigma, sigma_inv


    # def deterministic(self, mu):
    #     pass


    def normalization(self, obs_bias=None, obs_scale=None,
                            act_bias=None, act_scale=None,
                            out_bias=None, out_scale=None):

        device = self._device_

        if obs_bias is None:
            self.obs_bias   = T.zeros(self.obs_dim)
            self.obs_scale  = T.ones(self.obs_dim)
            self.act_bias   = T.zeros(self.act_dim)
            self.act_scale  = T.ones(self.act_dim)
            self.out_bias   = T.zeros(self.out_dim)
            self.out_scale  = T.ones(self.out_dim)

        self.obs_bias   = self.obs_bias.to(device)
        self.obs_scale  = self.obs_scale.to(device)
        self.act_bias   = self.act_bias.to(device)
        self.act_scale  = self.act_scale.to(device)
        self.out_bias   = self.out_bias.to(device)
        self.out_scale  = self.out_scale.to(device)
        self.mask = self.out_scale >= art_zero


    def forward(self, o, a, deterministic= False):

        normed_o = (o - self.obs_bias)/(self.obs_scale + art_zero)
        normed_a = (a - self.act_bias)/(self.act_scale + art_zero)

        ips = T.as_tensor(T.cat([normed_o, normed_a], dim=-1), dtype=T.float32).to(self._device_)

        mu, log_sigma, sigma, sigma_inv = self.get_model_dist_params(
            T.as_tensor(ips, dtype=T.float32).to(self._device_))

        if self.normalize_out:
            print('\nmu mean bf', T.mean(mu, dim=0))
            mu = mu * (self.out_scale + art_zero) + self.out_bias
            print('mu mean af', T.mean(mu, dim=0))

        if deterministic:
            predictions = self.deterministic(mu)
        else:
            normal_ditribution = Normal(mu, sigma)
            predictions = normal_ditribution.rsample()

        return predictions, mu, log_sigma, sigma, sigma_inv


    def train_Model(self, data_module, m):
        device = self._device_

        self.m = m

        M = self.configs['world_model']['num_ensembles']
        model_type = self.configs['world_model']['type']
        num_elites = self.configs['world_model']['num_elites']
        wm_epochs = self.configs['algorithm']['learning']['grad_WM_steps']

        batch_size = self.configs['world_model']['network']['batch_size']
        dropout = self.configs['world_model']['network']['dropout']
        # env_buffer.device = 'cpu'

        # data = DataModule(env_buffer, batch_size)
        # if dropout != None: self.train()

        self.trainer = Trainer(max_epochs=wm_epochs,
                          # log_every_n_steps=2,
                          accelerator=device, devices='auto',
                          gpus=0,
                          enable_model_summary=False,
                          enable_checkpointing=False,
                          progress_bar_refresh_rate=20,
                          # log_save_interval=100,
                          logger=False, #self.pl_logger,
                          # callbacks=[checkpoint_callback, enable_model_summary],
                          )

        self.normalize_out = False
        self.trainer.fit(self, data_module)
        self.normalize_out = True

        # print('\nNormalized:')
        # print(f'obs_bias={self.obs_bias}, \nobs_scale={self.obs_scale}')
        # print(f'act_bias={self.act_bias}, \nact_scale={self.act_scale}')
        # print(f'out_bias={self.out_bias}, \nout_scale={self.out_scale}')

        # if dropout != None: self.eval()

        if self.val:
            return self.train_log, self.val_log
        else:
            return self.train_log, None


    def test_Model(self, data_module):
        self.trainer.test(self, data_module)
        return self.test_loss


	### PyTorch Lightning ###
	# add: dropouts, regulaizers
    def configure_optimizers(self):
        opt = 'T.optim.' + self.configs['world_model']['network']['optimizer']
        lr = self.configs['world_model']['network']['lr']
        optimizer = eval(opt)(self.parameters(), lr=lr)
        return optimizer


    def training_step(self, batch, batch_idx):
        self.train_log = dict()

        Jmean, Jsigma, J = self.compute_objective(batch)

        # self.log(f'Model {self.m+1}, Jmean_train', Jmean.item(), prog_bar=True)
        self.log(f'Model {self.m+1}, J_train', J.item(), prog_bar=True)

        self.train_log['mean'] = Jmean.item()
        # self.train_log['sigma'] = Jsigma.item()
        self.train_log['total'] = J.item()

        return J


    def validation_step(self, batch, batch_idx):
        self.val = True
        self.val_log = dict()

        Jmean, Jsigma, J = self.compute_objective(batch)

        # self.log("Jmean_val", Jmean.item(), prog_bar=True)
        self.log("J_val", J.item(), prog_bar=True)

        self.val_log['mean'] = Jmean.item()
        # self.val_log['sigma'] = Jsigma.item()
        self.val_log['total'] = J.item()


    def test_step(self, batch, batch_idx):
        # Model prediction performance
        loss = self.compute_test_loss(batch)
        self.log("mse_loss", loss.item(), prog_bar=True)
        self.test_loss = loss.item()


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items


    def compute_objective(self, batch):
        O, A, R, O_next, D = batch
        D = T.as_tensor(D, dtype=T.bool).to(self._device_)

        if self.normalize:
            # print('compute_objective: normalize')
            obs_bias, act_bias = T.mean(O, dim=0), T.mean(A, dim=0)
            obs_scale, act_scale = T.mean(T.abs(O - obs_bias), dim=0), T.mean(T.abs(A - act_bias), dim=0)
            out_bias = T.mean(O - O_next, dim=0)
            out_scale = T.mean(T.abs(O - O_next - out_bias), dim=0)
            self.normalization(obs_bias, obs_scale, act_bias, act_scale, out_bias, out_scale)

        predictions, mean, log_sigma, _, inv_sigma = self(O, A) # dyn_delta, reward
        mean_target = T.cat([O_next - O, R], dim=-1)

        # Jmean = T.mean(T.mean(T.square(mean - mean_target) * inv_sigma * ~D, dim=-1), dim=-1) # batch loss
        Jmean = self.loss(mean, mean_target)
        Jsigma = T.tensor([0.0])#T.mean(T.mean(log_sigma * ~D, dim=-1), dim=-1)
        # Jwl2 = self.weight_l2_loss()
        J = Jmean #+ Jsigma + Jwl2
        # J += 0.01 * (T.sum(self.max_log_sigma) - T.sum(self.min_log_sigma))

        # J = self.loss(predictions, mean_target)

        return Jmean, Jsigma, J


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


    def compute_test_loss(self, batch):
        O, A, R, O_next, D = batch
        D = T.as_tensor(D, dtype=T.bool).to(self._device_)

        preds, _, _, _, _ = self(O, A) # dyn_delta, reward
        # print('preds: ', preds.shape)
        preds_target = T.cat([O_next - O, R], dim=-1)
        # print('preds_target: ', preds_target.shape)

        loss = self.loss(preds, preds_target)

        return loss
