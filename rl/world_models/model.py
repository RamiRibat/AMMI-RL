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


# from rl.networks.mlp import MLPNet


LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20


class RLDataset(IterableDataset):
    pass


class DataModule(pl.LightningDataModule):
    pass




class SimpleModel(pl.LightningModule):
    def __init__(self, obs_dim, act_dim, rew_dim,
                 net_configs, device, seed) -> None:
        print('Initialize SimpleModel!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        self.net_configs = net_configs
        self.device = device
        net_arch = net_configs['arch']

        self.mu_log_sigma_net = MLPNet(obs_dim + act_dim, 0, net_configs, seed)
        self.mu = nn.Linear(net_arch[-1], obs_dim + rew_dim)
        self.log_sigma = nn.Linear(net_arch[-1], obs_dim + rew_dim)

        self.max_log_sigma = nn.Parameter( T.ones([1, obs_dim + rew_dim]) / 2, requires_grad=False)
        self.min_log_sigma = nn.Parameter( -T.ones([1, obs_dim + rew_dim]) * 10, requires_grad=False)
        self.reparam_noise = 1e-6

        super().__init__() # To automatically use 'def forward'


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
        ips = T.as_tensor(T.cat([o, a], dim=-1), dtype=T.float32).to(self.device)
        mu, log_sigma, sigma, sigma_inv = self.get_model_dist_params(
            T.as_tensor(ips, dtype=T.float32).to(self.device))

        if deterministic:
            predictions = self.deterministic(mu)
        else:
            normal_ditribution = Normal(mu, sigma)
            predictions = normal_ditribution.sample()

        return predictions, mu, log_sigma, sigma, sigma_inv


    def trainModel(self, env_buffer, batch_size, m):
        # batch_data = env_buffer.get_recent_data(batch_size)
        # Os = batch_data['observations']
        # As = batch_data['actions']
        # self.obs_norm.update(Os)
        # self.act_norm.update(As)

        self.m = m

        # NT = self.config['Algorithm']['Learning']['epSteps']
        # Ni = self.config['Algorithm']['Learning']['iEpochs']
        M0 = self.config['Model']['Network']['mEpochs']
        # BSize = env_buffer.size
        mEpochs = M0#int(NT*Ni*M0/BSize)
        # mEpochs = int(300 - (9e-4)*BSize)

        batch_size = self.config['Model']['Network']['batch_size']
        dropout = self.config['Model']['Network']['dropout']
        env_buffer.device = 'cpu'

        data = DataModule(env_buffer, batch_size)
        if dropout != None: self.train()
        ws_summ = None
        # checkpoint_callback = ModelCheckpoint(save_last=None)
        dyn_trainer = pl.Trainer(max_epochs=mEpochs,
        						 gpus=1,
        						 weights_summary=ws_summ,
        						 checkpoint_callback=False,
        						 logger=False,
        						 # progress_bar_refresh_rate=0,
        						 # log_save_interval=100,
        						 # callbacks=[checkpoint_callback]
        						 )
        # print('SimpleModel Self: ', self)
        dyn_trainer.fit(self, data)
        if dropout != None: self.eval()

        env_buffer.device = self.config['Experiment']['device']
        # # dyn_trainer.save_checkpoint(f"/home/rami/AI/RL/myGitHub/FUSION/lightning_logs/DynModelsCkPts/model{m+1}")
        return self.Jdyn, mEpochs



	### PyTorch Lightning ###
	# add: dropouts, regulaizers
    def configure_optimizers(self):
        opt = 'th.optim.' + self.config['Model']['Network']['optimizer']
        lr = self.config['Model']['Network']['lr']
        optimizer = eval(opt)(self.parameters(), lr=lr)
        return optimizer

	# def compute_model_loss(self, batch):
	# 	device = self.config['Experiment']['device']
	# 	O, A, R, O_next, D = batch
	# 	D = th.as_tensor(D, dtype=th.bool).to(device)

	# 	Prediction, mean, log_std, std, inv_std = self(O, A) # dyn_delta, reward
	# 	mean_target = th.cat([O_next - O, R], dim=-1)


	# 	Jmean = th.mean(th.mean(th.square(mean - mean_target) * inv_std, dim=-1), dim=-1) # batch loss
	# 	Jstd = th.mean(th.mean(log_std, dim=-1), dim=-1)
	# 	pass

    def training_step(self, batch, batch_idx):
        device = self.config['Experiment']['device']
        O, A, R, O_next, D = batch
        D = th.as_tensor(D, dtype=th.bool).to(device)

        Prediction, mean, log_std, std, inv_std = self(O, A) # dyn_delta, reward
        mean_target = th.cat([O_next - O, R], dim=-1)

        # 2 Compute obj function
        # Jmean = th.mean(th.square(mean - mean_target) * inv_std * ~D) # batch loss
        # Jstd = th.mean(th.abs(log_std) * ~D)
        # Jmodel = Jmean + Jstd
        # Jmodel += 0.01 * th.norm(th.sum(self.max_log_std) - th.sum(self.min_log_std))
        # Jl2 = self.compute_l2_loss()
        # Jdyn = Jmodel + Jl2
        # print('additional ', 0.01 * (th.sum(self.max_log_std) - th.sum(self.min_log_std)))
        # dynAcc = accuracy(Prediction, Backup)

        # 2 Compute obj function
        # Jmean = th.mean(th.mean(th.square(mean - mean_target) * inv_std, dim=-1), dim=-1) # batch loss
        # Jstd = th.mean(th.mean(log_std, dim=-1), dim=-1)
        Jmean = th.mean(th.mean(th.square(mean - mean_target) * inv_std * ~D, dim=-1), dim=-1) # batch loss
        Jstd = th.mean(th.mean(log_std * ~D, dim=-1), dim=-1)
        Jl2 = self.compute_l2_loss()
        Jdyn = Jmean + Jstd + Jl2
        Jdyn += 0.01 * (th.sum(self.max_log_std) - th.sum(self.min_log_std))



        # Jmean = th.mean(th.mean(th.square(mean - mean_target) * inv_std * ~D, dim=-1), dim=-1) # batch loss
        # Jstd = th.mean(th.mean(log_std * ~D, dim=-1), dim=-1)
        # Jmodel =
        # Jl2 = self.compute_l2_loss()
        # Jdyn = Jmodel + Jl2
        # Jdyn += 0.01 * (th.sum(self.max_log_std) - th.sum(self.min_log_std))


        self.log(f'Model {self.m+1}, Loss', Jdyn.item(), prog_bar=True)
        # self.log('Acc', dynAcc, prog_bar=True)
        # print(f'Jdyn (simple): {Jdyn}')
        self.Jdyn = Jdyn.item() # We no longer need it; bc it's auto optimized

        # dynAcc = accuracy(Prediction, Backup)
        # pbar = {'train_acc': dynAcc}
        # return {'loss': Jdyn, 'progress_bar': pbar} # Lightning will auto detach it
        # return {'loss': Jdyn}

        return Jdyn

	# def validation_step(self, batch, batch_idx):
	# 	# In validation loop, we generally don't want to put a metrics for
	# 	# every batch (all batches are independent), plot metric for the whole validation set
	# 	results = self.training_step(batch, batch_idx)
	# 	results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
	# 	del results['progress_bar']['train_acc']
	# 	return results

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items


    # def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]):
    def compute_l2_loss(self):
        l2_loss_coefs = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]
        weight_norms = []
        for name, weight in self.named_parameters():
        	if "weight" in name:
        		weight_norms.append(weight.norm(2))
        weight_norms = T.stack(weight_norms, dim=0)
        weight_decay = (T.tensor(l2_loss_coefs, device=weight_norms.device) * weight_norms).sum()
        return weight_decay
