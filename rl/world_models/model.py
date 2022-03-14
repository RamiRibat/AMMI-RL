# TODO: model
import random
import copy
import typing

import warnings
warnings.filterwarnings('ignore')

# import logging
# logging.getLogger('lightning').setLevel(0)

import numpy as np
from numpy.random.mtrand import normal
import torch
import torch as T
from torch._C import dtype
from torch.distributions.normal import Normal
nn = T.nn
F = nn.functional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
# T.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from rl.networks.mlp import MLPNet



LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20

epsilon = 1e-8

# T.set_default_tensor_type(torch.cuda.FloatTensor)
# device = torch.device('cpu')
import itertools





"""
source: https://github.com/Xingyu-Lin/mbpo_pytorch/model.py
"""


def init_weights__(l):
    """
    source: https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
    """

    def truncated_normal_(w, mean=0.0, std=1.0):
        nn.init.normal_(w, mean=mean, std=std)
        while True:
            i = T.logical_or(w < mean - 2*std, w > mean + 2*std)
            bound = T.sum(i).item()
            if bound == 0: break
            w[i] = T.normal(mean, std, size=(bound, ), device=w.device)
        return w

    if isinstance(l, nn.Linear):
        ip_dim = l.weight.data.shape[0]
        std = 1 / (2 * np.sqrt(ip_dim))
        truncated_normal_(l.weight.data, std=std)
        l.bias.data.fill_(0.0)


def init_weights_(m):

    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape).to(t.device), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC) or isinstance(m, EnsembleLayer):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x






class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ensemble_size: int,
                 weight_decay: float = 0.,
                 bias: bool = True
                 ) -> None:

        super(EnsembleFC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay

        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )





class EnsembleLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    ensemble_size: int
    in_features: int
    out_features: int
    weight: T.Tensor

    def __init__(self,
                 ensemble_size: int,
                 in_features: int,
                 out_features: int,
                 weight_decay: float = 0.,
                 bias: bool = True
                 ) -> None:

        super(EnsembleLayer, self).__init__()

        self.ensemble_size = ensemble_size

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter( T.Tensor(ensemble_size, in_features, out_features) )
        self.weight_decay = weight_decay

        if bias:
            self.bias = nn.Parameter( T.Tensor(ensemble_size, out_features) )
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )




class EnsembleModel(nn.Module):

    def __init__(self,
                 state_size,
                 action_size,
                 reward_size,
                 ensemble_size,
                 hidden_size=200,
                 learning_rate=1e-3,
                 use_decay=False,
                 device='cpu'
                 ):

        super(EnsembleModel, self).__init__()

        self._device_ = device

        # self.hidden_size = hidden_size

        # self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025).to(device)
        # self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005).to(device)
        # self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075).to(device)
        # self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075).to(device)

        self.use_decay = use_decay

        self.output_dim = state_size + reward_size

        # Add variance output
        # self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001).to(device)



        net_arch = [200, 200, 200, 200] #net_configs['arch']
        activation = 'Swish' #'nn.' + net_configs['activation']
        # op_activation = 'nn.Identity' # net_config['output_activation']
        num_ensemble = ensemble_size
        layers_decay = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001]

        if len(net_arch) > 0:
            layers = [ EnsembleLayer(num_ensemble, state_size + action_size, net_arch[0], weight_decay=layers_decay[0]), eval(activation)() ]
            for l in range(len(net_arch)-1):
                layers.extend([ EnsembleLayer(num_ensemble, net_arch[l], net_arch[l+1], weight_decay=layers_decay[l+1]), eval(activation)() ])
            if self.output_dim > 0:
                layers.extend([ EnsembleLayer(num_ensemble, net_arch[-1], self.output_dim*2, weight_decay=layers_decay[-1])])
        else:
            raise 'No network arch!'

        self.nn_model = nn.Sequential(*layers)
        self.nn_model.to(device)



        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)

        self.gnll_loss = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.apply(init_weights_)

        # self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        # nn1_output = self.swish(self.nn1(x))
        # nn2_output = self.swish(self.nn2(nn1_output))
        # nn3_output = self.swish(self.nn3(nn2_output))
        # nn4_output = self.swish(self.nn4(nn3_output))
        # nn5_output = self.nn5(nn4_output)

        nn_output = self.nn_model(x)

        mean = nn_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC) or isinstance(m, EnsembleLayer):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def compute_loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3

        # inv_var = torch.exp(-logvar)

        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            # mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            # var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            # total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
            losses = T.tensor([ self.gnll_loss(mean[m, :, :], labels[m, :, :], T.exp(logvar[m, :, :])) for m in range(mean.shape[0]) ])
            total_loss = self.gnll_loss(mean, labels, T.exp(logvar))
            return total_loss, losses
        else:
            # losses = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            # total_loss = torch.sum(losses)
            losses = T.tensor([ self.mse_loss(mean[m, :, :], labels[m, :, :]) for m in range(mean.shape[0]) ])
            total_loss = self.mse_loss(mean, labels)
            return total_loss, losses

        # return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)

        if self.use_decay:
            loss += self.get_decay_loss()

        loss.backward()
        self.optimizer.step()





class EnsembleDynamicsModel():

    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False, device='cpu'):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay, device=device)

        self.scaler = StandardScaler()

        self._device_ = device


    def train(self, inputs, labels, batch_size=256, holdout_ratio=0., max_epochs_since_update=5):
        device = self._device_

        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        # LossList = []

        # for epoch in range(5):
        for epoch in itertools.count():
            # losses = []
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.compute_loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.compute_loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))

        return np.mean(holdout_mse_losses)

    def _save_best(self, epoch, val_losses):
        updated = False

        for i in range(len(val_losses)):
            current = val_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best

            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False


    def predict(self, inputs, batch_size=1024, factored=True):
        device = self._device_

        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []

        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())

        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var
















class DynamicsModel(LightningModule):

    def __init__(self, obs_dim, act_dim, rew_dim, configs, device,
                    obs_bias=None, obs_scale=None,
                    act_bias=None, act_scale=None,
                    out_bias=None, out_scale=None) -> None:
        # print('init SimpleModel!')
        super(DynamicsModel, self).__init__() # To automatically use 'def forward'
        # if seed:
        #     random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        self.val = False

        self.configs = configs
        self._device_ = device

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.inp_dim = inp_dim = obs_dim + act_dim
        self.out_dim = out_dim = obs_dim #+ rew_dim
        self.normalization(obs_bias, obs_scale, act_bias, act_scale, out_bias, out_scale)

        net_configs = configs['world_model']['network']
        net_arch = net_configs['arch']

        self.mu_log_sigma_net = MLPNet(inp_dim, 0, net_configs)
        self.mu = nn.Linear(net_arch[-1], out_dim)
        if configs['world_model']['type'][0] == 'P':
            self.log_sigma = nn.Linear(net_arch[-1], out_dim)

            self.min_log_sigma = nn.Parameter( -10.0 * T.ones([1, out_dim]),
                                              requires_grad=configs['world_model']['learn_log_sigma_limits'])
            self.max_log_sigma = nn.Parameter(T.ones([1, out_dim]) / 2.0,
                                          requires_grad=configs['world_model']['learn_log_sigma_limits'])
        self.reparam_noise = 1e-6

        self.apply(init_weights_)

        # print('DynamicsModel: ', self)

        self.normalize = True
        self.normalize_out = True

        self.gnll_loss = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()
        # self..to(device)
        # self.to(self._device_)


    def normalization(self, obs_bias=None, obs_scale=None,
                            act_bias=None, act_scale=None,
                            out_bias=None, out_scale=None):

        # device = self._device_

        if obs_bias is None:
            self.obs_bias   = T.zeros(self.obs_dim)
            self.obs_scale  = T.ones(self.obs_dim)
            self.act_bias   = T.zeros(self.act_dim)
            self.act_scale  = T.ones(self.act_dim)
            self.out_bias   = T.zeros(self.out_dim)
            self.out_scale  = T.ones(self.out_dim)

        self.obs_bias   = self.obs_bias#.to(device)
        self.obs_scale  = self.obs_scale#.to(device)
        self.act_bias   = self.act_bias#.to(device)
        self.act_scale  = self.act_scale#.to(device)
        self.out_bias   = self.out_bias#.to(device)
        self.out_scale  = self.out_scale#.to(device)
        self.mask = self.out_scale >= epsilon


    def to(self, device):
        self.obs_bias = self.obs_bias.to(device)
        self.obs_scale = self.obs_scale.to(device)
        self.act_bias = self.act_bias.to(device)
        self.act_scale = self.act_scale.to(device)
        self.out_bias = self.out_bias.to(device)
        self.out_scale = self.out_scale.to(device)
        return super(DynamicsModel, self).to(device)



    def get_model_dist_params(self, ips):
        net_out = self.mu_log_sigma_net(ips)
        mu, log_sigma = self.mu(net_out), self.log_sigma(net_out)
        log_sigma = self.max_log_sigma - F.softplus(self.max_log_sigma - log_sigma)
        log_sigma = self.min_log_sigma + F.softplus(log_sigma - self.min_log_sigma)

        sigma = T.exp(log_sigma)
        sigma_inv = T.tensor([0.0]) #T.exp(-log_sigma)
        #     exit()

        return mu, log_sigma, sigma, sigma_inv


    def forward(self, o, a, deterministic= False):
        # print('\nself.obs_bias: ', self.obs_bias)
        # print('o: ', o)
        normed_o = (o - self.obs_bias)/(self.obs_scale + epsilon)
        normed_a = (a - self.act_bias)/(self.act_scale + epsilon)

        ips = T.as_tensor(T.cat([normed_o, normed_a], dim=-1), dtype=T.float32)#.to(self._device_)

        mu, log_sigma, sigma, sigma_inv = self.get_model_dist_params(
            T.as_tensor(ips, dtype=T.float32)#.to(self._device_)
            )

        if self.normalize_out:
            mu = mu * (self.out_scale + epsilon) + self.out_bias

        return mu, log_sigma, sigma, sigma_inv


    def train_Model(self, data_module, m):
        device = self._device_
        # device = 'gpu' if self._device_=='cuda' else self._device_

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

        early_stop_callback = EarlyStopping(monitor="Jval",
                                            min_delta=0.0,
                                            patience=5,
                                            # verbose=False,
                                            mode="max"
                                            )

        self.trainer = Trainer(
                          # max_epochs=wm_epochs,
                          # log_every_n_steps=2,
                          # accelerator=device, devices='auto',
                          gpus=[eval(device[-1])] if device[:-2]=='cuda' else 0,
                          # gpus=2, strategy='dp',
                          enable_model_summary=False,
                          enable_checkpointing=False,
                          progress_bar_refresh_rate=20,
                          # log_save_interval=100,
                          logger=False, #self.pl_logger,
                          callbacks=[early_stop_callback],
                          )

        self.normalize_out = False
        self.trainer.fit(self, data_module)

        self.normalize_out = True

        # if dropout != None: self.eval()

        if self.val:
            return self.Jtrain, self.Jval
        else:
            return self.Jtrain, None


    def test_Model(self, data_module):
        self.trainer.test(self, data_module)
        return self.test_loss#, self.wm_mu, self.wm_sigma


	### PyTorch Lightning ###
	# add: dropouts, regulaizers
    def configure_optimizers(self):
        opt = 'T.optim.' + self.configs['world_model']['network']['optimizer']
        lr = self.configs['world_model']['network']['lr']
        wd = self.configs['world_model']['network']['wd']
        eps = self.configs['world_model']['network']['eps']
        optimizer = eval(opt)(self.parameters(), lr=lr, weight_decay=wd, eps=eps)
        return optimizer


    def training_step(self, batch, batch_idx):
        Jmu, Jsigma, J = self.compute_objective(batch)
        self.log(f'Model {self.m+1}, Jtrain', J.item(), prog_bar=True)
        return J

    def training_step_end(self, batch_parts):
        if batch_parts.shape == T.Size([]):
            return batch_parts
        else:
            return sum(batch_parts) / len(batch_parts)

    def training_epoch_end(self, outputs) -> None:
        outputs_list = []
        for out in outputs:
            outputs_list.append(out['loss'])
        self.Jtrain = T.stack(outputs_list).mean().item()


    def validation_step(self, batch, batch_idx):
        self.val = True
        Jmean, Jsigma, J = self.compute_objective(batch)
        self.log("Jval", J.item(), prog_bar=True, sync_dist=True)
        return J

    def validation_epoch_end(self, outputs) -> None:
        self.Jval = T.stack(outputs).mean().item()


    def test_step(self, batch, batch_idx):
        # Model prediction performance
        loss = self.compute_test_loss(batch)
        self.log("mse_loss", loss.item(), prog_bar=True, sync_dist=True)
        return loss

    def test_epoch_end(self, outputs) -> None:
        self.test_loss = T.stack(outputs).mean().item()


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items


    def compute_objective(self, batch):
        O, A, R, O_next, D = batch
        # D = T.as_tensor(D, dtype=T.bool).to(self._device_)

        if self.normalize:
            # print('compute_objective: normalize')
            obs_bias, act_bias = T.mean(O, dim=0), T.mean(A, dim=0)
            obs_scale, act_scale = T.mean(T.abs(O - obs_bias), dim=0), T.mean(T.abs(A - act_bias), dim=0)
            out_bias = T.mean(O - O_next, dim=0)
            out_scale = T.mean(T.abs(O - O_next - out_bias), dim=0)
            self.normalization(obs_bias, obs_scale, act_bias, act_scale, out_bias, out_scale)

        mu, log_sigma, sigma, sigma_inv = self(O, A) # dyn_delta, reward
        # print('mu: ', mu.shape)
        # print('sigma: ', sigma.shape)
        # mu_target = T.cat([O_next - O, R], dim=-1)
        mu_target = O_next - O
        # print('mu_target: ', mu_target.shape)

        # Gaussian NLL loss
        Jmu = T.tensor([0.0]) #T.mean(T.mean(T.square(mu - mu_target) * sigma_inv, dim=-1), dim=-1) # batch loss
        Jsigma = T.tensor([0.0]) #T.mean(T.mean(log_sigma, dim=-1), dim=-1)
        # Jgnll = Jmu + Jsigma
        Jgnll = self.gnll_loss(mu, mu_target, sigma)
        Jwl2 = self.weight_l2_loss()
        J = Jgnll + Jwl2

        J += 0.01 * ( T.sum(self.max_log_sigma) - T.sum(self.min_log_sigma) ) # optimize bounds

        return Jmu, Jsigma, J


    # def compute_l2_loss(self, l2_loss_coefs: Union[float, List[float]]):
    def weight_l2_loss(self): # must have 4 hid-layers in the WorldModel
        l2_loss_coefs = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]
        weight_norms = []
        for name, weight in self.named_parameters():
        	if "weight" in name:
        		weight_norms.append(weight.norm(2))
        weight_norms = T.stack(weight_norms, dim=0)
        weight_decay_loss = (T.tensor(l2_loss_coefs, device=weight_norms.device) * weight_norms).sum()
        return weight_decay_loss


    def compute_test_loss(self, batch):
        O, A, R, O_next, D = batch

        mu, log_sigma, sigma, sigma_inv = self(O, A) # dyn_delta, reward
        # mu_target = T.cat([O_next - O, R], dim=-1)
        mu_target = O_next - O

        loss = self.mse_loss(mu, mu_target)

        return loss
