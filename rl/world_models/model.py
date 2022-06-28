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

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
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

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
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
        # print('input: ', input)
        # print('self.weight: ', self.weight)
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False, device='cpu'):
        super(EnsembleModel, self).__init__()
        self._device_ = device
        # print('EnsembleModel: device: ', device)

        self.hidden_size = hidden_size
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025).to(device)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005).to(device)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075).to(device)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075).to(device)
        self.use_decay = use_decay

        self.output_dim = state_size + reward_size
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001).to(device)

        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        # self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.apply(init_weights_)
        self.swish = Swish()

        # self.gnll_loss = nn.GaussianNLLLoss()
        # self.mse_loss = nn.MSELoss()
        #
        # self.activation = nn.ReLU()

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean = nn5_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # print('mean: ', mean)
        # print('labels: ', labels)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        # print('total_loss: ', total_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        self.optimizer.step()





class EnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False, device='cpu'):
        # print('init EnsembleDynamicsModel')
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
        # print('EnsembleDynamicsModel: device: ', device)

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

        for epoch in range(2):
        # for epoch in itertools.count():
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    print(f"[ Break Model Training ] Epoch: {epoch} | HO MSEs: {[round(x,4) for x in holdout_mse_losses]}"+(" "*10))
                    break
            print(f"[ Model Training ] Epoch: {epoch}, HO MSEs: {[round(x,4) for x in holdout_mse_losses]}"+(" "*10), end='\r')

            # LossList.append(np.mean(losses))

        return np.mean(holdout_mse_losses)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
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
