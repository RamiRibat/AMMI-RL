# TODO: world_model
import random
from typing import Type, List, Tuple

import warnings
warnings.filterwarnings('ignore')

# import logging
# logging.getLogger('lightning').setLevel(0)

import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


## Fusion
from rl.world_models.model import DynamicsModel#, BayesianModel
# from fusion.data.buffer.norm import RunningNormalizer

# T.multiprocessing.set_sharing_strategy('file_system')



LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20


art_zero = 1e-8



class WorldModel(LightningModule):

    def __init__(self, obs_dim, act_dim, rew_dim, configs, seed, device
                # obs_bias=None, obs_scale=None,
                # act_bias=None, act_scale=None,
                # out_bias=None, out_scale=None
                ):
        # print('init World Model!')
        super(WorldModel, self).__init__() # To automatically use 'def forward'

        # device = self._device_ = configs['experiment']['device']
        self._device_ = device
        self.use_decay = use_decay
        self.in_dim = in_dim = obs_dim + act_dim
        self.out_dim = out_dim = obs_dim + rew_dim
        hid_dim = 200
        n_ensemble = 7
        n_elites = 5

        # self.obs_dim = obs_dim
        # self.act_dim = act_dim
        # self.out_dim = obs_dim + rew_dim
        # self.normalization(obs_bias, obs_scale, act_bias, act_scale, out_bias, out_scale)

        if configs['world_model']['type'] == 'P':
        	self.models = DynamicsModel(obs_dim,
        							 act_dim,
        							 rew_dim,
        							 config
        							 ).to(device)
        elif configs['world_model']['type'] == 'PE':
            self.nn1 = LinearEnsemble(n_ensemble, in_dim, hid_dim, weight_decay=0.000025).to(device)
            self.nn2 = LinearEnsemble(n_ensemble, hid_dim, hid_dim, weight_decay=0.00005).to(device)
            self.nn3 = LinearEnsemble(n_ensemble, hid_dim, hid_dim, weight_decay=0.000075).to(device)
            self.nn4 = LinearEnsemble(n_ensemble, hid_dim, hid_dim, weight_decay=0.000075).to(device)
            self.nn5 = LinearEnsemble(n_ensemble, hid_dim, out_dim * 2, weight_decay=0.0001).to(device)
            self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
            self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
            pass

        self.apply(init_weights_)

        self.gnll_loss = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()
        self.activation = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.configs = configs


    # def normalization(self, obs_bias=None, obs_scale=None,
    #                         act_bias=None, act_scale=None,
    #                         out_bias=None, out_scale=None):
    #
    #     device = self._device_
    #
    #     if obs_bias is None:
    #         self.obs_bias   = np.zeros(self.obs_dim)
    #         self.obs_scale  = np.ones(self.obs_dim)
    #         self.act_bias   = np.zeros(self.act_dim)
    #         self.act_scale  = np.ones(self.act_dim)
    #         self.out_bias   = np.zeros(self.out_dim)
    #         self.out_scale  = np.ones(self.out_dim)
    #
    #     # self.obs_bias   = self.obs_bias.to(device)
    #     # self.obs_scale  = self.obs_scale.to(device)
    #     # self.act_bias   = self.act_bias.to(device)
    #     # self.act_scale  = self.act_scale.to(device)
    #     # self.out_bias   = self.out_bias.to(device)
    #     # self.out_scale  = self.out_scale.to(device)
    #     self.mask = self.out_scale >= art_zero


    def sample(self, obs, act, sample_type='Random'):
    	num_elites = self.configs['world_model']['num_elites']

    	if num_elites == 1:
    		mu, log_sigma, sigma, inv_sigma = self.models(obs, act)
    	else:
    		dyn_models_elites = [ self.models[i] for i in self.inx_elites ]
    		dyn_models_outs = [dyn_model(obs, act) for dyn_model in dyn_models_elites]

    		if sample_type == 'Average':
    			# prediction = [dyn_models_outs[i][0] for i in range(len(dyn_models_outs))]
    			# mean = [dyn_models_outs[i][1] for i in range(len(dyn_models_outs))]
    			# log_std = [dyn_models_outs[i][2] for i in range(len(dyn_models_outs))]
    			# std = [dyn_models_outs[i][3] for i in range(len(dyn_models_outs))]
    			# inv_std = [dyn_models_outs[i][4] for i in range(len(dyn_models_outs))]
                #
    			# prediction = sum(prediction) / num_elites
    			# mean = sum(mean) / num_elites
    			# std = sum(std) / num_elites
    			pass

    		elif sample_type == 'Random':
    			indx = random.randint(1, num_elites)
    			mu, log_sigma, sigma, inv_sigma = dyn_models_outs[indx-1]

    		elif sample_type == 'All':
    			# prediction = [dyn_models_outs[i][0] for i in range(len(dyn_models_outs))]
    			# mean = [dyn_models_outs[i][1] for i in range(len(dyn_models_outs))]
    			# log_std = [dyn_models_outs[i][2] for i in range(len(dyn_models_outs))]
    			# std = [dyn_models_outs[i][3] for i in range(len(dyn_models_outs))]
    			# inv_std = [dyn_models_outs[i][4] for i in range(len(dyn_models_outs))]
    			pass

    		elif type(sample_type) == int:
    			# prediction = dyn_models_outs[sample_type][0]
    			# mean = dyn_models_outs[sample_type][1]
    			# log_std = dyn_models_outs[sample_type][2]
    			# std = dyn_models_outs[sample_type][3]
    			# inv_std = dyn_models_outs[sample_type][4]
    			pass

    	return mu, log_sigma, sigma, inv_sigma


    def forward(self, obs, act, deterministic=False, sample_type='Average'):
        nn1_output = self.activation(self.nn1(x))
        nn2_output = self.activation(self.nn2(nn1_output))
        nn3_output = self.activation(self.nn3(nn2_output))
        nn4_output = self.activation(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)
        nn_output = nn5_output

        mean = nn_output[:, :, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - nn_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

        if deterministic:
            prediction = mu
        else:
            normal_ditribution = Normal(mu, sigma)
            prediction = normal_ditribution.sample()

        # obs_next = prediction[:,:-1] + obs
        obs_next = prediction + obs
        rew = 0 #prediction[:,-1]
        # print('obs_next ', obs_next.shape)
        # print('rew ', rew.shape)
        # mu = mu + T.cat([obs, T.zeros(obs.shape[0], 1).to(device)], dim=1) # delta + obs | rew + 0
        mu = mu + obs # delta + obs | rew + 0

        return obs_next, rew, mu, sigma

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


    ### PyTorch Lightning ###
    def train_WM(self, data_module):
        device = self._device_

        M = self.configs['world_model']['num_ensembles']
        model_type = self.configs['world_model']['type']
        num_elites = self.configs['world_model']['num_elites']
        wm_epochs = self.configs['algorithm']['learning']['grad_WM_steps']

        if model_type == 'P':
        	# Jm, mEpochs = self.models.train_Model(env_buffer, batch_size, 0)
            pass
        elif model_type == 'PE':
            JTrain, JVal = [], []
            LossTest = []

            for m in range(M):
                Jtrain, Jval = self.models[m].train_Model(data_module, m)
                # test_loss, wm_mean, wm_sigma = self.models[m].test_Model(data_module)
                test_loss = self.models[m].test_Model(data_module)
                JTrain.append(Jtrain)
                JVal.append(Jval)
                LossTest.append(test_loss)

                self.models[m].to(device) # bc pl-training detatchs models

            inx_model = np.argsort(JVal)
            self.inx_elites = inx_model[:num_elites]

            JTrainLog = sum(JTrain) / M
            JValLog = sum(JVal) / M
            LossTest = sum(LossTest) / M

        print('Elite Models: ', [x+1 for x in self.inx_elites])

        return JTrainLog, JValLog, LossTest
