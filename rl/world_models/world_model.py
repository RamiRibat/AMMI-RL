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
        # Set a random seed, se whenever we call crtics they will be consistent
        # if seed: np.random.seed(seed), T.manual_seed(seed)

        # device = self._device_ = configs['experiment']['device']
        self._device_ = device

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
            M = configs['world_model']['num_ensembles']
            self.models = [DynamicsModel(obs_dim, act_dim, rew_dim, configs, device).to(device) for m in range(M)]
            # self.elit_models = []

        self.configs = configs
        # print(self.models)


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
        # print('obs: ', obs)
        # normed_obs = (obs - self.obs_bias)/(self.obs_scale + art_zero)
        # normed_act = (act - self.act_bias)/(self.act_scale + art_zero)

        M = self.configs['world_model']['num_ensembles']
        modelType = self.configs['world_model']['type']
        device = self._device_

        if modelType == 'P':
        	mu, log_sigma, sigma, inv_sigma = self.sample(obs, act, sample_type)
        elif modelType == 'PE':
        	mu, log_sigma, sigma, inv_sigma = self.sample(obs, act, sample_type)

        if deterministic:
            prediction = mu
        else:
            # normal_ditribution = Normal(mu, sigma)
            # prediction = normal_ditribution.sample()
            normal_ditribution = Normal(mu, T.sqrt(sigma))
            prediction = normal_ditribution.sample()

        # obs_next = prediction[:,:-1] + obs
        obs_next = prediction + obs
        rew = 0 #prediction[:,-1]
        # print('obs_next ', obs_next.shape)
        # print('rew ', rew.shape)
        # mu = mu + T.cat([obs, T.zeros(obs.shape[0], 1).to(device)], dim=1) # delta + obs | rew + 0
        mu = mu + obs # delta + obs | rew + 0

        return obs_next, rew, mu, sigma


    ### PyTorch Lightning ###
    def train_WM(self, data_module):
        device = self._device_

        M = self.configs['world_model']['num_ensembles']
        model_type = self.configs['world_model']['type']
        num_elites = self.configs['world_model']['num_elites']
        wm_epochs = self.configs['algorithm']['learning']['grad_WM_steps']

        if model_type == 'P':
        	Jm, mEpochs = self.models.train_Model(env_buffer, batch_size, 0)
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
