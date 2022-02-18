# TODO: world_model
import random
from typing import Type, List, Tuple

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('lightning').setLevel(0)

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

    def __init__(self, obs_dim, act_dim, rew_dim, configs, seed,
                # obs_bias=None, obs_scale=None,
                # act_bias=None, act_scale=None,
                # out_bias=None, out_scale=None
                ):
        # print('init World Model!')
        super(WorldModel, self).__init__() # To automatically use 'def forward'
        # Set a random seed, se whenever we call crtics they will be consistent
        # if seed: np.random.seed(seed), T.manual_seed(seed)

        device = self._device_ = configs['experiment']['device']

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
            self.models = [DynamicsModel(obs_dim, act_dim, rew_dim, configs).to(device) for m in range(M)]

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


    def sample(self, obs, act, determenistic, sample_type):
    	num_elites = self.configs['world_model']['num_elites']

    	if num_elites == 1:
    		prediction, mean, log_std, std, inv_std = self.models(obs, act, determenistic)
    	else:
    		dyn_models_elites = [ self.models[i] for i in self.inx_elites ]
    		dyn_models_preds = [dyn_model(obs, act, determenistic) for dyn_model in dyn_models_elites]

    		if sample_type == 'Average':
    			prediction = [dyn_models_preds[i][0] for i in range(len(dyn_models_preds))]
    			mean = [dyn_models_preds[i][1] for i in range(len(dyn_models_preds))]
    			log_std = [dyn_models_preds[i][2] for i in range(len(dyn_models_preds))]
    			std = [dyn_models_preds[i][3] for i in range(len(dyn_models_preds))]
    			inv_std = [dyn_models_preds[i][4] for i in range(len(dyn_models_preds))]

    			prediction = sum(prediction) / num_elites
    			mean = sum(mean) / num_elites
    			std = sum(std) / num_elites

    		elif sample_type == 'Random':
    			indx = random.randint(1, num_elites)
    			prediction, mean, log_std, std, inv_std = dyn_models_preds[indx-1]
    		elif sample_type == 'All':
    			prediction = [dyn_models_preds[i][0] for i in range(len(dyn_models_preds))]
    			mean = [dyn_models_preds[i][1] for i in range(len(dyn_models_preds))]
    			log_std = [dyn_models_preds[i][2] for i in range(len(dyn_models_preds))]
    			std = [dyn_models_preds[i][3] for i in range(len(dyn_models_preds))]
    			inv_std = [dyn_models_preds[i][4] for i in range(len(dyn_models_preds))]
    		elif type(sample_type) == int:
    			prediction = dyn_models_preds[sample_type][0]
    			mean = dyn_models_preds[sample_type][1]
    			log_std = dyn_models_preds[sample_type][2]
    			std = dyn_models_preds[sample_type][3]
    			inv_std = dyn_models_preds[sample_type][4]

    	return prediction, mean, log_std, std, inv_std


    def forward(self, obs, act, determenistic=False, sample_type='Average'):
        # print('obs: ', obs)
        # normed_obs = (obs - self.obs_bias)/(self.obs_scale + art_zero)
        # normed_act = (act - self.act_bias)/(self.act_scale + art_zero)

        M = self.configs['world_model']['num_ensembles']
        modelType = self.configs['world_model']['type']
        device = self._device_  # self.configs['experiment']['device']

        if modelType == 'P':
        	prediction, mean, log_std, std, inv_std = self.sample(obs, act, determenistic, sample_type)
        elif modelType == 'PE':
        	prediction, mean, log_std, std, inv_std = self.sample(obs, act, determenistic, sample_type)

        obs_next = prediction[:,:-1] + obs
        rew = prediction[:,-1]
        # print('xx:', T.cat([obs, T.zeros(obs.shape[0], 1).to(device)], dim=1).shape)
        mean = mean + T.cat([obs, T.zeros(obs.shape[0], 1).to(device)], dim=1)

        # print(f'Models: Mean {mean}, STD {std}')
        return obs_next, rew, mean, std


    ### PyTorch Lightning ###
    def train_WM(self, data_module):
        device = self._device_

        M = self.configs['world_model']['num_ensembles']
        model_type = self.configs['world_model']['type']
        num_elites = self.configs['world_model']['num_elites']
        wm_epochs = self.configs['algorithm']['learning']['grad_WM_steps']

        JTrainLog, JValLog = dict(), dict()

        # checkpoint_callback = False
        # enable_model_summary = False

        if model_type == 'P':
        	Jm, mEpochs = self.models.train_Model(env_buffer, batch_size, 0)
        elif model_type == 'PE':
            JMeanTrain, JTrain = [], []
            JMeanVal, JVal = [], []
            LossTest = []

            for m in range(M):
                train_log, val_log = self.models[m].train_Model(data_module, m)
                test_loss = self.models[m].test_Model(data_module)
                JMeanTrain.append(train_log['mean'])
                JTrain.append(train_log['total'])
                JMeanVal.append(val_log['mean'])
                JVal.append(val_log['total'])
                LossTest.append(test_loss)

                self.models[m].to(device) # bc pl-training detatchs models

            inx_model = np.argsort(JVal)
            self.inx_elites = inx_model[:num_elites]

            JTrainLog['mean'] = sum(JMeanTrain) / M
            JTrainLog['total'] = sum(JTrain) / M
            JValLog['mean'] = sum(JMeanVal) / M
            JValLog['total'] = sum(JVal) / M
            LossTest = sum(LossTest) / M

        print('Elite Models: ', [x+1 for x in self.inx_elites])

        return JTrainLog, JValLog, LossTest
