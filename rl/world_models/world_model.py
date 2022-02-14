# TODO: world_model
import random
from typing import Type, List, Tuple

import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


## Fusion
from rl.world_models.model import SimpleModel#, BayesianModel
# from fusion.data.buffer.norm import RunningNormalizer

# from rl.networks.mlp import MLPNet


LOG_SIGMA_MAX = 2
LOG_SIGMA_MIN = -20



class WorldModel(LightningModule):

    def __init__(self, obs_dim, act_dim, rew_dim, configs, seed):
        # print('init World Model!')
        super(WorldModel, self).__init__() # To automatically use 'def forward'
        # Set a random seed, se whenever we call crtics they will be consistent
        # random.seed(seed)###########
        if seed: np.random.seed(seed), T.manual_seed(seed)

        device = self._device_ = configs['experiment']['device']

        self.models = []

        if configs['world_model']['type'] == 'P':
        	self.models = SimpleModel(obs_dim,
        							 act_dim,
        							 rew_dim,
        							 config
        							 ).to(device)
        elif configs['world_model']['type'] == 'PE':
        	M = configs['world_model']['num_ensembles']
        	for m in range(M):
        		model = SimpleModel(obs_dim, act_dim, rew_dim, configs).to(device)
        		self.models.append(model)
        # elif configs['world_model']['type'] == 'BE':
        # 	M = configs['world_model']['Ensembles']
        # 	for m in range(M):
        # 		dyn_model = BayesianModel(obs_dim,
        # 								act_dim,
        # 								rew_dim,
        # 								config
        # 								).to(device)
        # 		self.models.append(dyn_model)

        self.configs = configs
        # print(self.models)


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
    	normed_obs = obs#self.obs_norm(obs)
    	normed_act = act#self.act_norm(act)

    	M = self.configs['world_model']['num_ensembles']
    	modelType = self.configs['world_model']['type']
    	device = self._device_  # self.configs['experiment']['device']

    	if modelType == 'P':
    		prediction, mean, log_std, std, inv_std = self.sample(normed_obs, normed_act, determenistic, sample_type)
    	elif modelType == 'PE':
    		prediction, mean, log_std, std, inv_std = self.sample(normed_obs, normed_act, determenistic, sample_type)

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

        checkpoint_callback = False
        trainer = Trainer(max_epochs=wm_epochs,
                          # log_every_n_steps=5,
                          # gpus=1,
                          weights_summary=None,
                          # checkpoint_callback=False,
                          # progress_bar_refresh_rate=20,
                          # log_save_interval=100,
                          # logger=None, #self.pl_logger,
                          # callbacks=[checkpoint_callback],
                           )
        #
        # max_epochs=mEpochs,
		# 						 gpus=1,
		# 						 weights_summary=ws_summ,
		# 						 checkpoint_callback=False,
		# 						 logger=False,
		# 						 # progress_bar_refresh_rate=0,
		# 						 # log_save_interval=100,
		# 						 # callbacks=[checkpoint_callback]

        if model_type == 'P':
        	Jm, mEpochs = self.models.train_Model(env_buffer, batch_size, 0)
        	# self.models.to(device) # bc pl-training detatchs models
        	# Jwm = Jdyns
        elif model_type == 'PE':
        	J = []
        	for m in range(M):
        		# Jm, mEpochs = self.models[m].train_Model(env_buffer, batch_size, m)
        		Jm = self.models[m].train_Model(trainer, data_module, m)
        		# print(f'modle {m}: Jdyn = {Jdyn}')
        		J.append(Jm)
        		self.models[m].to(device) # bc pl-training detatchs models
        	inx_model = np.argsort(J)
        	self.inx_elites = inx_model[:num_elites]
        	# print(f'Jdyns = {Jdyns}')
        	Jwm = sum(J) / M

        print('Jwm: ', round(Jwm, 5))
        print('Elite Models: ', self.inx_elites)

        return Jwm#, mEpochs
