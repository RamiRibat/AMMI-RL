import os, subprocess, sys
import argparse
import importlib
import datetime
import random

import time
import wandb

import numpy as np
import torch as T
import torch.nn.functional as F

from rl.algorithms.mbrl.mbrl import MBRL
from rl.algorithms.mfrl.sac import SAC
from rl.world_models.fake_world import FakeWorld
import rl.environments.mbpo.static as mbpo_static



class MBPO(MBRL, SAC):
    """
    Algorithm: Model-Based Policy Optimization (Dyna-style, Model-Based)

        1: Initialize policy πφ, predictive model pθ, environment dataset Denv, model dataset Dmodel
        2: for N epochs do
        3:      Train model pθ on Denv via maximum likelihood
        4:      for E steps do
        5:          Take action in environment according to πφ; add to Denv
        6:          for M model rollouts do
        7:              Sample st uniformly from Denv
        8:              Perform k-step model rollout starting from st using policy πφ; add to Dmodel
        9:          for G gradient updates do
        10:             Update policy parameters on model data: φ ← φ − λπ ˆ∇φ Jπ(φ, Dmodel)

    """
    def __init__(self, exp_prefix, configs, seed) -> None:
        super(MBPO, self).__init__(exp_prefix, configs, seed)
        print('Initialize MBPO Algorithm!')
        self.configs = configs
        self.seed = seed
        self._build()

    ## build MEMB components: (env, D, AC, alpha)
    def _build(self):
        super(MBPO, self)._build()
        self._set_sac()
        self._set_fake_world()

    ## SAC
    def _set_sac(self):
        SAC._build_sac(self)

    ## FakeEnv
    def _set_fake_world(self):
        env_name = self.configs['Environment']['name']
        device = self.configs['Experiment']['device']
        if self.configs['Environment']['name'][:4] == 'pddm':
        	static_fns = None
        else:
        	static_fns = mbpo_static[env_name[:-3].lower()]
        self.fake_env = FakeEnv(self.dyn_models, static_fns, env_name, self.train_env, self.config)




    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']
        # Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        G_sac = self.configs['algorithm']['learning']['grad_SAC_steps']

        # batch_size = self.configs['data']['batch_size']

        model_train_frequency = self.configs['Model']['model_train_freq']
        batch_size_m = self.configs['Model']['Network']['batch_size'] # bs_m
        mEpochs = self.configs['Model']['Network']['mEpochs']
        real_ratio = self.configs['Data']['real_ratio'] # rr
        batch_size = self.configs['Data']['batch_size'] # bs
        batch_size_ro = self.configs['Data']['rollout_batch_size'] # bs_ro

        o, Z, el, t = self.learn_env.reset(), 0, 0, 0
        # o, Z, el, t = self.initialize_learning(NT, Ni)
        oldJs = [0, 0, 0]
        JWMList, JQList, JAlphaList, JPiList = [0]*Ni, [0]*Ni, [0]*Ni, [0]*Ni
        AlphaList = [self.alpha]*Ni
        logs = dict()
        lastEZ, lastES = 0, -2
        K = 1

        start_time_real = time.time()
        for n in range(1, N+1):
            if self.configs['experiment']['print_logs']:
                print('=' * 80)
                if n > Nx:
                    print(f'\n[ Epoch {n}   Learning ]')
                elif n > Ni:
                    print(f'\n[ Epoch {n}   Inintial Exploration + Learning ]')
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]')

            # print(f'[ Replay Buffer ] Size: {self.replay_buffer.size}, pointer: {self.replay_buffer.ptr}')
            nt = 0
            learn_start_real = time.time()
            while nt < NT:
                # Interaction steps
                for e in range(1, E+1):
                    o, Z, el, t = self.internact(n, o, Z, el, t)

                # Taking gradient steps after exploration
                if n > Ni:
                    if nt % model_train_frequency == 0:
                        #03. Train model pθ on Denv via maximum likelihood
                        # PyTorch Lightning Model Training
                        print(f'\n\n[ Epoch {n}   WM Training | mEpochs = {mEpochs}]')
                        # print(f'\n\n[ Training ] Dynamics Model(s), mEpochs = {mEpochs}                                             ')
                        Jwm, mEpochs = self.fake_env.train(self.rl_data_module, model_train_frequency)
                        JWMList.append(Jwm.item())

                        # Update K-steps length
                        K = self.set_rollout_length(n)

                        # Reallocate model buffer
                        # print('Rellocate Model Buffer...')
                        self.reallocate_model_buffer(batch_size_ro, K, NT, model_train_frequency)

                        # Generate M k-steps imaginary rollouts for SAC traingin
                        self.rollout_world_model(batch_size_ro, K)
                        # print(f'[after training] Train env: {self.train_env.env.state_vector()[0]}')
                        # print(f'[after training] Train env: {list(self.actor_critic.actor.parameters())[0][0]}')

                    for g in range(1, G_sac+1): # it was "for g in (1, G_sac+1):" for 2 months, and I did't know!! ;(
                        # print(f'Actor-Critic Grads...{g}', end='\r')
                        print(f'[ Interaction & Training ] Env Steps: {nt}, AC Grads: {g}, Reward: {round(R, 5)}', end='\r')
                        ## Sample a batch B_sac
                        B_sac = self.sac_batch(real_ratio, batch_size)
                        ## Train networks using batch B_sac
                        Jq, Jalpha, Jpi = self.trainAC(g, B_sac, oldJs)
                        oldJs = [Jq, Jalpha, Jpi]
                        JQList.append(Jq.item())
                        JPiList.append(Jpi.item())
                        if self.configs['actor']['automatic_entropy']:
                            JAlphaList.append(Jalpha.item())
                            AlphaList.append(self.alpha)

                nt += E

            logs['time/training                  '] = time.time() - learn_start_real
            logs['training/objectives/sac/Jwm     '] = np.mean(JQList)
            logs['training/objectives/sac/Jq     '] = np.mean(JQList)
            logs['training/objectives/sac/Jpi    '] = np.mean(JPiList)
            if self.configs['actor']['automatic_entropy']:
                logs['training/objectives/sac/Jalpha '] = np.mean(JAlphaList)
                logs['training/objectives/sac/alpha  '] = np.mean(AlphaList)

            eval_start_real = time.time()
            EZ, ES, EL = self.evaluate()

            logs['time/evaluation                '] = time.time() - eval_start_real

            if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
                logs['evaluation/episodic_score_mean '] = np.mean(ES)
                logs['evaluation/episodic_score_std  '] = np.std(ES)
            else:
                logs['evaluation/episodic_return_mean'] = np.mean(EZ)
                logs['evaluation/episodic_return_std '] = np.std(EZ)
            logs['evaluation/episodic_length_mean'] = np.mean(EL)

            logs['time/total                     '] = time.time() - start_time_real

            if n > (N - 50):
                if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
                    if np.mean(ES) > lastES:
                        print(f'[ Epoch {n}   Agent Saving ]                    ')
                        env_name = self.configs['environment']['name']
                        alg_name = self.configs['algorithm']['name']
                        T.save(self.actor_critic.actor,
                        f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pth.tar')
                        lastES = np.mean(ES)
                else:
                    if np.mean(EZ) > lastEZ:
                        print(f'[ Epoch {n}   Agent Saving ]                    ')
                        env_name = self.configs['environment']['name']
                        alg_name = self.configs['algorithm']['name']
                        T.save(self.actor_critic.actor,
                        f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pth.tar')
                        lastEZ = np.mean(EZ)

            # Printing logs
            if self.configs['experiment']['print_logs']:
                for k, v in logs.items():
                    print(f'{k}  {round(v, 2)}')

            # WandB
            if self.configs['experiment']['WandB']:
                wandb.log(logs)

        self.learn_env.close()
        self.eval_env.close()




    def set_rollout_length(self, n):
        if self.configs['Model']['rollout_schedule'] == None:
        	K = 1
        else:
        	min_epoch, max_epoch, min_length, max_length = self.configs['Model']['rollout_schedule']

        	if n <= min_epoch:
        		K = min_length
        	else:
        		dx = (n - min_epoch) / (max_epoch - min_epoch)
        		dx = min(dx, 1)
        		K = dx * (max_length - min_length) + min_length

        K = int(K)
        return K


    def rollout_world_model(self, batch_size_ro, K):
    	#07. Sample st uniformly from Denv
    	device = self.configs['Experiment']['device']
    	batch_size = min(batch_size_ro, self.env_buffer.size)
    	print(f'[ Model Rollout ] Batch Size {batch_size}, Rollout Len {K}                ')
    	B_ro = self.env_buffer.sample_batch(batch_size)
    	O = B_ro['observations']
    	# O_next = B_ro['observations_next']
    	# R = B_ro['rewards']
    	# D = B_ro['terminals']

    	#08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	for k in range(1, K+1):
    		with th.no_grad():
    			A, _ = self.actor_critic.actor(O) # ip:Tensor, op:Tensor

    		# O_next, R, D, _ = self.fake_env.step(O, A) # ip:Tensor, op:Tensor
    		O_next, R, D, _ = self.fake_env.step_np(O, A) # ip:Tensor, op:Numpy

    		O = O.detach().cpu().numpy()
    		A = A.detach().cpu().numpy()
    		# O_next = O_next.detach().cpu().numpy()
    		# R = R.detach().cpu().numpy()
    		# D = th.as_tensor(D, dtype=th.bool).detach().cpu().numpy()

    		self.model_buffer.store_batch(O, A, R, O_next, D) # ip:Numpy
    		# print('model buff ptr: ', self.model_buffer.ptr)

    		nonD = ~D
    		if nonD.sum() == 0:
    		    print(f'[ Model Rollout ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}')
    		    break
    		nonD = np.repeat(nonD, len(O[0,:]), axis=1).reshape(-1, len(O[0,:]))

    		O = O_next[nonD].reshape(-1,len(O[0,:]))
    		O = th.as_tensor(O, dtype=th.float32).to(device)


    def sac_batch(self, real_ratio, batch_size):
    	batch_size_real = int(real_ratio * batch_size) # 0.05*256
    	batch_size_img = batch_size - batch_size_real # 256 - (0.05*256)
    	B_real = self.env_buffer.sample_batch(batch_size_real)

    	if batch_size_img > 0:
    		B_img = self.model_buffer.sample_batch(batch_size_img)
    		keys = B_real.keys()
    		B = {k: th.cat((B_real[k], B_img[k]), dim=0) for k in keys}
    	else:
    		B = B_real
    	return B









def main(exp_prefix, config, seed):

    print('Start an MBPO experiment...')
    print('\n')

    configs = config.configurations

    agent = MBPO(exp_prefix, configs, seed)

    # agent.learn()

    print('\n')
    print('... End the MBPO experiment')

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-exp_prefix', type=str)
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-seed', type=str)

    args = parser.parse_args()

    exp_prefix = args.exp_prefix
    sys.path.append(f"{os.getcwd()}/configs")
    config = importlib.import_module(args.cfg)
    seed = int(args.seed)

    main(exp_prefix, config, seed)
