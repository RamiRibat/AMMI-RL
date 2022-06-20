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

# T.multiprocessing.set_sharing_strategy('file_system')

from rl.algorithms.mbrl.mbrl import MBRL
from rl.algorithms.mfrl.sac import SAC
from rl.world_models.fake_world import FakeWorld
import rl.environments.mbpo.static as mbpo_static
# from rl.data.dataset import RLDataModule



class color:
    """
    Source: https://stackoverflow.com/questions/8924173/how-to-print-bold-text-in-python
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'






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
    def __init__(self, exp_prefix, configs, seed, device, wb) -> None:
        super(MBPO, self).__init__(exp_prefix, configs, seed, device)
        # print('init MBPO Algorithm!')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()


    ## build MBPO components: (env, D, AC, alpha)
    def _build(self):
        super(MBPO, self)._build()
        self._set_sac()
        self._set_fake_world()


    ## SAC
    def _set_sac(self):
        SAC._build_sac(self)


    ## FakeEnv
    def _set_fake_world(self):
        env_name = self.configs['environment']['name']
        device = self._device_
        if self.configs['environment']['name'][:4] == 'pddm':
        	static_fns = None
        else:
        	static_fns = mbpo_static[env_name[:-3].lower()]

        # self.fake_world = FakeWorld(self.world_model, static_fns, env_name, self.learn_env, self.configs, device)
        self.fake_world = FakeWorld(self.world_model)


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        G_sac = self.configs['algorithm']['learning']['grad_SAC_steps']

        # batch_size = self.configs['data']['batch_size']

        model_train_frequency = self.configs['world_model']['model_train_freq']
        batch_size_m = self.configs['world_model']['network']['batch_size'] # bs_m
        wm_epochs = self.configs['algorithm']['learning']['grad_WM_steps']
        # real_ratio = self.configs['data']['real_ratio'] # rr
        # batch_size = self.configs['data']['batch_size'] # bs
        # batch_size_ro = self.configs['data']['rollout_batch_size'] # bs_ro

        o, Z, el, t = self.learn_env.reset(), 0, 0, 0
        # o, Z, el, t = self.initialize_learning(NT, Ni)
        # oldJs = [0, 0, 0]
        # JQList, JAlphaList, JPiList = [0], [0], [0]
        # AlphaList = [self.alpha]*Ni

        # JTrainList, JValList, LossTestList = [0], [0], [0]
        # WMList = {'mu': [0]*Ni, 'sigma': [0]*Ni}
        # JMeanTrainList, JTrainList, JMeanValList, JValList = [], [], [], []
        # LossTestList = []
        # WMList = {'mu': [], 'sigma': []}

        logs = dict()
        lastEZ, lastES = 0, -2
        K = 1

        start_time_real = time.time()
        for n in range(1, N+1):
            if self.configs['experiment']['print_logs']:
                print('=' * 50)
                if (n > Nx) and (Nx > 0):
                    print(f'\n[ Epoch {n}   Learning ]'+(' '*50))
                    oldJs = [0, 0, 0]
                    JQList, JAlphaList, JPiList = [], [], []
                    HList = []
                    JTrainList, JValList, LossTestList = [], [], []
                elif (n > Ni) and (Ni > 0):
                    print(f'\n[ Epoch {n}   Exploration + Learning ]'+(' '*50))
                    JQList, JPiList = [], []
                    HList = []
                    JTrainList, JValList, LossTestList = [], [], []
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]'+(' '*50))
                    oldJs = [0, 0, 0]
                    JQList, JAlphaList, JPiList = [0], [0], [0]
                    HList = [0]
                    JTrainList, JValList, LossTestList = [0], [0], [0]

            print(f'[ Replay Buffer ] Size: {self.buffer.size}')

            nt = 0
            ZList, elList = [0], [0]
            ZListImag, elListImag = [0, 0], [0, 0]
            AvgZ, AvgEL = 0, 0

            learn_start_real = time.time()
            while nt < NT: # full epoch
                # Interaction steps
                for e in range(1, E+1):
                    o, Z, el, t = self.internact(n, o, Z, el, t)
                    # print('Return: ', Z)

                    if el > 0:
                        currZ = Z
                        AvgZ = (sum(ZList)+currZ)/(len(ZList))
                        currEL = el
                        AvgEL = (sum(elList)+currEL)/(len(elList))
                    else:
                        lastZ = currZ
                        ZList.append(lastZ)
                        AvgZ = sum(ZList)/(len(ZList)-1)
                        lastEL = currEL
                        elList.append(lastEL)
                        AvgEL = sum(elList)/(len(elList)-1)

                    # print(f'[ Epoch {n}   Interaction ] Env Steps: {e} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}'+(" "*10), end='\r')

                # Taking gradient steps after exploration
                if n > Ni:
                    if nt % model_train_frequency == 0:
                        #03. Train model pθ on Denv via maximum likelihood
                        # PyTorch Lightning Model Training
                        print(f'\n[ Epoch {n} | Training World Model ]'+(' '*50))
                        # print(f'\n\n[ Training ] Dynamics Model(s), mEpochs = {mEpochs}
                        # self.data_module = RLDataModule(self.buffer, self.configs['data'])

                        # JTrainLog, JValLog, LossTest = self.fake_world.train(self.data_module)
                        # JTrainList.append(JTrainLog)
                        # JValList.append(JValLog)
                        # LossTestList.append(LossTest)

                        ho_mean = self.fake_world.train_fake_world(self.buffer)

                        # model_fit_bs = min(self.configs['data']['buffer_size'], self.buffer.size)
                        # model_fit_batch = self.buffer.sample_batch(model_fit_bs, self._device_)
                        # s, a, r, sp, _ = model_fit_batch.values()
                        # if n == Ni+1:
                        #     samples_to_collect = min(Ni*1000, self.buffer.size)
                        # else:
                        #     samples_to_collect = 250
                        #
                        # LossGen = []
                        # for i, model in enumerate(self.models):
                        #     # print(f'\n[ Epoch {n}   Training World Model {i+1} ]'+(' '*50))
                        #     loss_general = model.compute_loss(s[-samples_to_collect:],
                        #                                       a[-samples_to_collect:],
                        #                                       sp[-samples_to_collect:]) # generalization error
                        #     dynamics_loss = model.fit_dynamics(s, a, sp, fit_mb_size=200, fit_epochs=25)
                        #     reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), fit_mb_size=200, fit_epochs=25)
                        # LossGen.append(loss_general)
                        # ho_mean = np.mean(LossGen)

                        JValList.append(ho_mean) # ho: holdout

                        self.reallocate_model_buffer(n)

                        # Generate M k-steps imaginary rollouts for SAC traingin
                        ZListImag, elListImag = self.rollout_world_model(n) # GCP-A
                        # ZListImag, elListImag = self.rollout_world_modelII(n) # Mac/GCP-B

                    # JQList, JPiList = [], []
                    # AlphaList = [self.alpha]*G_sac
                    for g in range(1, G_sac+1): # it was "for g in (1, G_sac+1):" for 2 months, and I did't notice!! ;(
                        # print(f'Actor-Critic Grads...{g}', end='\r')
                        print(f'[ Epoch {n} | Training AC ] Env Steps: {nt+1} | AC Grads: {g} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}'+(" "*10), end='\r')
                        ## Sample a batch B_sac
                        B_sac = self.sac_batch()
                        ## Train networks using batch B_sac
                        Jq, Jalpha, Jpi, PiInfo = self.trainAC(g, B_sac, oldJs)
                        oldJs = [Jq, Jalpha, Jpi]
                        JQList.append(Jq)
                        JPiList.append(Jpi)
                        HList.append(PiInfo['entropy'])
                        if self.configs['actor']['automatic_entropy']:
                            JAlphaList.append(Jalpha.item())
                            AlphaList.append(self.alpha)

                nt += E

            print('\n')
            # logs['time/training                  '] = time.time() - learn_start_real

            # logs['training/wm/Jtrain_mean        '] = np.mean(JMeanTrainList)
            # logs['training/wm/Jtrain             '] = np.mean(JTrainList)
            # logs['training/wm/Jval                    '] = ho_mean
            logs['training/wm/Jval                    '] = np.mean(JValList)
            # logs['training/wm/test_mse           '] = np.mean(LossTestList)

            logs['training/sac/critic/Jq              '] = np.mean(JQList)
            # logs['training/sac/critic/Q(s,a)          '] = T.mean(self.model_buffer.q_buf).item()
            # logs['training/sac/critic/Q-R             '] = T.mean(self.model_buffer.q_buf).item()-T.mean(self.model_buffer.ret_buf).item()

            logs['training/sac/actor/Jpi              '] = np.mean(JPiList)
            logs['training/sac/actor/H                '] = np.mean(HList)
            if self.configs['actor']['automatic_entropy']:
                logs['training/sac/actor/Jalpha           '] = np.mean(JAlphaList)
                logs['training/sac/actor/alpha            '] = np.mean(AlphaList)

            logs['data/env_buffer_size                '] = self.buffer.size
            if hasattr(self, 'model_buffer'):
                logs['data/model_buffer_size              '] = self.model_buffer.size
            else:
                logs['data/model_buffer_size              '] = 0.
            logs['data/rollout_length                 '] = self.set_rollout_length(n)

            logs['learning/real/rollout_return_mean   '] = np.mean(ZList[1:])
            logs['learning/real/rollout_return_std    '] = np.std(ZList[1:])
            logs['learning/real/rollout_length        '] = np.mean(elList[1:])

            # logs['learning/imag/rollout_return_mean   '] = np.mean(ZListImag[1:])
            # logs['learning/imag/rollout_return_std    '] = np.std(ZListImag[1:])
            # logs['learning/imag/rollout_length        '] = np.mean(elListImag[1:])

            eval_start_real = time.time()
            EZ, ES, EL = self.evaluate()

            # logs['time/evaluation                '] = time.time() - eval_start_real

            if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
                logs['evaluation/episodic_score_mean      '] = np.mean(ES)
                logs['evaluation/episodic_score_std       '] = np.std(ES)
            else:
                logs['evaluation/episodic_return_mean     '] = np.mean(EZ)
                logs['evaluation/episodic_return_std      '] = np.std(EZ)
            logs['evaluation/episodic_length_mean     '] = np.mean(EL)
            logs['evaluation/return_to_length         '] = np.mean(EZ)/np.mean(EL)

            logs['time/total                          '] = time.time() - start_time_real

            # if n > (N - 50):
            #     if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
            #         if np.mean(ES) > lastES:
            #             print(f'[ Epoch {n}   Agent Saving ]                    ')
            #             env_name = self.configs['environment']['name']
            #             alg_name = self.configs['algorithm']['name']
            #             T.save(self.actor_critic.actor,
            #             f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pTtar')
            #             lastES = np.mean(ES)
            #     else:
            #         if np.mean(EZ) > lastEZ:
            #             print(f'[ Epoch {n}   Agent Saving ]                    ')
            #             env_name = self.configs['environment']['name']
            #             alg_name = self.configs['algorithm']['name']
            #             T.save(self.actor_critic.actor,
            #             f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pTtar')
            #             lastEZ = np.mean(EZ)

            # Printing logs
            if self.configs['experiment']['print_logs']:
                return_means = ['learning/real/rollout_return_mean   ',
                                'learning/imag/rollout_return_mean   ',
                                'evaluation/episodic_return_mean     ',
                                'evaluation/return_to_length         ']
                for k, v in logs.items():
                    if k in return_means:
                        print(color.RED+f'{k}  {round(v, 4)}'+color.END+(' '*10))
                    else:
                        print(f'{k}  {round(v, 4)}'+(' '*10))

            # WandB
            if self.WandB:
                wandb.log(logs)

        self.learn_env.close()
        self.eval_env.close()


    def rollout_world_modelII(self, n):
        ZListImag, elListImag = [0], [0]

        K = self.set_rollout_length(n)

        # 07. Sample st uniformly from Denv
        device = self._device_
        batch_size_ro = self.configs['data']['rollout_batch_size'] # bs_ro
        batch_size = min(batch_size_ro, self.buffer.size)
        print(f'[ Epoch {n} | Model Rollout ] Batch Size: {batch_size} | Rollout Length: {K}'+(' '*50))
        # B_ro = self.buffer.sample_batch(batch_size) # Torch
        # O = B_ro['observations']

    	# 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
        for m, model in enumerate(self.models):
            B_ro = self.buffer.sample_batch(int(batch_size/4)) # Torch
            O = B_ro['observations']
            for k in range(1, K+1):
                A = self.actor_critic.get_action(O) # Stochastic action | No reparameterization

                # O_next, R, D, _ = self.fake_world.step(O, A) # ip: Tensor, op: Tensor
                O_next = model.forward(O, A).detach() # ip: Tensor, op: Tensor
                R = model.reward(O, A).detach()
                D = self._termination_fn("Hopper-v2", O, A, O_next)
                D = T.tensor(D, dtype=T.bool)

                # self.model_buffer.store_batch(O.numpy(), A, R, O_next, D) # ip: Numpy
                self.model_buffer.store_batch(O, A, R, O_next, D) # ip: Tensor

                O_next = T.Tensor(O_next)
                nonD = ~D.squeeze(-1)

                if nonD.sum() == 0:
                    print(f'[ Epoch {n} | Model Rollout ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}')
                    break

                O = O_next[nonD]

        return ZListImag, elListImag


    def rollout_world_model(self, n):
        ZListImag, elListImag = [0], [0]

        K = self.set_rollout_length(n)

        # 07. Sample st uniformly from Denv
        device = self._device_
        batch_size_ro = self.configs['data']['rollout_batch_size'] # bs_ro
        batch_size = min(batch_size_ro, self.buffer.size)
        print(f'[ Epoch {n}   Model Rollout ] Batch Size: {batch_size} | Rollout Length: {K}'+(' '*50))
        B_ro = self.buffer.sample_batch(batch_size) # Torch
        O = B_ro['observations']

    	# 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
        for k in range(1, K+1):
            A = self.actor_critic.get_action(O) # Stochastic action | No reparameterization

            O_next, R, D, _ = self.fake_world.step(O, A) # ip: Tensor, op: Tensor
            # O_next, R, D, _ = self.fake_world.step_np(O, A) # ip: Tensor, op: Numpy

            # self.model_buffer.store_batch(O.numpy(), A, R, O_next, D) # ip: Numpy
            self.model_buffer.store_batch(O, A, R, O_next, D) # ip: Tensor

            O_next = T.Tensor(O_next)
            D = T.tensor(D, dtype=T.bool)
            # nonD = ~D
            nonD = ~D.squeeze(-1)

            if nonD.sum() == 0:
                print(f'[ Epoch {n}   Model Rollout ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}')
                break

            O = O_next[nonD]

        return ZListImag, elListImag


    def set_rollout_length(self, n):
        if self.configs['world_model']['rollout_schedule'] == None:
        	K = 1
        else:
        	min_epoch, max_epoch, min_length, max_length = self.configs['world_model']['rollout_schedule']

        	if n <= min_epoch:
        		K = min_length
        	else:
        		dx = (n - min_epoch) / (max_epoch - min_epoch)
        		dx = min(dx, 1)
        		K = dx * (max_length - min_length) + min_length

        K = int(K)
        return K


    def sac_batch(self):
        real_ratio = self.configs['data']['real_ratio'] # rr
        batch_size = self.configs['data']['batch_size'] # bs

        batch_size_real = int(real_ratio * batch_size) # 0.05*256
        batch_size_img = batch_size - batch_size_real # 256 - (0.05*256)

        B_real = self.buffer.sample_batch(batch_size_real, self._device_)

        if batch_size_img > 0:
            B_img = self.model_buffer.sample_batch(batch_size_img, self._device_)
            keys = B_real.keys()
            B = {k: T.cat((B_real[k], B_img[k]), dim=0) for k in keys}
        else:
            B = B_real
        return B

    def _reward_fn(self, env_name, obs, act):
        if len(obs.shape) == 1 and len(act.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        elif len(obs.shape) == 1:
            obs = obs[None]
            return_single = True
        else:
            return_single = False

        next_obs = next_obs.numpy()
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(act.shape) == 2
            vel_x = obs[:, -6] / 0.02
            power = np.square(act).sum(axis=-1)
            height = obs[:, 0]
            ang = obs[:, 1]
            alive_bonus = 1.0 * (height > 0.7) * (np.abs(ang) <= 0.2)
            rewards = vel_x + alive_bonus - 1e-3*power

            return rewards
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            pass
        elif 'walker_' in env_name:
            pass


    def _termination_fn(self, env_name, obs, act, next_obs):
        if len(obs.shape) == 1 and len(act.shape) == 1:
            obs = obs[None]
            act = act[None]
            next_obs = next_obs[None]
            return_single = True
        elif len(obs.shape) == 1:
            obs = obs[None]
            next_obs = next_obs[None]
            return_single = True
        else:
            return_single = False

        next_obs = next_obs.numpy()
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done







def main(exp_prefix, config, seed, device, wb):

    print('Start an MBPO experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']
    wm_epochs = configs['algorithm']['learning']['grad_WM_steps']
    DE = configs['world_model']['num_ensembles']

    # group_name = f"{env_name}-{alg_name}-Mac-B"
    group_name = f"{env_name}-{alg_name}-GCP-A"
    exp_prefix = f"seed:{seed}"

    if wb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            project='AMMI-RL-2022',
            config=configs
        )

    agent = MBPO(exp_prefix, configs, seed, device, wb)

    agent.learn()

    print('\n')
    print('... End the MBPO experiment')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-exp_prefix', type=str)
    parser.add_argument('-cfg', type=str)
    parser.add_argument('-seed', type=str)
    parser.add_argument('-device', type=str)
    parser.add_argument('-wb', type=str)

    args = parser.parse_args()

    exp_prefix = args.exp_prefix
    sys.path.append(f"{os.getcwd()}/configs")
    config = importlib.import_module(args.cfg)
    seed = int(args.seed)
    device = args.device
    wb = eval(args.wb)

    main(exp_prefix, config, seed, device, wb)
