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
from rl.algorithms.mfrl.ovoq import OVOQ
from rl.dynamics.world_model import WorldModel
from rl.world_models.model import EnsembleDynamicsModel
from rl.world_models.fake_world import FakeWorld
import rl.environments.mbpo.static as mbpo_static


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




class MOVOQ(MBRL, OVOQ):
    """
    Algorithm: Model-based On-policy(V) Off-policy(Q) policy optimization (Dyna-style, Model-Based)

        01: Initialize:
            Models parameters( Policy net πφ0, Value net Vψ0, Quality net Qω0, ensemble of MDP world models {M^θ0}_{1:nM} )
        02. Initialize: Replay Buffer D
        03. Hyperparameters:
        04. for n = 1, 2, ..., N (nEpochs) do
            // Model-Based Off-Policy Gradient (Quality-function)
        05.     for e = 1, 2, ..., E do
        06.         Interact with World by πφn --> {s, a, r, s'}
        07.         Aggregate the Replay Buffer: D = D U {s, a, r, s'}
        08.         Fit dynamics model(s) M^ using all data in the buffer D [with frequency = f]
        09.         Reallocate: Replay Model Buffer D_model (growing size)
        10.         for k = 1, 2, ..., Kq (Off-policy rollout horizon) do
        11.             Rollout transitions {s, a, r, s'}^k from {M^θ}_{1:nM} by πφ starting from p(s0) ~ D
        12.             Aggregate the Replay Model Buffer: D_model = D_model U {s, a, r, s'}^k
        13.         for g = 1, 2, ..., Gq (Off-policy PG updates) do
        14.             Update Off-policy Actor-Critic (Qω, πφ; D_model)
            // Model-Based On-Policy Gradient (Value-function)
        15.     Fit dynamics model(s) M^ using recent data B in the buffer D [B << D_max]
        16.     Initialize: Trajectory Model Buffer Dτ_model (fixed size)
        17.     for k = 1, 2, ..., Kv (On-policy rollout horizon) do
        18.         Rollout a traj {τ_π/M^}^k from {M^θ}_{1:nM} by πφ starting from p(s0) ~ B (recent data of D)
        19.         Aggregate the Trajectory Model Buffer: Dτ_model = Dτ_model U {τ}^k
        20.     for g = 1, 2, ..., Gv:
        21.         Update On-policy Actor-Critic (Vψ, πφ; Dτ_model)

    """

    def __init__(self, exp_prefix, configs, seed, device, wb) -> None:
        super(MOVOQ, self).__init__(exp_prefix, configs, seed, device)
        print('Initialize MOVOQ Algorithm!')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()


    def _build(self):
        super(MOVOQ, self)._build()
        self._set_ovoq()
        self._set_fake_world() # OQ
        self._set_ov_world_model()
        self.init_model_traj_buffer() # OV


    def _set_ovoq(self):
        OVOQ._build(self)


    def _set_ov_world_model(self):
        device = self._device_
        num_ensembles = self.configs['world_model']['ov_ensembles']
        # num_elites = self.configs['world_model']['num_elites']
        net_arch = self.configs['world_model']['network']['arch']

        self.ov_world_model = [ WorldModel(self.obs_dim, self.act_dim, seed=0+m, device=device) for m in range(num_ensembles) ]


    def _set_oq_world_model(self):
        device = self._device_
        num_ensembles = self.configs['world_model']['oq_ensembles']
        num_elites = self.configs['world_model']['oq_elites']
        net_arch = self.configs['world_model']['network']['arch']
        self.oq_world_model = EnsembleDynamicsModel(num_ensembles, num_elites,
                                                    self.obs_dim, self.act_dim, 1,
                                                    net_arch[0], use_decay=True, device=device)


    def _set_fake_world(self):
        self._set_oq_world_model()
        env_name = self.configs['environment']['name']
        device = self._device_
        if self.configs['environment']['name'][:4] == 'pddm':
        	static_fns = None
        else:
        	static_fns = mbpo_static[env_name[:-3].lower()]

        # self.fake_world = FakeWorld(self.oq_world_model, static_fns, env_name, self.learn_env, self.configs, device)
        self.fake_world = FakeWorld(self.oq_world_model)


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Niv = self.configs['algorithm']['learning']['ov_init_epochs']
        Niq = self.configs['algorithm']['learning']['oq_init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        GV = self.configs['algorithm']['learning']['grad_OV_steps']
        GPPO = self.configs['algorithm']['learning']['grad_PPO_steps']
        GQ = self.configs['algorithm']['learning']['grad_OQ_SAC_steps']
        max_dev = self.configs['actor']['max_dev']

        oq_model_train_freq = self.configs['world_model']['oq_model_train_freq']
        # ov_model_train_freq = self.configs['world_model']['ov_model_train_freq']

        global_step = 0
        start_time = time.time()
        logs = dict()
        lastEZ, lastES = 0, -2
        t = 0

        # o, Z, el, t = self.learn_env.reset(), 0, 0, 0

        start_time_real = time.time()
        for n in range(1, N+1):

            if self.configs['experiment']['print_logs']:
                print('=' * 50)
                if n > Nx:
                    if n > Niv:
                        if n > Niq:
                            print(f'\n[ Epoch {n}   Learning (OV+OQ) ]'+(' '*50))
                            JVList, JQList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [], [], [], [], [], []
                            HVList, HQList, DevList = [], [], []
                        else:
                            print(f'\n[ Epoch {n}   Learning (OV) ]'+(' '*50))
                            JVList, JQList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [0], [], [0], [], [], [0]
                            HVList, HQList, DevList = [], [0], []
                    oldJs = [0, 0, 0, 0]
                elif n > Niv:
                    if n > Niq:
                        print(f'\n[ Epoch {n}   Exploration + Learning (OV+OQ) ]'+(' '*50))
                        JVList, JQList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [], [], [], [], [], []
                        HVList, HQList, DevList = [], [], []
                    else:
                        print(f'\n[ Epoch {n}   Exploration + Learning (OV) ]'+(' '*50))
                        JVList, JQList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [0], [], [0], [], [], [0]
                        HVList, HQList, DevList = [], [0], []
                    oldJs = [0, 0, 0, 0]
                elif n > Niq:
                    if n > Niv:
                        print(f'\n[ Epoch {n}   Exploration + Learning (OV+OQ) ]'+(' '*50))
                        JVList, JQList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [], [], [], [], [], []
                        HVList, HQList, DevList = [], [], []
                    else:
                        print(f'\n[ Epoch {n}   Exploration + Learning (OQ) ]'+(' '*50))
                        JVList, JQList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [0], [], [0], [], [0], [0], []
                        HVList, HQList, DevList = [0], [], [0]
                    oldJs = [0, 0, 0, 0]
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]'+(' '*50))
                    JVList, JQList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [0], [0], [0], [0], [0], [0], [0]
                    HVList, HQList, DevList = [0], [0], [0]
                    oldJs = [0, 0, 0, 0]

            nt = 0
            o, d, Z, el = self.learn_env.reset(), 0, 0, 0
            ZList, elList = [0], [0]
            ZListImag, elListImag = [0, 0], [0, 0]
            AvgZ, AvgEL = 0, 0
            ppo_grads, sac_grads = 0, 0

            # if n > Niq:
            #     on_policy = False if (n%2==0) else True
            #     OP = 'on-policy' if on_policy else 'off-policy'
            # else:
            #     on_policy = True
            #     OP = 'on-policy'

            on_policy = False
            OP = 'off-policy'

            learn_start_real = time.time()
            while nt < NT: # full epoch
                # Interaction steps
                for e in range(1, E+1):
                    o, Z, el, t = self.internact_ovoq(n, o, Z, el, t, on_policy=on_policy)
                    # o, Z, el, t = self.internact_ovoq(n, o, Z, el, t, on_policy=True)
                    # o, Z, el, t = self.internact_ovoq(n, o, Z, el, t, on_policy=False)

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

                    # print(f'[ Epoch {n}   Interaction ] Env Steps: {el} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}'+(" "*10), end='\r')

                # Taking gradient steps after exploration
                if n > Niq:
                    # 03. Train model pθ on Denv via maximum likelihood
                    if nt % oq_model_train_freq == 0: # 250
                        print(f'\n[ Epoch {n} | {color.BLUE}Training Q-World Model{color.END} ]'+(' '*50))
                        hoq_mean = self.fake_world.train_fake_world(self.buffer)

                        JHOQList.append(hoq_mean) # ho: holdout

                        self.reallocate_oq_model_buffer(n)

                        self.rollout_oq_world_model(n)

                    # SAC >>>>
                    for gq in range(1, GQ+1):
                        print(f'[ Epoch {n} | {color.BLUE}Training AQ{color.END} ] Env Steps ({OP}): {nt+1} | GQ Grads: {gq}/{GQ} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)} | x{round(AvgZ/AvgEL, 2)}'+(" "*10), end='\r')
                        ## Sample a batch B_sac
                        B_OQ = self.sample_oq_batch()
                        ## Train networks using batch B_sac
                        _, Jq, Jalpha, Jpi, PiInfo = self.trainAC(gq, B_OQ, oldJs, on_policy=False)
                        oldJs = [0, Jq, Jalpha, Jpi]
                        JQList.append(Jq)
                        JPiQList.append(Jpi)
                        HQList.append(PiInfo['entropy'])
                        if self.configs['actor']['automatic_entropy']:
                            JAlphaList.append(Jalpha.item())
                            AlphaList.append(self.alpha)
                        sac_grads += 1
                    # SAC <<<<

                nt += E

            with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
            self.buffer.finish_path(el, v)

            if n > Niv:
                print(f'\n\n[ Epoch {n} | {color.RED}Training V-World Model{color.END} ]'+(' '*50))
                grad_MV_steps = self.configs['algorithm']['learning']['grad_MV_steps']

                model_fit_bs = min(self.configs['data']['recent_buffer_size'], self.buffer.total_size())
                model_fit_batch = self.buffer.sample_batch(batch_size=model_fit_bs, recent=model_fit_bs, device=self._device_)
                s, a, sp, r, _, _, _, _, _ = model_fit_batch.values()
                if n == Niv+1:
                    samples_to_collect = min((Niv+1)*1000, self.buffer.total_size())
                else:
                    samples_to_collect = 1000

                LossGen = []
                for i, model in enumerate(self.ov_world_model):
                    loss_general = model.compute_loss(s[-samples_to_collect:],
                                                      a[-samples_to_collect:],
                                                      sp[-samples_to_collect:]) # generalization error
                    dynamics_loss = model.fit_dynamics(s, a, sp, fit_mb_size=200, fit_epochs=grad_MV_steps)
                    reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), fit_mb_size=200, fit_epochs=grad_MV_steps)
                LossGen.append(loss_general)
                hov_mean = np.mean(LossGen)
                JHOVList.append(hov_mean)

                # PPO >>>>
                for gv in range(1, GV+1):
                    # Reset model buffer
                    self.model_traj_buffer.reset()
                    # Generate M k-steps imaginary rollouts for PPO training
                    ZListImag, elListImag = self.rollout_ov_world_model_trajectories(gv, n)
                    ppo_batch_size = int(self.model_traj_buffer.total_size())
                    kl, dev = 0, 0
                    stop_pi = False
                    for gg in range(1, GPPO+1): # 101
                        print(f"[ Epoch {n} | {color.RED}Training AV{color.END} ] GV: {gv}/{GV} | ac: {gg}/{GPPO} || stopPG={stop_pi} | Dev={round(dev, 4)}"+(" "*40), end='\r')
                        batch = self.model_traj_buffer.sample_batch(batch_size=ppo_batch_size, device=self._device_)
                        Jv, _, _, Jpi, PiInfo = self.trainAC(gv, batch, oldJs, on_policy=True)
                        oldJs = [Jv, 0, 0, Jpi]
                        JVList.append(Jv)
                        JPiVList.append(Jpi)
                        KLList.append(PiInfo['KL'])
                        HVList.append(PiInfo['entropy'])
                        DevList.append(PiInfo['deviation'])
                        dev = PiInfo['deviation']
                        if not PiInfo['stop_pi']:
                            ppo_grads += 1
                        stop_pi = PiInfo['stop_pi']
                # PPO <<<<


            print('\n')

            logs['training/world_model/Jhov           '] = np.mean(JHOVList)
            logs['training/world_model/Jhoq           '] = np.mean(JHOQList)

            logs['training/ovoq/critic/Jv             '] = np.mean(JVList)
            logs['training/ovoq/critic/V(s)           '] = T.mean(self.model_traj_buffer.val_buf).item()
            logs['training/ovoq/critic/V-R            '] = T.mean(self.model_traj_buffer.val_buf).item()-T.mean(self.model_traj_buffer.ret_buf).item()
            logs['training/ovoq/critic/Jq             '] = np.mean(JQList)

            logs['training/ovoq/actor/Jpi_ov          '] = np.mean(JPiVList)
            logs['training/ovoq/actor/Jpi_oq          '] = np.mean(JPiQList)
            logs['training/ovoq/actor/HV              '] = np.mean(HVList)
            logs['training/ovoq/actor/HQ              '] = np.mean(HQList)
            # logs['training/ovoq/actor/ov-KL           '] = np.mean(KLList) #
            logs['training/ovoq/actor/ov-deviation    '] = np.mean(DevList)
            logs['training/ovoq/actor/ppo-grads       '] = ppo_grads
            logs['training/ovoq/actor/sac-grads       '] = sac_grads

            logs['data/real/on-policy                 '] = int(on_policy)
            logs['data/real/buffer_size               '] = self.buffer.total_size()
            if hasattr(self, 'model_traj_buffer') and hasattr(self, 'model_repl_buffer'):
                logs['data/imag/ov-traj_buffer_size       '] = self.model_traj_buffer.total_size()
                logs['data/imag/oq-repl_buffer_size       '] = self.model_repl_buffer.size
            elif hasattr(self, 'model_traj_buffer'):
                logs['data/imag/ov-traj_buffer_size       '] = self.model_traj_buffer.total_size()
                logs['data/imag/oq-repl_buffer_size       '] = 0.
            elif hasattr(self, 'model_repl_buffer'):
                logs['data/imag/ov-traj_buffer_size       '] = 0.
                logs['data/imag/oq-repl_buffer_size       '] = self.model_repl_buffer.size
            else:
                logs['data/imag/ov-traj_buffer_size       '] = 0.
                logs['data/imag/oq-repl_buffer_size       '] = 0.

            logs['learning/real/rollout_return_mean   '] = np.mean(ZList[1:])
            logs['learning/real/rollout_return_std    '] = np.std(ZList[1:])
            logs['learning/real/rollout_length        '] = np.mean(elList[1:])

            logs['learning/ov-img/rollout_return_mean '] = np.mean(ZListImag[1:])
            logs['learning/ov-img/rollout_return_std  '] = np.std(ZListImag[1:])
            logs['learning/ov-img/rollout_length      '] = np.mean(elListImag[1:])

            eval_start_real = time.time()
            EZ, ES, EL = self.evaluate()

            if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
                logs['evaluation/episodic_score_mean      '] = np.mean(ES)
                logs['evaluation/episodic_score_std       '] = np.std(ES)
            else:
                logs['evaluation/episodic_return_mean     '] = np.mean(EZ)
                logs['evaluation/episodic_return_std      '] = np.std(EZ)
            logs['evaluation/episodic_length_mean     '] = np.mean(EL)
            logs['evaluation/return_to_length         '] = np.mean(EZ)/np.mean(EL)
            logs['evaluation/return_to_full_length    '] = (np.mean(EZ)/1000)

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
                                'learning/ov-img/rollout_return_mean ',
                                'evaluation/episodic_return_mean     ',
                                'evaluation/return_to_length         ',
                                'evaluation/return_to_full_length    ']
                for k, v in logs.items():
                    if k in return_means:
                        print(color.PURPLE+f'{k}  {round(v, 4)}'+color.END+(' '*10))
                    else:
                        print(f'{k}  {round(v, 4)}'+(' '*10))

            # WandB
            if self.WandB:
                wandb.log(logs)

        self.learn_env.close()
        self.eval_env.close()


    def rollout_oq_world_model(self, n):
        # ZListImag, elListImag = [0], [0]

        K = self.set_oq_rollout_length(n)
        buffer_size = self.buffer.total_size()

        # 07. Sample st uniformly from Denv
        device = self._device_
        batch_size_ro = self.configs['data']['oq_rollout_batch_size'] # bs_ro
        batch_size = min(batch_size_ro, buffer_size)
        print(f'[ Epoch {n} | {color.BLUE}Q-Model Rollout{color.END} ] Batch Size: {batch_size} | Rollout Length: {K}'+(' '*50))
        B_ro = self.buffer.sample_batch(batch_size) # Torch
        O = B_ro['observations']

    	# 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
        for k in range(1, K+1):
            A = self.actor_critic.get_action(O, on_policy=False) # Stochastic action | No reparameterization

            O_next, R, D, _ = self.fake_world.step(O, A) # ip: Tensor, op: Tensor
            self.model_repl_buffer.store_batch(O, A, R, O_next, D) # ip: Tensor

            O_next = T.Tensor(O_next)
            D = T.tensor(D, dtype=T.bool)
            nonD = ~D.squeeze(-1)

            if nonD.sum() == 0:
                print(f'[ Epoch {n} | {color.BLUE}Q-Model Rollout{color.END} ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}')
                break

            O = O_next[nonD]

        # return ZListImag, elListImag


    def rollout_ov_world_model_trajectories(self, g, n):
    	# 07. Sample st uniformly from Denv
    	device = self._device_
    	Nτ = 250
    	K = 1000

    	O = O_init = self.buffer.sample_init_obs_batch(Nτ)
    	O_Nτ = len(O_init)

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	k_end_total = 0
    	ZList, elList = [0], [0]
    	AvgZ, AvgEL = 0, 0

    	for nτ, oi in enumerate(O_init): # Generate trajectories
            for m, model in enumerate(self.ov_world_model):
                o, Z, el = oi, 0, 0
                for k in range(1, K+1): # Generate rollouts
                    print(f'[ Epoch {n} | {color.RED}AV {g} | V-Model Rollout{color.END} ] nτ = {nτ+1} | M = {m+1}/{len(self.ov_world_model)} | k = {k}/{K} | Buffer = {self.model_traj_buffer.total_size()} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}', end='\r')
                    with T.no_grad(): a, log_pi, _, v = self.actor_critic.get_a_and_v(o)

                    o_next = model.forward(o, a).detach().cpu() # ip: Tensor, op: Tensor
                    r = model.reward(o, a).detach()
                    d = self._termination_fn("Hopper-v2", o, a, o_next)
                    d = T.tensor(d, dtype=T.bool)

                    Z += float(r)
                    el += 1
                    self.model_traj_buffer.store(o, a, r, o_next, v, log_pi, el)
                    o = o_next

                    currZ = Z
                    AvgZ = (sum(ZList)+currZ)/(len(ZList))
                    currEL = el
                    AvgEL = (sum(elList)+currEL)/(len(elList))

                    if d or (el == K):
                        break

                if el == K:
                    with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
                else:
                    v = T.Tensor([0.0])
                self.model_traj_buffer.finish_path(el, v)

                k_end_total += k

                lastZ = currZ
                ZList.append(lastZ)
                AvgZ = sum(ZList)/(len(ZList)-1)
                lastEL = currEL
                elList.append(lastEL)
                AvgEL = sum(elList)/(len(elList)-1)

            if self.model_traj_buffer.total_size() >= self.configs['data']['ov_model_buffer_size']:
                # print(f'Breaking img rollouts at nτ={nτ+1}/m={m+1} | Buffer = {self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*85)
                break
        print(f'[ Epoch {n} | AC {g} ] RollBuffer={self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | L={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*35)

    	return ZList, elList


    def set_oq_rollout_length(self, n):
        if self.configs['world_model']['oq_rollout_schedule'] == None:
        	K = 1
        else:
        	min_epoch, max_epoch, min_length, max_length = self.configs['world_model']['oq_rollout_schedule']

        	if n <= min_epoch:
        		K = min_length
        	else:
        		dx = (n - min_epoch) / (max_epoch - min_epoch)
        		dx = min(dx, 1)
        		K = dx * (max_length - min_length) + min_length

        K = int(K)
        return K


    def sample_oq_batch(self):
        real_ratio = self.configs['data']['oq_real_ratio'] # rr
        batch_size = self.configs['data']['oq_batch_size'] # bs

        batch_size_real = int(real_ratio * batch_size) # 0.05*256
        batch_size_img = batch_size - batch_size_real # 256 - (0.05*256)

        B_real = self.buffer.sample_batch_for_reply(batch_size=batch_size_real, device=self._device_)

        if batch_size_img > 0:
            B_img = self.model_repl_buffer.sample_batch(batch_size=batch_size_img, device=self._device_)
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

    print('Start an MOVOQ experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']

    group_name = f"{env_name}-{alg_name}-Mac-2"
    # group_name = f"{env_name}-{alg_name}-GCP-A"
    # group_name = f"{env_name}-{alg_name}-OQ-GCP-A"
    # group_name = f"{env_name}-{alg_name}-OV-GCP-A"
    exp_prefix = f"seed:{seed}"

    if wb:
        # print('WandB')
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            # project='AMMI-RL-2022',
            project=f'AMMI-RL-{env_name}',
            config=configs
        )

    agent = MOVOQ(exp_prefix, configs, seed, device, wb)

    agent.learn()

    print('\n')
    print('... End the MOVOQ experiment')


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
