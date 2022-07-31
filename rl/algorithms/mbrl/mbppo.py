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
from rl.algorithms.mfrl.ppo import PPO
from rl.dynamics.world_model import WorldModel
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





class MBPPO(MBRL, PPO):
    """
    Greek alphabet:
        Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, Μ μ,
        Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ/ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω.

    Algorithm: Model-Based Game: Policy As Leader (PAL) – Practical Version

        01: Initialize: Models parameters( Policy net πφ0, Value net Vψ0, ensemble of MDP world models {M^θ0}_{1:nM} )
        02. Initialize: Replay buffer D.
        03: Hyperparameters: Initial samples Ni = 2 epochs, samples per update E=1000, buffer size B = 2500 ≈ N, number of NPG steps K = 4 ≈ 1
        04: Initial Data: Collect N_init samples from the environment by interacting with initial policy. Store data in buffer D.
        05: for n = 0, 1, 2, ..., N (nEpochs) do
        06:    Learn dynamics model(s) M^_n+1 using data in the buffer.
        07:    Policy updates: π_n+1; V_n+1 = MB-PPO(π_n, V_n, M^_n+1) // call K times
        08:    Collect dataset of N samples from World by interacting with πn+1.
        09.    Add data to replay buffer D, discarding old data if size is larger than B.
        10: end for


    Algorithm: Model-Based Game: Model As Leader (MAL) – Practical Version
        1: Initialize: Policy network π_0, model network(s) Mhat_0, value network V_0.
        2: Hyperparameters: Initial samples N_init, samples per update N, number of PPO steps K >> 1
        3: Initial Data: Collect N_init samples from the environment by interacting with initial policy. Store data in buffer D.
        4: Initial Model: Learn model(s) Mhat_0 using data in D.
        5: for k = 0, 1, 2, ... do
        6:    Optimize π_k+1 using Mck by running K >> 1 steps of model-based PPO (Subroutine 1).
        7:    Collect dataset D_k+1 of N samples from world using πk+1. Aggregate data D = D U D_k+1.
        8:    Learn dynamics model(s) Mhat_k+1 using data in D.
        9: end for


    Algorithm: Model-Based Proximal Policy Optimization (On-Policy, Dyna-style, Model-Based)

        01. Inputs: Models parameters( Policy net πφn, Value net Vψn, ensemble of MDP world models {M^θn+1}_{1:nM}, Replay buffer D)
        02. Hyperparameters: Disc. factor γ, GAE λ, num traj's Nτ, model rollout horizon H
        03. Initialize: Trajectory buffer Dτ = {}
        04. for k = 0, 1, 2, ..., Nτ:
        05.     Sample init p(s^k_0) from distribution/buffer
        05.     Rollout a traj {τ^k_π/M} from {M^θ}_{1:nM} by πn = π(φn) for e = 0, 1, 2, ..., H
        06.     Aggregate traj τ^k in the traj buffer, Dτ = Dτ U {τ^k_π/M}
        07.     Compute RTG R^_t, GAE A^_t based on Vn = V(θn)
        08. end for
        09. Update πφ by maxz Jπ
                φ = arg max_φ {(1/T|Dk|) sum sum min((π/πk), 1 +- eps) Aπk }
        10. Fit Vθ by MSE(Jv)
                θ = arg min_θ {(1/T|Dk|) sum sum (Vθ(st) - RTG)^2 }
        11. Return: Policy net πθn+1, value net Vφn+1

    Subroutine 1: Model-Based Natural Policy Gradient Update Step

        1: Require: Policy (stochastic) network πθ, value/baseline network V , ensemble of MDP dynamics models fMcφg,
        reward function R, initial state distribution or buffer.
        2: Hyperparameters: Discount factor γ, GAE λ, number of trajectories Nτ, rollout horizon H, normalized NPG step
        size δ
        3: Initialize trajectory buffer Dτ = fg
        4: for k = 1; 2; : : : ; Nτ do
        5: Sample initial state sk 0 from initial state distribution/buffer
        6: Perform H step rollout from sk 0 with πθ to get τjk = (sk 0; ak 0; sk 1; ak 2; : : : sk H; ak H), one for each model Mcφj in the
        ensemble.
        7: Query reward function to obtain rewards for each step of the trajectories
        8: Truncate trajectories if termination/truncation conditions are part of the environment
        9: Aggregate the trajectories in trajectory buffer, Dτ = Dτ [ fτg
        10: end for
        11: Compute advantages for each trajectory using V and GAE (Schulman et al., 2016).
        12: Compute vanilla policy gradient using the dataset
        g = E(s;a)∼Dτ [rθ log πθ(ajs)Aπ(s; a)]
        13: Perform normalized NPG update (F denotes the Fisher matrix)
        θ = θ + sgT Fδ−1g F −1g
        14: Update value/baseline network V to fit the computed returns in Dτ.
        15: Return Policy network πθ, value network V

    """
    def __init__(self, exp_prefix, configs, seed, device, wb) -> None:
        super(MBPPO, self).__init__(exp_prefix, configs, seed, device)
        # print('init MBPPO Algorithm!')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()


    ## build MBPPO components: (env, D, AC, alpha)
    def _build(self):
        super(MBPPO, self)._build()
        self._set_ppo()
        self._set_ov_world_model()
        self.init_model_traj_buffer()


    ## PPO
    def _set_ppo(self):
        PPO._build_ppo(self)


    def _set_ov_world_model(self):
        device = self._device_
        num_ensembles = self.configs['world_model']['num_ensembles']
        # num_elites = self.configs['world_model']['num_elites']
        net_arch = self.configs['world_model']['network']['arch']

        self.models = [ WorldModel(self.obs_dim, self.act_dim, seed=0+m, device=device) for m in range(num_ensembles) ]


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        G_AC = self.configs['algorithm']['learning']['grad_AC_steps']
        G_PPO = self.configs['algorithm']['learning']['grad_PPO_steps']
        max_dev = self.configs['actor']['max_dev']

        global_step = 0
        # o, d, Z, el = self.learn_env.reset(), 0, 0, 0
        start_time = time.time()
        logs = dict()
        lastEZ, lastES = 0, -2
        t = 0

        start_time_real = time.time()
        for n in range(1, N+1):
            if self.configs['experiment']['print_logs']:
                print('=' * 50)
                if n > Nx:
                    print(f'\n[ Epoch {n}   Learning ]'+(' '*50))
                    oldJs = [0, 0]
                    JVList, JPiList, KLList = [], [], []
                    HList, DevList = [], []
                    ho_mean = 0
                elif n > Ni:
                    print(f'\n[ Epoch {n}   Exploration + Learning ]'+(' '*50))
                    JVList, JPiList, KLList = [], [], []
                    HList, DevList = [], []
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]'+(' '*50))
                    oldJs = [0, 0]
                    JVList, JPiList, KLList = [0], [0], [0]
                    HList, DevList = [0], [0]
                    ho_mean = 0

            nt = 0
            o, d, Z, el, = self.learn_env.reset(), 0, 0, 0
            ZList, elList = [0], [0]
            ZListImag, elListImag = [0, 0], [0, 0]
            AvgZ, AvgEL = 0, 0
            # ac_grads = 0
            ppo_grads = 0

            learn_start_real = time.time()
            while nt < NT: # full epoch
                # Interaction steps
                for e in range(1, E+1):
                    # o, Z, el, t = self.internact(n, o, Z, el, t)
                    o, Z, el, t = self.internact_opB(n, o, Z, el, t, return_pre_pi=False)
                    # o, Z, el, t = self.internactII(n, o, Z, el, t)

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

                    print(f'[ Epoch {n}   Interaction ] Env Steps: {e} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}'+(" "*10), end='\r')
                with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
                # self.buffer.traj_tail(d, v, el)
                self.buffer.finish_path(el, v)

                # print(f'Buffer size = {self.buffer.total_size()}')
                # Taking gradient steps after exploration
                if n > Ni:
                    # 03. Train model pθ on Denv via maximum likelihood
                    print(f'\n[ Epoch {n} | Training World Model ]'+(' '*50))

                    # ho_mean = self.fake_world.train_fake_world(self.buffer)

                    model_fit_bs = min(self.configs['data']['buffer_size'], self.buffer.total_size())
                    model_fit_batch = self.buffer.sample_batch(batch_size=model_fit_bs, device=self._device_)
                    s, _, a, sp, r, _, _, _, _, _ = model_fit_batch.values()
                    if n == Ni+1:
                        samples_to_collect = min((Ni+1)*1000, self.buffer.total_size())
                    else:
                        samples_to_collect = 1000

                    LossGen = []
                    for i, model in enumerate(self.models):
                        # print(f'\n[ Epoch {n}   Training World Model {i+1} ]'+(' '*50))
                        loss_general = model.compute_loss(s[-samples_to_collect:],
                                                          a[-samples_to_collect:],
                                                          sp[-samples_to_collect:]) # generalization error
                        dynamics_loss = model.fit_dynamics(s, a, sp, fit_mb_size=200, fit_epochs=25)
                        reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), fit_mb_size=200, fit_epochs=25)
                    LossGen.append(loss_general)
                    ho_mean = np.mean(LossGen)

                    # self.init_model_traj_buffer()

                    # # PPO-P >>>>
                    # # GG = [40, 60, 80, 100, 120]
                    # GG = [ int((1.045-0.009*n)*g) for g in [40, 60, 80, 100, 120] ]
                    # # print('GG: ', GG)
                    # # KLtargs = [0.045, 0.04, 0.035, 0.03, 0.025]
                    # KLtargs = [ int((kl-0.01*(kl-0.02)*(n-5))) for kl in [0.06, 0.055, 0.05, 0.045, 0.04] ]
                    # for g in range(1, G_ppo+1):
                    #     # Reset model buffer
                    #     self.model_traj_buffer.reset()
                    #     # # Generate M k-steps imaginary rollouts for PPO training
                    #     k_avg = self.rollout_real_world_trajectories(g, n)
                    #     # k_avg = self.rollout_world_model_trajectories(g, n)
                    #     # self.rollout_world_model_trajectories_batch(g, n)
                    #     # k_avg = self.rollout_world_model_trajectoriesII(g, n)
                    #
                    #     # batch_size = int(self.model_traj_buffer.total_size())
                    #     batch_size = min(int(self.model_traj_buffer.total_size()), 4000)
                    #     stop_pi = False
                    #     kl = 0
                    #     print(f'\n\n[ Epoch {n}   Training Actor-Critic ({g}) ] Model Buffer: Size={self.model_traj_buffer.total_size()} | AvgK={self.model_traj_buffer.average_horizon()}'+(" "*25)+'\n')
                    #     for gg in range(1, 60+1): # 101
                    #         print(f'[ Epoch {n} ] AC: {g} | ac: {gg}/{GG[g-1]} || stopPG={stop_pi} | KL={round(kl, 4)}'+(' '*40), end='\r')
                    #         batch = self.model_traj_buffer.sample_batch(batch_size, self._device_)
                    #         Jv, Jpi, kl, stop_pi = self.trainAC(g, batch, oldJs, 0.04)
                    #         # print(f'[ Epoch {n} ] AC: {g} | ac: {gg} || PG={stop_pi} | KL={kl}'+(' '*80), end='\r')
                    #         oldJs = [Jv, Jpi]
                    #         JVList.append(Jv.item())
                    #         JPiList.append(Jpi.item())
                    #         KLList.append(kl)
                    # # PPO-P <<<<

                    # PPO V2 >>>>
                    # f = lambda n: 20-2*(round(n+4.5, -1)-10)/10
                    # f = lambda n: 15 - ((round((n-2)/3, 0))-4)
                    # if n <= 20:
                    #     G_AC = 15
                    # elif n <= 40:
                    #     G_AC = 10
                    # else:
                    #     G_AC = 5
                    # G_PPO = int(750//G_AC)
                    # print(f'G_AC={G_AC} | G_PPO={G_PPO}')

                    for g in range(1, G_AC+1):
                        # Reset model buffer
                        self.model_traj_buffer.reset()
                        # # Generate M k-steps imaginary rollouts for PPO training
                        # k_avg, ZListImag, elListImag = self.rollout_real_world(g, n)
                        # k_avg = self.rollout_real_world_trajectories(g, n)
                        ZListImag, elListImag = self.rollout_world_model_trajectories(g, n)
                        # ZListImag, elListImag = self.rollout_world_model_trajectories_q(g, n)
                        ppo_batch_size = int(self.model_traj_buffer.total_size())
                        # batch_size = 10000 #min(int(self.model_traj_buffer.total_size()), 10000)
                        stop_pi = False
                        kl = 0
                        dev = 0
                        # G_PPO = int(110 - g*5)
                        # print(f'\n\n[ Epoch {n}   Training Actor-Critic ({g}/{G}) ] Model Buffer: Size={self.model_traj_buffer.total_size()} | AvgK={self.model_traj_buffer.average_horizon()}'+(" "*25)+'\n')
                        for gg in range(1, G_PPO+1): # 101
                            # print(f'[ Epoch {n} ] AC: {g}/{G_AC} | ac: {gg}/{G_PPO} || stopPG={stop_pi} | KL={round(kl, 4)}'+(' '*50), end='\r')
                            print(f"[ Epoch {n} | Training AC ] AC: {g}/{G_AC} | ac: {gg}/{G_PPO} || stopPG={stop_pi} | Dev={round(dev, 4)}"+(" "*30), end='\r')
                            batch = self.model_traj_buffer.sample_batch(batch_size=ppo_batch_size, device=self._device_)
                            Jv, Jpi, kl, PiInfo = self.trainAC(g, batch, oldJs)
                            oldJs = [Jv, Jpi]
                            JVList.append(Jv)
                            JPiList.append(Jpi)
                            KLList.append(kl)
                            HList.append(PiInfo['entropy'])
                            DevList.append(PiInfo['deviation'])
                            dev = PiInfo['deviation']
                            if not self.stop_pi:
                                ppo_grads += 1
                            stop_pi = PiInfo['stop_pi']

                        # ac_grads += 1
                        # if (np.std(ZListImag[1:]) > (np.mean(ZListImag[1:])/3)) and (np.std(ZListImag[1:])>100):
                        #     print(f'Breaking AC loop at {g}'+(' ')*80)
                        #     break
                        # self.actor_critic.actor.log_std -= 0.05*T.ones(self.act_dim)
                        # self.buffer.gae_lambda += 2e-4
                    # PPO-P <<<<

                nt += E

            print('\n')

            # logs['time/training                       '] = time.time() - learn_start_real

            # logs['training/wm/Jtrain_mean             '] = np.mean(JMeanTrainList)
            # logs['training/wm/Jtrain                  '] = np.mean(JTrainList)
            logs['training/wm/Jval                    '] = ho_mean
            # logs['training/wm/test_mse                '] = np.mean(LossTestList)

            logs['training/ppo/critic/Jv              '] = np.mean(JVList)
            logs['training/ppo/critic/V(s)            '] = T.mean(self.model_traj_buffer.val_buf).item()
            # logs['training/ppo/critic/V_pi            '] = T.mean(self.model_traj_buffer.val_buf).item()
            # logs['training/ppo/critic/R_pi            '] = T.mean(self.model_traj_buffer.ret_buf).item()
            logs['training/ppo/critic/V-R             '] = T.mean(self.model_traj_buffer.val_buf).item()-T.mean(self.model_traj_buffer.ret_buf).item()

            logs['training/ppo/actor/Jpi              '] = np.mean(JPiList)
            # logs['training/ppo/actor/ac-grads         '] = ac_grads
            logs['training/ppo/actor/ppo-grads        '] = ppo_grads
            logs['training/ppo/actor/H                '] = np.mean(HList)
            logs['training/ppo/actor/KL               '] = np.mean(KLList)
            logs['training/ppo/actor/deviation        '] = np.mean(DevList)
            # logs['training/ppo/actor/log_pi            '] = PiInfo['log_pi']

            logs['data/env_buffer_size                '] = self.buffer.total_size()
            # logs['data/env_rollout_steps              '] = self.buffer.average_horizon()
            if hasattr(self, 'model_traj_buffer'):
                # logs['data/init_obs                       '] = 0. #len(self.buffer.init_obs)
                logs['data/model_buffer_size              '] = self.model_traj_buffer.total_size()
                # logs['data/model_rollout_steps            '] = self.model_traj_buffer.average_horizon()
            else:
                # logs['data/gae_lambda                '] = self.buffer.gae_lambda
                logs['data/init_obs                       '] = 0.
                logs['data/model_buffer_size              '] = 0.
                logs['data/model_rollout_steps            '] = 0.
            # else:
            #     logs['data/model_buffer              '] = 0
            # logs['data/rollout_length            '] = K

            logs['learning/real/rollout_return_mean   '] = np.mean(ZList[1:])
            logs['learning/real/rollout_return_std    '] = np.std(ZList[1:])
            logs['learning/real/rollout_length        '] = np.mean(elList[1:])

            logs['learning/imag/rollout_return_mean   '] = np.mean(ZListImag[1:])
            logs['learning/imag/rollout_return_std    '] = np.std(ZListImag[1:])
            logs['learning/imag/rollout_length        '] = np.mean(elListImag[1:])

            eval_start_real = time.time()
            print('\n[ Evaluation ]')
            EZ, ES, EL = self.evaluate()
            # EZ, ES, EL = self.evaluateII()
            # EZ, ES, EL = self.evaluate_op()

            # logs['time/evaluation                '] = time.time() - eval_start_real

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
                                'learning/imag/rollout_return_mean   ',
                                'evaluation/episodic_return_mean     ',
                                'evaluation/return_to_length         ',
                                'evaluation/return_to_full_length    ']
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




    def rollout_real_world(self, g, n):
    	# 07. Sample st uniformly from Denv
        device = self._device_
        EM = 10000
        max_el = 1000

        o, Z, el = self.traj_env.reset(), 0, 0
        ZList, elList = [0], [0]
        AvgZ, AvgEL = 0, 0

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
        k_end_total = 0
        for em in range(1, EM+1): # Generate trajectories
            print(f'[ Epoch {n} ] Model Rollout: em = {em} | Buffer = {self.model_traj_buffer.total_size()} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}'+(' ')*20, end='\r')
            # a = self.actor_critic.get_action_np(o)
            with T.no_grad(): a, log_pi, v = self.actor_critic.get_a_and_v_np(T.Tensor(o))
            # print('log_pi: ', log_pi)
            o_next, r, d, _ = self.traj_env.step(a)
            Z += r
            el += 1
            # t += 1
            self.model_traj_buffer.store(o, a, r, o_next, v, log_pi, el)
            o = o_next
            if d or (el == max_el):
                if el == max_el:
                    with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
                else:
                    # print('v=0')
                    v = T.Tensor([0.0])
                self.model_traj_buffer.finish_path(el, v)
                # print(f'termination: t={t} | el={el} | total_size={self.buffer.total_size()}')
                o, Z, el = self.traj_env.reset(), 0, 0

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

        with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
        self.model_traj_buffer.finish_path(el, v)

        return 0., ZList, elList #k_end_total//Nτ


    def rollout_real_world_trajectories(self, g, n):
    	# 07. Sample st uniformly from Denv
    	device = self._device_
    	Nτ = 500 # number of trajectories x number of models
    	# Nτ = 100 # number of trajectories x number of models
    	K = 1000

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	k_end_total = 0
    	for nτ in range(1, Nτ+1): # Generate trajectories
            o, Z, el = self.traj_env.reset(), 0, 0
            # print(f'nτ = {nτ}'+(' '*70), end='\r')
            for k in range(1, K+1): # Generate rollouts
                print(f'[ Epoch {n} ] Model Rollout: nτ = {nτ}/{Nτ} | k = {k}/{K} | Buffer = {self.model_traj_buffer.total_size()} | Return = {round(Z, 2)}', end='\r')
                # print(f'[ Epoch {n} ] AC Training Grads: {g} || Model Rollout: nτ = {nτ} | k = {k} | Buffer size = {self.model_traj_buffer.total_size()}'+(' '*10))
                with T.no_grad(): a, log_pi, v = self.actor_critic.get_a_and_v_np(T.Tensor(o))
                o_next, r, d, _ = self.traj_env.step(a)
                Z += r
                el += 1
                self.model_traj_buffer.store(o, a, r, o_next, v, log_pi, el)
                o = o_next
                if d or (el == K):
                    break
            if el == K:
                # print(f'[ Model Rollout ] Average Breaking : {k_end_total//Nτ}'+(' '*50))
                with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
            else:
                v = T.Tensor([0.0])
            self.model_traj_buffer.finish_path(el, v)
            k_end_total += k
            if self.model_traj_buffer.total_size() >= 10000:
                print(f'Breaking img rollouts at nτ={nτ}'+(' ')*80)
                break

    	return k_end_total//Nτ


    def rollout_world_model_trajectories_q(self, g, n):
    	# 07. Sample st uniformly from Denv
    	device = self._device_
    	Nτ = 250
    	K = 1000

    	O = O_init = self.buffer.sample_init_obs_batch(Nτ)
    	O_Nτ = len(O_init)
    	L = 5

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	ZList, elList = [0], [0]
    	AvgZ, AvgEL = 0, 0

    	for l in range(1, L+1):
            for nτ, oi in enumerate(O_init): # Generate trajectories
                o, Z, el = oi, 0, 0
                for k in range(1, K+1): # Generate rollouts
                    print(f'[ Epoch {n} | AC {g} ] Model Rollout: L={l} | nτ={nτ+1}/{O_Nτ} | k={k}/{K} | Buffer={self.model_traj_buffer.total_size()} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}', end='\r')
                    # print('\no: ', o)
                    # print(f'[ Epoch {n} ] AC Training Grads: {g} || Model Rollout: nτ = {nτ} | k = {k} | Buffer size = {self.model_traj_buffer.total_size()}'+(' '*10))
                    with T.no_grad(): a, log_pi, _, v = self.actor_critic.get_a_and_v(o)

                    # o_next, r, d, _ = self.fake_world.step(o, a) # ip: Tensor, op: Tensor
                    o_next, r, d, _ = self.fake_world.step(o, a, deterministic=True) # ip: Tensor, op: Tensor

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

                lastZ = currZ
                ZList.append(lastZ)
                AvgZ = sum(ZList)/(len(ZList)-1)
                lastEL = currEL
                elList.append(lastEL)
                AvgEL = sum(elList)/(len(elList)-1)

                if self.model_traj_buffer.total_size() >= self.configs['data']['model_buffer_size']:
                    # print(f'Breaking img rollouts at L={l}/nτ={nτ+1} | Buffer = {self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*85)
                    break

            if self.model_traj_buffer.total_size() >= self.configs['data']['model_buffer_size']:
                print(f'Breaking img rollouts at L={l}/nτ={nτ+1} | Buffer = {self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*85)
                break

    	return ZList, elList


    def rollout_world_model_trajectories_batch(self, g, n):
        # 07. Sample st uniformly from Denv
        device = self._device_

        rollout_trajs = self.configs['data']['rollout_trajectories'] # bs_ro
        Ksteps = self.configs['data']['rollout_horizon']
        # K = 500

        O = O_init = self.buffer.sample_init_obs_batch(rollout_trajs, self._device_)
        init_size = len(O_init)
        Z, el = 0, 0
        ZZ = T.zeros((init_size, 1), dtype=T.float32)
        EL = T.zeros((init_size, 1), dtype=T.float32)

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
        env_avg_horizon = self.buffer.average_horizon()
        K = min(Ksteps, int(env_avg_horizon*n*0.5))
        print('\n')
        for k in range(1, K+1):
            print(f'[ Epoch {n} | AC: {g} ] Model Rollout: k={k}/{K}'+(' '*50), end='\r')
            # print(f'[ Epoch {n} ] AC Training Grads: {g} || Model Rollout: nτ = {nτ} | k = {k} | Buffer size = {self.model_traj_buffer.total_size()}'+(' '*10))
            with T.no_grad(): A, Log_Pi, _, V = self.actor_critic.get_pi_and_v(T.Tensor(O)) # Stoch action | No repara
            O_next, R, D, _ = self.fake_world.step(O, A, deterministic=True) # ip: Tensor, op: Tensor

            self.model_traj_buffer.store_batch(O, A, R, O_next, V, Log_Pi, k) # ip: Tensor

            # O_next = T.Tensor(O_next)
            D = T.tensor(D, dtype=T.bool)
            nonD = ~D.squeeze(-1)

            ZZ[nonD] += R[nonD]
            EL[nonD] += T.ones((nonD.sum(), 1), dtype=T.float32)

            if nonD.sum() == 0:
                print(f'[ Epoch {n} Model Rollout ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}')
                break

            O = O_next

        V = T.zeros((init_size, 1), dtype=T.float32)
        if k == K:
            for i, e in enumerate(EL):
                if e == K and nonD[i]:
                    with T.no_grad(): V[i] = self.actor_critic.get_v(T.Tensor(O[i])).cpu()
        # print('EL: ', EL)
        # print('ZZ: ', ZZ)

        self.model_traj_buffer.finish_path_batch(EL, V)


    def rollout_world_model_trajectories(self, g, n):
    	# 07. Sample st uniformly from Denv
    	device = self._device_
    	Nτ = self.configs['data']['init_obs_size']
    	K = 1000

    	O = O_init = self.buffer.sample_init_obs_batch(Nτ)
    	O_Nτ = len(O_init)

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	k_end_total = 0
    	ZList, elList = [0], [0]
    	AvgZ, AvgEL = 0, 0

    	for nτ, oi in enumerate(O_init): # Generate trajectories
            for m, model in enumerate(self.models):
                o, Z, el = oi, 0, 0
                for k in range(1, K+1): # Generate rollouts
                    print(f'[ Epoch {n} | AC {g} ] Model Rollout: nτ = {nτ+1} | M = {m+1}/{len(self.models)} | k = {k}/{K} | Buffer = {self.model_traj_buffer.total_size()} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}', end='\r')
                    # print('\no: ', o)
                    # print(f'[ Epoch {n} ] AC Training Grads: {g} || Model Rollout: nτ = {nτ} | k = {k} | Buffer size = {self.model_traj_buffer.total_size()}'+(' '*10))
                    with T.no_grad(): pre_a, a, log_pi, _, v = self.actor_critic.get_a_and_v(o, on_policy=True, return_pre_pi=True)

                    # o_next = model.forward(o, a).detach() # ip: Tensor, op: Tensor
                    # # print('o_next: ', o_next)
                    # r = model.reward(o, a).detach()
                    # d = self._termination_fn("Hopper-v2", o, a, o_next)
                    # d = T.tensor(d, dtype=T.bool)

                    o_next = model.forward(o, a).detach().cpu() # ip: Tensor, op: Tensor
                    r = model.reward(o, a).detach()
                    d = self._termination_fn("Hopper-v2", o, a, o_next)
                    d = T.tensor(d, dtype=T.bool)

                    Z += float(r)
                    el += 1
                    # print('r: ', r)
                    self.model_traj_buffer.store(o, pre_a, a, r, o_next, v, log_pi, el)
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
                # print(f'[ Epoch {n} | AC {g} ] Breaking img rollouts at nτ={nτ+1}/m={m+1} | Buffer = {self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*40)
                break
    	print(f'[ Epoch {n} | AC {g} ] RollBuffer={self.model_traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | L={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*35)

    	# EZ, ES, EL = self.evaluate()
        #
    	# print(color.RED+f'[ Epoch {n} | AC {g} ] Inner Evaluation | Z={round(np.mean(EZ), 2)}±{round(np.std(EZ), 2)} | L={round(np.mean(EL), 2)}±{round(np.std(EL), 2)} | x{round(np.mean(EZ)/np.mean(EL), 2)}'+color.END+(' ')*40+'\n')

    	return ZList, elList




    def _reward_fn(self, env_name, obs, act):
        next_obs = next_obs.numpy()
        if len(obs.shape) == 1 and len(act.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        elif len(obs.shape) == 1:
            obs = obs[None]
            return_single = True
        else:
            return_single = False

        next_obs = next_obs.cpu().numpy()

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

        next_obs = next_obs.cpu().numpy()

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

    print('Start an MBPPO experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    alg_mode = configs['algorithm']['mode']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']
    wm_epochs = configs['algorithm']['learning']['grad_WM_steps']
    DE = configs['world_model']['num_ensembles']

    group_name = f"{env_name}-{alg_name}-Tanh(Pi)-V2-13" # Local
    # group_name = f"{env_name}-{alg_name}-GCP-0" # GCP
    exp_prefix = f"seed:{seed}"

    if wb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            # project='AMMI-RL-2022',
            project=f'AMMI-RL-{env_name}',
            config=configs
        )

    agent = MBPPO(exp_prefix, configs, seed, device, wb)

    agent.learn()

    print('\n')
    print('... End the MBPPO experiment')


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
