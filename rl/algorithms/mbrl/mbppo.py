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
from rl.world_models.fake_world import FakeWorld
import rl.environments.mbpo.static as mbpo_static
# from rl.data.dataset import RLDataModule





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
        06:    Learn dynamics model(s) Mkat_k+1 using data in the buffer.
        07:    Policy updates: π_n+1; V_n+1 = MB-PPO(π_n, V_n, Mhat_n+1) // call K times
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
        self._set_fake_world()


    ## PPO
    def _set_ppo(self):
        PPO._build_ppo(self)


    ## FakeEnv
    def _set_fake_world(self):
        env_name = self.configs['environment']['name']
        device = self._device_
        if self.configs['environment']['name'][:4] == 'pddm':
        	static_fns = None
        else:
        	static_fns = mbpo_static[env_name[:-3].lower()]

        self.fake_world = FakeWorld(self.world_model)


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        G_ppo = self.configs['algorithm']['learning']['grad_PPO_steps']

        model_train_frequency = self.configs['world_model']['model_train_freq']
        batch_size_m = self.configs['world_model']['network']['batch_size'] # bs_m
        wm_epochs = self.configs['algorithm']['learning']['grad_WM_steps']
        real_ratio = self.configs['data']['real_ratio'] # rr
        batch_size = self.configs['data']['batch_size'] # bs
        mini_batch_size = self.configs['data']['mini_batch_size'] # bs
        rollout_trajectories = self.configs['data']['rollout_trajectories'] # bs_ro
        K = self.configs['data']['rollout_horizon']

        global_step = 0
        start_time = time.time()
        o, Z, el, t = self.learn_env.reset(), 0, 0, 0
        oldJs = [0, 0]
        JVList, JPiList = [0]*Ni, [0]*Ni
        logs = dict()
        lastEZ, lastES = 0, -2

        start_time_real = time.time()
        for n in range(1, N+1):
            if self.configs['experiment']['print_logs']:
                print('=' * 50)
                if n > Nx:
                    print(f'\n[ Epoch {n}   Learning ]'+(' '*50))
                    oldJs = [0, 0]
                    JVList, JPiList = [0], [0]
                    JValList = [0]
                elif n > Ni:
                    print(f'\n[ Epoch {n}   Exploration + Learning ]'+(' '*50))
                    JVList, JPiList = [], []
                    JValList = []
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]'+(' '*50))
                    oldJs = [0, 0]
                    JVList, JPiList = [0], [0]
                    JValList = [0]

            print(f'[ Replay Buffer ] Size: {self.buffer.size}')
            nt = 0
            learn_start_real = time.time()
            while nt < NT: # full epoch
                # Interaction steps
                for e in range(1, E+1):
                    o, Z, el, t = self.internact(n, o, Z, el, t)
                    print(f'[ Epoch {n}   Interaction ] Env Steps: {e} | Return: {round(Z, 2)}'+(" "*10), end='\r')

                # Taking gradient steps after exploration
                if n > Ni:
                    # if nt % model_train_frequency == 0:
                    # 03. Train model pθ on Denv via maximum likelihood
                    # print(f'\n[ Epoch {n}   Training World Model ]'+(' '*50))

                    ho_mean = self.fake_world.train_fake_world(self.buffer)
                    JValList.append(ho_mean) # ho: holdout

                    for g in range(1, G_ppo+1):
                        # print(f'Actor-Critic Grads...{g}', end='\r')
                        print(f'[ Epoch {n}   Training Actor-Critic ] AC Grads: {g}'+(" "*50), end='\r')
                        # # Reallocate model buffer
                        self.initialize_model_buffer()
                        # # Generate M k-steps imaginary rollouts for SAC traingin
                        # self.rollout_world_model_op(rollout_trajectories, K, n)
                        self.rollout_world_model_trajectories(rollout_trajectories, K, n)
                        # PPO-P >>>>
                        batch_size = int(self.model_buffer.ptr // mini_batch_size)
                        for b in range(0, batch_size, mini_batch_size):
                            # print('ptr: ', self.buffer.ptr)
                            mini_batch = self.model_buffer.sample_batch(mini_batch_size, self._device_)
                            Jv, Jpi, stop_pi = self.trainAC(g, mini_batch, oldJs)
                            oldJs = [Jv, Jpi]
                            JVList.append(Jv.item())
                            JPiList.append(Jpi.item())
                        # PPO-P <<<<

                nt += E

            print('\n')
            # logs['time/training                  '] = time.time() - learn_start_real

            # logs['training/wm/Jtrain_mean        '] = np.mean(JMeanTrainList)
            # logs['training/wm/Jtrain             '] = np.mean(JTrainList)
            logs['training/wm/Jval               '] = np.mean(JValList)
            # logs['training/wm/test_mse           '] = np.mean(LossTestList)

            logs['training/sac/Jv                '] = np.mean(JVList)
            logs['training/sac/Jpi               '] = np.mean(JPiList)

            logs['data/env_buffer                '] = self.buffer.size
            # if hasattr(self, 'model_buffer'):
            #     logs['data/model_buffer              '] = self.model_buffer.size
            # else:
            #     logs['data/model_buffer              '] = 0
            # logs['data/rollout_length            '] = K

            eval_start_real = time.time()
            EZ, ES, EL = self.evaluate_op()

            # logs['time/evaluation                '] = time.time() - eval_start_real

            if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
                logs['evaluation/episodic_score_mean '] = np.mean(ES)
                logs['evaluation/episodic_score_std  '] = np.std(ES)
            else:
                logs['evaluation/episodic_return_mean'] = np.mean(EZ)
                logs['evaluation/episodic_return_std '] = np.std(EZ)
            logs['evaluation/episodic_length_mean'] = np.mean(EL)

            logs['time/total                     '] = time.time() - start_time_real

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
                for k, v in logs.items():
                    print(f'{k}  {round(v, 2)}'+(' '*10))

            # WandB
            if self.WandB:
                wandb.log(logs)

        self.learn_env.close()
        self.eval_env.close()


    def rollout_world_model_trajectories(self, rollout_trajectories, K, n):
    	# 07. Sample st uniformly from Denv
    	device = self._device_
    	batch_size = rollout_trajectories
    	# print(f'[ Epoch {n}   Model Rollout ] Batch Size: {batch_size} | Rollout Horizon: {K}'+(' '*50))
    	B_ro = self.buffer.sample_batch(batch_size) # Torch
    	O = B_ro['observations_next']
    	D = B_ro['terminals']

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	k_end_total = 0
    	for o, d in zip(O, D):
            # print('o: ', o.shape)
            # print('d: ', d.shape)
            for k in range(1, K+1):
                # print(f'k = {k}', end='\r')
                with T.no_grad(): a, log_pi, _, v = self.actor_critic.get_pi_and_v(o)
                a, log_pi, v = a.cpu(), log_pi.cpu(), v.cpu()

                o_next, r, d_next, _ = self.fake_world.step(o, a) # ip: Tensor, op: Tensor
            	# O_next, R, D, _ = self.fake_world.step_np(O, A) # ip: Tensor, op: Numpy

                self.model_buffer.store_transition(o, a, r, d, v, log_pi)

                # o_next = T.Tensor(o_next)
                # d = T.tensor(d, dtype=T.bool)
                # nond = ~d.squeeze(-1)

                # if nonD.sum() == 0:
            	#     print(f'[ Epoch {n}   Model Rollout ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}')
            	#     break

                if d_next:
    	            print(f'[ Model Rollout ] Breaking early: {k}'+(' '*50), end='\r')
    	            k_end_total += k
    	            break

                o, d = o_next, d_next

    	print(f'[ Model Rollout ] Average Breaking : {k_end_total//batch_size}'+(' '*50))
    	with T.no_grad(): v = self.actor_critic.get_v(o)
    	self.model_buffer.traj_tail(d, v)


    def rollout_world_model_op(self, rollout_trajectories, K, n):
    	# 07. Sample st uniformly from Denv
    	device = self._device_
    	batch_size = rollout_trajectories
    	# print(f'[ Epoch {n}   Model Rollout ] Batch Size: {batch_size} | Rollout Horizon: {K}'+(' '*50))
    	B_ro = self.buffer.sample_batch(batch_size) # Torch
    	O = B_ro['observations']
    	D = B_ro['terminals']

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	for k in range(1, K+1):
            # print('k = ', k)
            # print('k = ', k, end='\r')
            A = self.actor_critic.get_action(O) # Stochastic action | No reparameterization
            with T.no_grad(): A, log_Pi, _, V = self.actor_critic.get_pi_and_v(O)

            O_next, R, D_next, _ = self.fake_world.step(O, A) # ip: Tensor, op: Tensor
        	# O_next, R, D, _ = self.fake_world.step_np(O, A) # ip: Tensor, op: Numpy

            self.model_buffer.store_batch(O, A, R, D, V, log_Pi)

            O_next = T.Tensor(O_next)
            D = T.tensor(D, dtype=T.bool)
            nonD = ~D.squeeze(-1)
            if nonD.sum() == 0:
        	    print(f'[ Epoch {n}   Model Rollout ] Breaking early: {k} | {nonD.sum()} / {nonD.shape}')
        	    break

            O, D = O_next[nonD], D_next[nonD]

    	with T.no_grad(): V = self.actor_critic.get_v(O)
    	self.model_buffer.traj_tail(D, V)


    def ppo_mini_batch(self, mini_batch_size):
    	B = self.model_buffer.sample_batch(mini_batch_size, self._device_)
    	return B





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

    group_name = f"{env_name}-{alg_name}-{alg_mode}"
    exp_prefix = f"seed:{seed}"

    if wb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            project='AMMI-RL-2022',
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
