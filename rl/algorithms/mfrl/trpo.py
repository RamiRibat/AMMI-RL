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

from rl.algorithms.mfrl.mfrl import MFRL
from rl.control.policy import StochasticPolicy
# from rl.value_functions.q_function import QFunction
from rl.value_functions.v_function import VFunction



class ActorCritic: # Done
    """
    Actor-Critic
        An entity contains both the actor (policy) that acts on the environment,
        and a critic (V-function) that evaluate that state given a policy.
    """
    def __init__(self,
                 obs_dim, act_dim,
                 act_up_lim, act_low_lim,
                 configs, seed
                 ) -> None:
        # print('Initialize AC!')
        # Initialize parameters
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.act_up_lim, self.act_low_lim = act_up_lim, act_low_lim
        self.configs, self.seed = configs, seed
        self.device = configs['experiment']['device']

        self.actor, self.critic = None, None
        self._build()


    def _build(self):
        self.actor = self._set_actor()
        self.critic = self._set_critic()
        # parameters will be updated using a weighted average
        for p in self.critic_target.parameters():
            p.requires_grad = False


    def _set_actor(self):
        net_configs = self.configs['actor']['network']
        return StochasticPolicy(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            net_configs, self.device, self.seed).to(self.device)


    def _set_critic(self):
        net_configs = self.configs['critic']['network']
        return VFunction(
            self.obs_dim,
            net_configs, self.seed).to(self.device)



class NPG(MFRL):
    """
    Algorithm: Natural Policy Gradient (On-Policy, Model-free)

        01. Input: θ1, θ2, φ                                    > Initial parameters
        02. ¯θ1 ← θ1, ¯θ2 ← θ2                                  > Initialize target network weights
        03. D ← ∅                                               > Initialize an empty replay pool
        04.    for each iteration do
        05.       for each environment step do
        06.          at ∼ πφ(at|st)                             > Sample action from the policy
        07.          st+1 ∼ p(st+1|st, at)                      > Sample transition from the environment
        08.          D ← D ∪ {(st, at, r(st, at), st+1)}        > Store the transition in the replay pool
        09.       end for
        10.       for each gradient step do
        11.          θi ← θi − λ_Q ˆ∇θi J_Q(θi) for i ∈ {1, 2}  > Update the Q-function parameters
        12.          φ ← φ − λ_π ˆ∇φ J_π(φ)                     > Update policy weights
        13.          α ← α − λ ˆ∇α J(α)                         > Adjust temperature
        14.          ¯θi ← τ θi + (1 − τ) + ¯θi for i ∈ {1, 2}  > Update target network weights
        15.       end for
        16.    end for
        17. Output: θ1, θ2, φ                                   > Optimized parameters

    """
    def __init__(self, exp_prefix, configs, seed) -> None:
        super(SAC, self).__init__(exp_prefix, configs, seed)
        print('Initialize NPG Algorithm!')
        self.configs = configs
        self.seed = seed
        self._build()


    def _build(self):
        super(SAC, self)._build()
        self._build_npg()


    def _build_npg(self):
        self._set_actor_critic()
        # self._set_alpha()


    def _set_actor_critic(self):
        self.actor_critic = ActorCritic(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.configs, self.seed)



    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        G = self.configs['algorithm']['learning']['grad_AC_steps']

        batch_size = self.configs['data']['batch_size']

        o, Z, el, t = self.learn_env.reset(), 0, 0, 0
        # o, Z, el, t = self.initialize_learning(NT, Ni)
        oldJs = [0, 0, 0]
        JQList, JAlphaList, JPiList = [0]*Ni, [0]*Ni, [0]*Ni
        AlphaList = [self.alpha]*Ni
        logs = dict()
        lastEZ, lastES = 0, -2

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
                    for g in range(1, G+1):
                        batch = self.replay_buffer.sample_batch(batch_size)
                        Jq, Jalpha, Jpi = self.trainAC(g, batch, oldJs)
                        oldJs = [Jq, Jalpha, Jpi]
                        JQList.append(Jq.item())
                        JPiList.append(Jpi.item())
                        if self.configs['actor']['automatic_entropy']:
                            JAlphaList.append(Jalpha.item())
                            AlphaList.append(self.alpha)

                nt += E

            logs['time/training                  '] = time.time() - learn_start_real
            logs['training/npg/Jq                '] = np.mean(JQList)
            logs['training/npg/Jpi               '] = np.mean(JPiList)
            # if self.configs['actor']['automatic_entropy']:
            #     logs['training/npg/Jalpha            '] = np.mean(JAlphaList)
            #     logs['training/npg/alpha             '] = np.mean(AlphaList)

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


    def trainAC(self, g, batch, oldJs):
        AUI = self.configs['algorithm']['learning']['alpha_update_interval']
        # PUI = self.configs['algorithm']['learning']['policy_update_interval']
        TUI = self.configs['algorithm']['learning']['target_update_interval']

        Jv = self.updateV(batch)
        # Jalpha = self.updateAlpha(batch)# if (g % AUI == 0) else oldJs[1]
        Jpi = self.updatePi(batch)# if (g % PUI == 0) else oldJs[2]
        # if g % TUI == 0:
        # self.updateTarget()

        return Jv, Jpi


    def updateV(self, batch):
        """"
        JQ(θ) = E(st,at)∼D[ 0.5 ( Qθ(st, at)
                            − r(st, at)
                            + γ Est+1∼D[ Eat+1~πφ(at+1|st+1)[ Qθ¯(st+1, at+1)
                                                − α log(πφ(at+1|st+1)) ] ] ]
        """
        gamma = self.configs['critic']['gamma']

        O = batch['observations']
        A = batch['actions']
        R = batch['rewards']
        O_next = batch['observations_next']
        D = batch['terminals']

        # Calculate two Q-functions
        Qs = self.actor_critic.critic(O, A)
        # # Bellman backup for Qs
        with T.no_grad():
            pi_next, log_pi_next = self.actor_critic.actor(O_next, reparameterize=True, return_log_pi=True)
            A_next = pi_next
            # Qs_targ = T.cat(self.actor_critic.critic(O_next, A_next), dim=1) # WRONG!! :"D
            Qs_targ = T.cat(self.actor_critic.critic_target(O_next, A_next), dim=1)
            min_Q_targ, _ = T.min(Qs_targ, dim=1, keepdim=True)
            Qs_backup = R + gamma * (1 - D) * (min_Q_targ - self.alpha * log_pi_next)

        # # MSE loss
        Jq = 0.5 * sum([F.mse_loss(Q, Qs_backup) for Q in Qs])

        # Gradient Descent
        self.actor_critic.critic.optimizer.zero_grad()
        Jq.backward()
        self.actor_critic.critic.optimizer.step()

        return Jq


    def updatePi(self, batch):
        """
        Jπ(φ) = Est∼D[ Eat∼πφ[α log (πφ(at|st)) − Qθ(st, at)] ]
        """

        O = batch['observations']

        # Policy Evaluation
        pi, log_pi = self.actor_critic.actor(O, return_log_pi=True)
        Qs_pi = T.cat(self.actor_critic.critic(O, pi), dim=1)
        min_Q_pi, _ = T.min(Qs_pi, dim=1, keepdim=True)

        # Policy Improvement
        Jpi = (self.alpha * log_pi - min_Q_pi).mean()

        # Gradient Ascent
        self.actor_critic.actor.optimizer.zero_grad()
        Jpi.backward()
        self.actor_critic.actor.optimizer.step()

        return Jpi


    def compute_fisher(self):
        pass

    def compute_advantage(self):
        pass






def main(exp_prefix, config, seed):

    print('Start an SAC experiment...')
    print('\n')

    configs = config.configurations

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']

    group_name = f"{env_name}-{alg_name}"
    exp_prefix = f"seed:{seed}"

    if configs['experiment']['WandB']:
        # print('WandB')
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            project='AMMI-RL-2022',
            config=configs
        )

    agent = SAC(exp_prefix, configs, seed)

    agent.learn()

    print('\n')
    print('... End the SAC experiment')

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
