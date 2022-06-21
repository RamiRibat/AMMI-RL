"""
Inspired by:

    1. RLKit: https://github.com/rail-berkeley/rlkit
    2. StabelBaselines: https://github.com/hill-a/stable-baselines
    3. SpinningUp OpenAI: https://github.com/openai/spinningup
    4. CleanRL: https://github.com/vwxyzjn/cleanrl

"""

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
from rl.control.policy import StochasticPolicy, OVOQPolicy
from rl.value_functions.q_function import SoftQFunction


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






class ActorCritic: # Done
    """
    Actor-Critic
        An entity contains both the actor (policy) that acts on the environment,
        and a critic (Q-function) that evaluate that state-action given a policy.
    """
    def __init__(self,
                 obs_dim, act_dim,
                 act_up_lim, act_low_lim,
                 configs, seed, device
                 ) -> None:
        print('Initialize AC')
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.act_up_lim, self.act_low_lim = act_up_lim, act_low_lim
        self.configs, self.seed = configs, seed
        self._device_ = device

        self.actor, self.critic, self.critic_target = None, None, None
        self._build()


    def _build(self):
        self.actor = self._set_actor()
        self.critic, self.critic_target = self._set_critic(), self._set_critic()
        # parameters will be updated using a weighted average
        for p in self.critic_target.parameters():
            p.requires_grad = False


    def _set_actor(self):
        net_configs = self.configs['actor']['network']
        return StochasticPolicy(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            net_configs, self._device_, self.seed)
        # return OVOQPolicy(
        #     self.obs_dim, self.act_dim,
        #     self.act_up_lim, self.act_low_lim,
        #     net_configs, self._device_, self.seed)


    def _set_critic(self):
        net_configs = self.configs['critic']['network']
        return SoftQFunction(
            self.obs_dim, self.act_dim,
            net_configs, self._device_, self.seed)


    def get_q(self, o, a):
        # Update Q
        return self.critic(o, a)


    def get_q_target(self, o, a):
        # Update Q
        return self.critic_target(o, a)


    # def get_pi(self, o, a=None, reparameterize=True, deterministic=False, return_log_pi=False):
    #     pi, log_pi, entropy = self.actor(o, a, reparameterize, deterministic, return_log_pi)
    #     return pi, log_pi, entropy

    def get_pi(self, o, a=None,
               on_policy=False,
               reparameterize=True,
               deterministic=False,
               return_log_pi=True,
               return_entropy=True):
        # Update AC
        action, log_pi, entropy = self.actor(o, a, on_policy,
                                             reparameterize,
                                             deterministic,
                                             return_log_pi,
                                             return_entropy)
        return action, log_pi, entropy


    # def get_action(self, o, a=None, reparameterize=False, deterministic=False, return_log_pi=False):
    #     o = T.Tensor(o)
    #     if a: a = T.Tensor(a)
    #     with T.no_grad(): a, _, _ = self.actor(o, a, reparameterize, deterministic, return_log_pi)
    #     return a.cpu()

    def get_action(self, o, a=None,
                   on_policy=False,
                   reparameterize=False,
                   deterministic=False,
                   return_log_pi=False,
                   return_entropy=False):
        o = T.Tensor(o)
        if a: a = T.Tensor(a)
        with T.no_grad(): a, _, _ = self.actor(o, a, on_policy,
                                               reparameterize,
                                               deterministic,
                                               return_log_pi,
                                               return_entropy)
        return a.cpu()


    # def get_action_np(self, o, a=None, reparameterize=False, deterministic=False, return_log_pi=False):
    #     # Interaction/Evaluation
    #     return self.get_action(o, a, reparameterize, deterministic, return_log_pi).numpy()

    def get_action_np(self, o, a=None,
                      on_policy=False,
                      reparameterize=False,
                      deterministic=False,
                      return_log_pi=False,
                      return_entropy=False):
        # Interaction/Evaluation
        return self.get_action(o, a, on_policy,
                               reparameterize,
                               deterministic,
                               return_log_pi,
                               return_entropy).numpy()


    # def get_pi_and_q(self, o, a=None):
    #     pi, log_pi, entropy = self.actor(o, a)
    #     return pi, log_pi, entropy, self.critic(o, a)

    def get_pi_and_q(self, o, a=None,
                    on_policy=False,
                    reparameterize=True,
                    deterministic=False,
                    return_log_pi=True,
                    return_entropy=True):
        pi, log_pi, entropy = self.actor(o, a, on_policy,
                                         reparameterize,
                                         deterministic,
                                         return_log_pi,
                                         return_entropy)
        return pi, log_pi, entropy, self.critic(o, a)





class SAC(MFRL):
    """
    Algorithm: Soft Actor-Critic (Off-policy, Model-free)

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
    def __init__(self, exp_prefix, configs, seed, device, wb) -> None:
        super(SAC, self).__init__(exp_prefix, configs, seed, device)
        print('Initialize SAC Algorithm')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()


    def _build(self):
        super(SAC, self)._build()
        self._build_sac()


    def _build_sac(self):
        self._set_actor_critic()
        self._set_alpha()


    def _set_actor_critic(self):
        self.actor_critic = ActorCritic(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.configs, self.seed, self._device_)


    def _set_alpha(self):
        if self.configs['actor']['automatic_entropy']:
            # Learned Temprature
            device = self._device_
            optimizer = 'T.optim.' + self.configs['actor']['network']['optimizer']
            lr = self.configs['actor']['network']['lr']
            target_entropy = self.configs['actor']['target_entropy']

            if target_entropy == 'auto':
                self.target_entropy = (
                    - 1.0 * T.prod(
                        T.Tensor(self.learn_env.action_space.shape).to(device)
                    ).item())

            self.log_alpha = T.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = eval(optimizer)([self.log_alpha], lr)
        else:
            # Fixed Temprature
            self.alpha = self.configs['actor']['alpha']


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
        logs = dict()
        lastEZ, lastES = 0, -2

        start_time_real = time.time()
        for n in range(1, N+1):
            if self.configs['experiment']['print_logs']:
                print('=' * 80)
                if n > Nx:
                    print(f'\n[ Epoch {n}   Learning ]'+(' '*50))
                    oldJs = [0, 0, 0]
                    JQList, JAlphaList, JPiList = [], [], []
                    HList, LogPiList = [], []
                elif n > Ni:
                    print(f'\n[ Epoch {n}   Exploration + Learning ]'+(' '*50))
                    JQList, JAlphaList, JPiList = [], [], []
                    HList, LogPiList = [], []
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]'+(' '*50))
                    oldJs = [0, 0, 0]
                    JQList, JAlphaList, JPiList = [0], [self.alpha], [0]
                    HList, LogPiList = [0], [0]

            nt = 0
            ZList, elList = [0], [0]
            ZListImag, elListImag = [0, 0], [0, 0]
            AvgZ, AvgEL = 0, 0

            learn_start_real = time.time()
            while nt < NT:
                # Interaction steps
                for e in range(1, E+1):
                    o, Z, el, t = self.internact(n, o, Z, el, t)

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


                # Taking gradient steps after exploration
                if n > Ni:
                    for g in range(1, G+1):
                        batch = self.buffer.sample_batch(batch_size, device=self._device_)
                        Jq, Jalpha, Jpi, PiInfo = self.trainAC(g, batch, oldJs)
                        oldJs = [Jq, Jalpha, Jpi]
                        JQList.append(Jq)
                        JPiList.append(Jpi)
                        if self.configs['actor']['automatic_entropy']:
                            JAlphaList.append(Jalpha)
                            AlphaList.append(self.alpha)

                nt += E

            # logs['time/training                     '] = time.time() - learn_start_real
            logs['training/sac/critic/Jq              '] = np.mean(JQList)
            logs['training/sac/actor/Jpi              '] = np.mean(JPiList)
            if self.configs['actor']['automatic_entropy']:
                logs['training/sac/actor/Jalpha           '] = np.mean(JAlphaList)
                logs['training/sac/actor/alpha            '] = np.mean(AlphaList)

            logs['data/env_buffer_size                '] = self.buffer.size

            logs['learning/real/rollout_return_mean   '] = np.mean(ZList[1:])
            logs['learning/real/rollout_return_std    '] = np.std(ZList[1:])
            logs['learning/real/rollout_length        '] = np.mean(elList[1:])

            eval_start_real = time.time()
            EZ, ES, EL = self.evaluate()

            # logs['time/evaluation                     '] = time.time() - eval_start_real

            if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
                logs['evaluation/episodic_score_mean      '] = np.mean(ES)
                logs['evaluation/episodic_score_std       '] = np.std(ES)
            else:
                logs['evaluation/episodic_return_mean     '] = np.mean(EZ)
                logs['evaluation/episodic_return_std      '] = np.std(EZ)
            logs['evaluation/episodic_length_mean     '] = np.mean(EL)
            logs['evaluation/return_to_length         '] = np.mean(EZ)/np.mean(EL)

            logs['time/total                          '] = time.time() - start_time_real

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
                return_means = ['learning/real/rollout_return_mean   ',
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


    def trainAC(self, g, batch, oldJs):
        AUI = self.configs['algorithm']['learning']['alpha_update_interval']
        PUI = self.configs['algorithm']['learning']['policy_update_interval']
        TUI = self.configs['algorithm']['learning']['target_update_interval']

        Jq = self.updateQ(batch, oldJs[0])
        Jq = Jq.item()
        Jalpha = self.updateAlpha(batch, oldJs[1])# if (g % AUI == 0) else oldJs[1]
        Jpi, PiInfo = self.updatePi(batch, oldJs[2])# if (g % PUI == 0) else oldJs[2]
        Jpi = Jpi.item()

        if g % TUI == 0:
            self.updateTarget()

        return Jq, Jalpha, Jpi, PiInfo


    def updateQ(self, batch, Jq_old):
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
        # Qs = self.actor_critic.critic(O, A)
        Qs = self.actor_critic.get_q(O, A)

        # Bellman backup for Qs
        with T.no_grad():
            # pi_next, log_pi_next, entropy_next = self.actor_critic.actor(O_next, reparameterize=True, return_log_pi=True)
            pi_next, log_pi_next, entropy_next = self.actor_critic.get_pi(O_next, reparameterize=True, return_log_pi=True)
            A_next = pi_next
            # Qs_targ = T.cat( self.actor_critic.critic_target(O_next, A_next), dim=1 )
            Qs_targ = T.cat( self.actor_critic.get_q_target(O_next, A_next), dim=1 )
            min_Q_targ, _ = T.min(Qs_targ, dim=1, keepdim=True)
            Qs_backup = R + gamma * (1 - D) * (min_Q_targ - self.alpha * log_pi_next)

        # MSE loss
        Jq = 0.5 * sum([F.mse_loss(Q, Qs_backup) for Q in Qs])

        # Gradient Descent
        self.actor_critic.critic.optimizer.zero_grad()
        Jq.backward()
        self.actor_critic.critic.optimizer.step()

        return Jq


    def updateAlpha(self, batch, Jalpha_old):
        """

        αt* = arg min_αt Eat∼πt∗[ −αt log( πt*(at|st; αt) ) − αt H¯

        """
        if self.configs['actor']['automatic_entropy']:
            # Learned Temprature
            O = batch['observations']

            with T.no_grad():
                # _, log_pi = self.actor_critic.actor(O, return_log_pi=True)
                _, log_pi, _ = self.actor_critic.get_pi(O, return_log_pi=True)
            Jalpha = - ( self.log_alpha * (log_pi + self.target_entropy) ).mean()

            # Gradient Descent
            self.alpha_optimizer.zero_grad()
            Jalpha.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

            return Jalpha
        else:
            # Fixed Temprature
            return 0.0


    def updatePi(self, batch, Jpi_old):
        """
        Jπ(φ) = Est∼D[ Eat∼πφ[α log (πφ(at|st)) − Qθ(st, at)] ]
        """
        PiInfo = dict()

        O = batch['observations']

        # Policy Evaluation
        # pi, log_pi, entropy = self.actor_critic.actor(O, return_log_pi=True)
        # Qs_pi = T.cat(self.actor_critic.critic(O, pi), dim=1)
        pi, log_pi, entropy = self.actor_critic.get_pi(O, reparameterize=True, return_log_pi=True)
        Qs_pi = T.cat(self.actor_critic.get_q(O, pi), dim=1)
        min_Q_pi, _ = T.min(Qs_pi, dim=1, keepdim=True)


        # Policy Improvement
        Jpi = (self.alpha * log_pi - min_Q_pi).mean()

        # Gradient Ascent
        self.actor_critic.actor.optimizer.zero_grad()
        Jpi.backward()
        self.actor_critic.actor.optimizer.step()

        PiInfo['entropy'] = entropy.mean().item()
        PiInfo['log_pi'] = log_pi.mean().item()

        return Jpi, PiInfo


    def updateTarget(self):
        # print('updateTarget')
        tau = self.configs['critic']['tau']
        with T.no_grad():
            for p, p_targ in zip(self.actor_critic.critic.parameters(),
                                 self.actor_critic.critic_target.parameters()):
                p_targ.data.copy_(tau * p.data + (1 - tau) * p_targ.data)






def main(exp_prefix, config, seed, device, wb):

    print('Start an SAC experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']

    group_name = f"{env_name}-{alg_name}-Mac-R"
    # group_name = f"{env_name}-{alg_name}-GCP-A"
    exp_prefix = f"seed:{seed}"

    if wb:
        # print('WandB')
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            project='AMMI-RL-2022',
            config=configs
        )

    agent = SAC(exp_prefix, configs, seed, device, wb)

    agent.learn()

    print('\n')
    print('... End the SAC experiment')

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
    # print('\ndevice: ', device)
    wb = eval(args.wb)

    main(exp_prefix, config, seed, device, wb)
