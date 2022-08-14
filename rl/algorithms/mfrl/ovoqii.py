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
nn = T.nn

from rl.algorithms.mfrl.mfrl import MFRL
from rl.value_functions.v_function import VFunction
from rl.value_functions.q_function import SoftQFunction
from rl.control.policy import PPOPolicy, StochasticPolicy, OVOQPolicy, Policy
from rl.data.buffer import TrajBuffer, ReplayBuffer


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
        print('Initialize AC(OVOQ)')
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.act_up_lim, self.act_low_lim = act_up_lim, act_low_lim
        self.configs, self.seed = configs, seed
        # self.net_configs = configs['actor']['network']
        self._device_ = device

        self.actor, self.ov, self.oq, self.oq_target = None, None, None, None
        self._build()


    def _build(self):
        self.actor = self._set_actor()
        self.ov = self._set_critic('V')
        self.oq, self.oq_target = self._set_critic('Q'), self._set_critic('Q')
        # parameters will be updated using a weighted average
        for p in self.oq_target.parameters():
            p.requires_grad = False


    def _set_actor(self):
        net_configs = self.configs['actor']['network']
        # return StochasticPolicy(
        #     self.obs_dim, self.act_dim,
        #     self.act_up_lim, self.act_low_lim,
        #     net_configs, self._device_, self.seed)
        # return OVOQPolicy(
        #     self.obs_dim, self.act_dim,
        #     self.act_up_lim, self.act_low_lim,
        #     net_configs, self._device_, self.seed)
        return Policy(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            net_configs, self._device_, self.seed)


    def _set_critic(self, value):
        if value == 'V':
            net_configs = self.configs['critic-v']['network']
            return VFunction(
                self.obs_dim, self.act_dim,
                net_configs, self._device_, self.seed)
        else: # Q
            net_configs = self.configs['critic-q']['network']
            return SoftQFunction(
                self.obs_dim, self.act_dim,
                net_configs, self._device_, self.seed)


    def get_v(self, o):
        return self.ov(o)


    def get_q(self, o, a):
        return self.oq(o, a)


    def get_q_target(self, o, a):
        return self.oq_target(o, a)


    def get_pi(self, o, a=None,
               on_policy=True,
               reparameterize=False,
               deterministic=False,
               return_log_pi=True,
               return_entropy=True):
        _, pi, log_pi, entropy = self.actor(o, a, on_policy,
                                             reparameterize,
                                             deterministic,
                                             return_log_pi,
                                             return_entropy)
        return pi, log_pi, entropy


    def get_action(self, o, a=None,
                   on_policy=True,
                   reparameterize=False,
                   deterministic=False,
                   return_log_pi=True,
                   return_entropy=True):
        o = T.Tensor(o)
        if a: a = T.Tensor(a)
        with T.no_grad(): _, a, _, _ = self.actor(o, a, on_policy,
                                               reparameterize,
                                               deterministic,
                                               return_log_pi,
                                               return_entropy)
        return a.cpu()


    def get_action_np(self, o, a=None,
                      on_policy=True,
                      reparameterize=False,
                      deterministic=False,
                      return_log_pi=False,
                      return_entropy=False):
        return self.get_action(o, a, on_policy,
                               reparameterize,
                               deterministic,
                               return_log_pi,
                               return_entropy).numpy()


    # def get_pi_and_v(self, o, a=None,
    #                 on_policy=True,
    #                 reparameterize=False,
    #                 deterministic=False,
    #                 return_log_pi=True,
    #                 return_entropy=True):
    #     pi, log_pi, entropy = self.actor(o, a, on_policy,
    #                                      reparameterize,
    #                                      deterministic,
    #                                      return_log_pi,
    #                                      return_entropy)
    #     return pi, log_pi, entropy, self.ov(o)


    def get_a_and_v(self, o, a=None,
                    on_policy=True,
                    reparameterize=False,
                    deterministic=False,
                    return_log_pi=True,
                    return_entropy=True,
                    return_pre_pi=True):
        pre_a, a, log_pi, entropy = self.actor(o, a, on_policy,
                                             reparameterize,
                                             deterministic,
                                             return_log_pi,
                                             return_entropy,
                                             return_pre_pi
                                             )
        return pre_a.cpu(), a.cpu(), log_pi.cpu(), entropy, self.ov(o).cpu()


    def get_a_and_v_np(self, o, a=None,
                       on_policy=True,
                       reparameterize=False,
                       deterministic=False,
                       return_log_pi=True,
                       return_entropy=True,
                       return_pre_pi=True):
        o = T.Tensor(o)
        if a: a = T.Tensor(a)
        with T.no_grad(): pre_a, a, log_pi, entropy = self.actor(o, a, on_policy,
                                                          reparameterize,
                                                          deterministic,
                                                          return_log_pi,
                                                          return_entropy,
                                                          return_pre_pi
                                                          )
        return pre_a.cpu().numpy(), a.cpu().numpy(), log_pi.cpu().numpy(), self.ov(o).cpu().numpy()


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
        return pi, log_pi, entropy, self.oq(o, a)






# class OVOQ:
class OVOQ(MFRL):
    """
    Algorithm: On-policy(V) Off-policy(Q) policy optimization (Model-Free)

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
        super(OVOQ, self).__init__(exp_prefix, configs, seed, device)
        print('Initialize OVOQII Algorithm!')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()


    def _build(self):
        super(OVOQ, self)._build()
        # self._set_env()
        # self._set_buffers()
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



    def _set_buffers(self):
        device = self._device_
        seed = self.seed

        # Trajectory Buffer
        horizon = 1000
        max_size_ov = self.configs['data']['ov_buffer_size']
        num_traj = max_size_ov//10
        gamma = self.configs['critic-v']['gamma']
        gae_lam = self.configs['critic-v']['gae_lam']
        self.traj_buffer = TrajBuffer(self.obs_dim, self.act_dim, horizon, num_traj, max_size_ov, seed, device, gamma, gae_lam)

        # Replay Buffer
        max_size_oq = self.configs['data']['oq_buffer_size']
        self.repl_buffer = ReplayBuffer(self.obs_dim, self.act_dim, max_size_oq, seed, device)


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Niv = self.configs['algorithm']['learning']['ov_init_epochs']
        Niq = self.configs['algorithm']['learning']['oq_init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        VNF = self.configs['algorithm']['learning']['ov_N_freq']
        VEF = self.configs['algorithm']['learning']['ov_E_freq']
        GV = self.configs['algorithm']['learning']['grad_OV_steps']
        GPPO = self.configs['algorithm']['learning']['grad_PPO_steps']
        GQ = self.configs['algorithm']['learning']['grad_OQ_SAC_steps']
        max_dev = self.configs['actor']['max_dev']

        global_step = 0
        start_time = time.time()
        logs = dict()
        lastEZ, lastES = 0, -2
        t = 0
        o, Z, el, t = self.learn_env.reset(), 0, 0, 0

        start_time_real = time.time()
        for n in range(1, N+1):

            if self.configs['experiment']['print_logs']:
                print('=' * 50)
                if n > Nx:
                    if n > Niq:
                        if n > Niv and (n%VNF==0):
                            print(f'\n[ Epoch {n}   Learning (OV+OQ) ]'+(' '*50))
                            JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [], [], [], [], [], [], []
                            HVList, HQList, DevList = [], [], []
                            LogPiList, LogPiVList, LogPiQList = [], [], []
                        else:
                            print(f'\n[ Epoch {n}   Learning (OQ) ]'+(' '*50))
                            JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [0], [], [0], [0], [], [0], [0], []
                            HVList, HQList, DevList = [0], [], [0]
                            LogPiList, LogPiVList, LogPiQList = [0], [0], []
                    else:
                        print(f'\n[ Epoch {n}   Learning (OV) ]'+(' '*50))
                        JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [0], [], [], [0], [], [], [0]
                        HVList, HQList, DevList = [], [0], []
                        LogPiList, LogPiVList, LogPiQList = [], [], [0]
                    oldJs = [0, 0, 0, 0, 0]
                elif n > Niv and (n%VNF==0):
                    if n > Niq:
                        print(f'\n[ Epoch {n}   Exploration + Learning (OV+OQ) ]'+(' '*50))
                        JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [], [], [], [], [], [], []
                        HVList, HQList, DevList = [], [], []
                        LogPiList, LogPiVList, LogPiQList = [], [], []
                    else:
                        print(f'\n[ Epoch {n}   Exploration + Learning (OV) ]'+(' '*50))
                        JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [0], [], [], [0], [], [], [0]
                        HVList, HQList, DevList = [], [0], []
                        LogPiList, LogPiVList, LogPiQList = [], [], [0]
                    oldJs = [0, 0, 0, 0, 0]
                elif n > Niq:
                    if n > Niv and (n%VNF==0):
                        print(f'\n[ Epoch {n}   Exploration + Learning (OV+OQ) ]'+(' '*50))
                        JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [], [], [], [], [], [], [], []
                        HVList, HQList, DevList = [], [], []
                        LogPiList, LogPiVList, LogPiQList = [], [], []
                    else:
                        print(f'\n[ Epoch {n}   Exploration + Learning (OQ) ]'+(' '*50))
                        JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [0], [], [0], [0], [], [0], [0], []
                        HVList, HQList, DevList = [0], [], [0]
                        LogPiList, LogPiVList, LogPiQList = [0], [0], []
                    oldJs = [0, 0, 0, 0, 0]
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]'+(' '*50))
                    JVList, JQList, JPiList, JPiVList, JPiQList, KLList, JHOVList, JHOQList = [0], [0], [0], [0], [0], [0], [0], [0]
                    HVList, HQList, DevList = [0], [0], [0]
                    LogPiList, LogPiVList, LogPiQList = [0], [0], [0]
                    oldJs = [0, 0, 0, 0, 0]

            nt = 0
            # o, d, Z, el = self.learn_env.reset(), 0, 0, 0
            ZList, elList = [0], [0]
            ZListImag, elListImag = [0, 0], [0, 0]
            AvgZ, AvgEL = 0, 0
            ppo_grads, sac_grads = 0, 0

            if n > Niq:
                on_policy = False if (n%2==0) else True
                OP = 'on-policy' if on_policy else 'off-policy'
            else:
                on_policy = True
                OP = 'on-policy'

            # on_policy = False
            # OP = 'off-policy'
            # # on_policy = True
            # # OP = 'on-policy'

            learn_start_real = time.time()
            while nt < NT: # full epoch
                # Interaction steps
                for e in range(1, 0+1):
                    # print('OQ, el: ', el)
                    # o, Z, el, t = self.internact_ovoq(n, o, Z, el, t, on_policy=on_policy)
                    # o, Z, el, t = self.internact_ovoq(n, o, Z, el, t, on_policy=True)
                    o, Z, el, t = self.internact_ovoq(n, o, Z, el, t, on_policy=False)

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
                    # SAC >>>>
                    for gq in range(1, GQ+1):
                        print(f'[ Epoch {n} | {color.BLUE}Training AQ{color.END} ] Env Steps ({OP}): {nt+1} | GQ Grads: {gq}/{GQ} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)} | x{round(AvgZ/AvgEL, 2)}'+(" "*5), end='\r')
                        ## Sample a batch B_sac
                        batch_size = self.configs['data']['oq_batch_size'] # bs
                        B_OQ = self.repl_buffer.sample_batch(batch_size, device=self._device_)
                        ## Train networks using batch B_sac
                        _, Jq, Jalpha, Jpi, PiInfo = self.trainAC(gq, B_OQ, oldJs, on_policy=False)
                        oldJs = [0, Jq, Jalpha, Jpi]
                        JQList.append(Jq)
                        JPiQList.append(Jpi)
                        HQList.append(PiInfo['entropy'])
                        LogPiQList.append(PiInfo['log_pi'])
                        if self.configs['actor']['automatic_entropy']:
                            JAlphaList.append(Jalpha.item())
                            AlphaList.append(self.alpha)
                        sac_grads += 1
                    # SAC <<<<

                nt += E
                # if (n%20==0): print('Pi: ', self.actor_critic.actor.act)

                if (n > Niv) and ((n%VNF == 0) and (nt%VEF == 0)):

                    # EZ, ES, EL = self.evaluate()
                    #
                    # print(color.GREEN+f'[ Epoch {n} ] Inner Evaluation | Z={round(np.mean(EZ), 2)}±{round(np.std(EZ), 2)} | L={round(np.mean(EL), 2)}±{round(np.std(EL), 2)} | x{round(np.mean(EZ)/np.mean(EL), 2)}'+color.END+(' ')*40+'\n')

                    # self.traj_buffer.reset()
                    # ov, dv, Zv, elv = self.traj_env.reset(), 0, 0, 0
                    # ZListImag, elListImag = [0], [0]

                    # for ev in range(1, 10000+1):
                    #     ov, Zv, elv, _ = self.internact_ovoq(n, ov, Zv, elv, t, on_policy=True)
                    #     if elv > 0:
                    #         currZ = Zv
                    #         AvgZ = (sum(ZListImag)+currZ)/(len(ZListImag))
                    #         currEL = elv
                    #         AvgEL = (sum(elListImag)+currEL)/(len(elListImag))
                    #     else:
                    #         lastZ = currZ
                    #         ZListImag.append(lastZ)
                    #         AvgZ = sum(ZListImag)/(len(ZListImag)-1)
                    #         lastEL = currEL
                    #         elListImag.append(lastEL)
                    #         AvgEL = sum(elListImag)/(len(elListImag)-1)
                    #     print(f'[ Epoch {n}   OV Interaction ] Env Steps: {ev} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}'+(" "*40), end='\r')
                    # with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(ov)).cpu()
                    # self.traj_buffer.finish_path(elv, v)

                    # PPO >>>>
                    for gv in range(1, GV+1):
                        # Reset model buffer
                        self.traj_buffer.reset()
                        ov, dv, Zv, elv = self.traj_env.reset(), 0, 0, 0
                        ZListImag, elListImag = [0], [0]
                        # # Generate M k-steps imaginary rollouts for PPO training
                        # # ZListImag, elListImag = self.rollout_ov_world_trajectories(gv, n)
                        for ev in range(1, 10000+1):
                            ov, Zv, elv, _ = self.internact_ovoqii(n, ov, Zv, elv, t, on_policy=True)
                            if elv > 0:
                                currZ = Zv
                                AvgZ = (sum(ZListImag)+currZ)/(len(ZListImag))
                                currEL = elv
                                AvgEL = (sum(elListImag)+currEL)/(len(elListImag))
                            else:
                                lastZ = currZ
                                ZListImag.append(lastZ)
                                AvgZ = sum(ZListImag)/(len(ZListImag)-1)
                                lastEL = currEL
                                elListImag.append(lastEL)
                                AvgEL = sum(elListImag)/(len(elListImag)-1)
                            print(f'[ Epoch {n} | {color.RED}OV ({gv}/{GV}) Interaction{color.END} ] Env Steps: {ev} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)} | x{round(AvgZ/AvgEL, 2)}'+(" "*20), end='\r')
                        with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(ov)).cpu()
                        self.traj_buffer.finish_path(elv, v)

                        ppo_batch_size = int(self.traj_buffer.total_size())
                        sac_batch_size = self.configs['data']['oq_batch_size']
                        kl, dev = 0, 0
                        stop_pi = False

                        # # Version A
                        # for gg in range(1, GPPO+1): # 101
                        #     print(f"[ Epoch {n} | {color.PURPLE}Training AV+AQ{color.END} ] GV: {gv}/{GV} | ac: {gg}/{GPPO} || stopPG={stop_pi} | Dev={round(dev, 4)}"+(" "*30), end='\r')
                        #     ppo_batch = self.traj_buffer.sample_batch(batch_size=ppo_batch_size, device=self._device_)
                        #     Jv, _, _, Jpiv, PiInfov = self.trainAC(gv, ppo_batch, oldJs, on_policy=True)
                        #     sac_batch = self.repl_buffer.sample_batch(batch_size=sac_batch_size, device=self._device_)
                        #     _,  Jq, _, Jpiq, PiInfoq = self.trainAC(gv, sac_batch, oldJs, on_policy=False)
                        #     oldJs = [Jv, Jq, 0, Jpiv, Jpiq]
                        #     JVList.append(Jv)
                        #     JQList.append(Jq)
                        #     JPiVList.append(Jpiv)
                        #     JPiQList.append(Jpiq)
                        #     LogPiVList.append(PiInfov['log_pi'])
                        #     LogPiQList.append(PiInfoq['log_pi'])
                        #     DevList.append(PiInfov['deviation'])
                        #     dev = PiInfov['deviation']
                        #     if not PiInfov['stop_pi']:
                        #         ppo_grads += 1
                        #     stop_pi = PiInfov['stop_pi']

                        # Version B
                        for gg in range(1, GPPO+1): # 101
                            print(f"[ Epoch {n} | {color.PURPLE}Training AV+AQ{color.END} ] GV: {gv}/{GV} | ac: {gg}/{GPPO} || stopPG={stop_pi} | Dev={round(dev, 4)}"+(" "*30), end='\r')
                            ppo_batch = self.traj_buffer.sample_batch(batch_size=ppo_batch_size, device=self._device_)
                            Jv, Jq, Jalpha, Jpi, PiInfo = self.trainACDual(gv, ppo_batch, oldJs, on_policy=True)
                            oldJs = [Jv, Jq, Jpi, 0]
                            JVList.append(Jv)
                            JQList.append(Jq)
                            JPiList.append(Jpi)
                            LogPiList.append(PiInfo['log_pi'])
                            DevList.append(PiInfo['deviation'])
                            dev = PiInfo['deviation']
                            if not PiInfo['stop_pi']:
                                ppo_grads += 1
                            stop_pi = PiInfo['stop_pi']


                        # Version C
                        # for gg in range(1, GPPO+1): # 101
                        #     print(f"[ Epoch {n} | {color.RED}Training AV{color.END} ] GV: {gv}/{GV} | ac: {gg}/{GPPO} || stopPG={stop_pi} | Dev={round(dev, 4)}"+(" "*30), end='\r')
                        #     ppo_batch = self.traj_buffer.sample_batch(batch_size=ppo_batch_size, device=self._device_)
                        #     Jv, _, _, Jpiv, PiInfov = self.trainAC(gv, ppo_batch, oldJs, on_policy=True)
                        #     oldJs = [Jv, 0, 0, Jpiv, 0]
                        #     JVList.append(Jv)
                        #     JPiVList.append(Jpiv)
                        #     LogPiVList.append(PiInfov['log_pi'])
                        #     DevList.append(PiInfov['deviation'])
                        #     dev = PiInfov['deviation']
                        #     if not PiInfov['stop_pi']:
                        #         ppo_grads += 1
                        #     stop_pi = PiInfov['stop_pi']
                        #
                        #
                        # for gg in range(1, 10+1): # 101
                        #     print(f"[ Epoch {n} | {color.PURPLE}Training AQ{color.END} ] GV: {gv}/{GV} | ac: {gg}/{10} || stopPG={stop_pi} | Dev={round(dev, 4)}"+(" "*30), end='\r')
                        #     sac_batch = self.repl_buffer.sample_batch(batch_size=sac_batch_size, device=self._device_)
                        #     _, Jq, _, Jpiq, PiInfoq = self.trainAC(gv, sac_batch, oldJs, on_policy=False)
                        #     oldJs = [Jv, Jq, 0, Jpiv, Jpiq]
                        #     JQList.append(Jq)
                        #     JPiQList.append(Jpiq)
                        #     LogPiQList.append(PiInfoq['log_pi'])

                    # PPO <<<<




            print('\n')

            logs['training/ovoq/critic/Jv             '] = np.mean(JVList)
            logs['training/ovoq/critic/V(s)           '] = T.mean(self.traj_buffer.val_buf).item()
            logs['training/ovoq/critic/V-R            '] = T.mean(self.traj_buffer.val_buf).item()-T.mean(self.traj_buffer.ret_buf).item()
            logs['training/ovoq/critic/Jq             '] = np.mean(JQList)

            logs['training/ovoq/actor/Jpi             '] = np.mean(JPiList)
            # logs['training/ovoq/actor/Jpi_ov          '] = np.mean(JPiVList)
            # logs['training/ovoq/actor/Jpi_oq          '] = np.mean(JPiQList)
            logs['training/ovoq/actor/STD             '] = self.actor_critic.actor.std_value.clone().mean().item()
            logs['training/ovoq/actor/log_pi          '] = np.mean(LogPiList)
            # logs['training/ovoq/actor/log_pi-v        '] = np.mean(LogPiVList)
            # logs['training/ovoq/actor/log_pi-q        '] = np.mean(LogPiQList)
            # logs['training/ovoq/actor/HV              '] = np.mean(HVList)
            # logs['training/ovoq/actor/HQ              '] = np.mean(HQList)
            # logs['training/ovoq/actor/ov-KL           '] = np.mean(KLList) #
            logs['training/ovoq/actor/ov-deviation    '] = np.mean(DevList)
            logs['training/ovoq/actor/ppo-grads       '] = ppo_grads
            logs['training/ovoq/actor/sac-grads       '] = sac_grads

            logs['data/real/on-policy                 '] = int(on_policy)
            logs['data/real/repl_buffer_size          '] = self.repl_buffer.size
            logs['data/real/traj_buffer_size          '] = self.traj_buffer.total_size()
            # if hasattr(self, 'traj_buffer') and hasattr(self, 'repl_buffer'):
            #     logs['data/imag/ov-traj_buffer_size       '] = self.traj_buffer.total_size()
            #     logs['data/imag/oq-repl_buffer_size       '] = self.repl_buffer.size
            # elif hasattr(self, 'mraj_buffer'):
            #     logs['data/imag/ov-traj_buffer_size       '] = self.traj_buffer.total_size()
            #     logs['data/imag/oq-repl_buffer_size       '] = 0.
            # elif hasattr(self, 'repl_buffer'):
            #     logs['data/imag/ov-traj_buffer_size       '] = 0.
            #     logs['data/imag/oq-repl_buffer_size       '] = self.repl_buffer.size
            # else:
            #     logs['data/imag/ov-traj_buffer_size       '] = 0.
            #     logs['data/imag/oq-repl_buffer_size       '] = 0.

            logs['learning/real/rollout_return_mean   '] = np.mean(ZList[1:])
            logs['learning/real/rollout_return_std    '] = np.std(ZList[1:])
            logs['learning/real/rollout_length        '] = np.mean(elList[1:])

            logs['learning/ov-img/rollout_return_mean '] = np.mean(ZListImag[1:])
            logs['learning/ov-img/rollout_return_std  '] = np.std(ZListImag[1:])
            logs['learning/ov-img/rollout_length      '] = np.mean(elListImag[1:])

            eval_start_real = time.time()
            EZ, ES, EL = self.evaluate(on_policy=True)

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
                # for i in range(int(E/1000)):
                # for i in range(10):
                wandb.log(logs)

        self.learn_env.close()
        self.traj_env.close()
        self.eval_env.close()



    def rollout_ov_world_trajectories(self, g, n):
    	# 07. Sample st uniformly from Denv
    	device = self._device_
    	Nτ = 250
    	K = 1000

        # 08. Perform k-step model rollout starting from st using policy πφ; add to Dmodel
    	k_end_total = 0
    	ZList, elList = [0], [0]
    	AvgZ, AvgEL = 0, 0

    	for nτ in range(1, Nτ+1): # Generate trajectories
            o, Z, el = self.traj_env.reset(), 0, 0
            for k in range(1, K+1): # Generate rollouts
                print(f'[ Epoch {n} | {color.RED}AV {g} | V-Model Rollout{color.END} ] nτ = {nτ+1} | k = {k}/{K} | Buffer = {self.traj_buffer.total_size()} | AvgZ={round(AvgZ, 2)} | AvgEL={round(AvgEL, 2)}', end='\r')
                with T.no_grad(): a, log_pi, _, v = self.actor_critic.get_a_and_v(o)

                o_next, r, d, _ = self.traj_env.step(a)
                Z += r
                el += 1

                Z += float(r)
                el += 1
                self.traj_buffer.store(o, a, r, o_next, v, log_pi, el)
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
            self.traj_buffer.finish_path(el, v)

            k_end_total += k

            lastZ = currZ
            ZList.append(lastZ)
            AvgZ = sum(ZList)/(len(ZList)-1)
            lastEL = currEL
            elList.append(lastEL)
            AvgEL = sum(elList)/(len(elList)-1)

            if self.traj_buffer.total_size() >= self.configs['data']['ov_buffer_size']:
                # print(f'Breaking img rollouts at nτ={nτ+1}/m={m+1} | Buffer = {self.traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | EL={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*85)
                break
    	print(f'[ Epoch {n} | AC {g} ] RollBuffer={self.traj_buffer.total_size()} | Z={round(np.mean(ZList[1:]), 2)}±{round(np.std(ZList[1:]), 2)} | L={round(np.mean(elList[1:]), 2)}±{round(np.std(elList[1:]), 2)} | x{round(np.mean(ZList[1:])/np.mean(elList[1:]), 2)}'+(' ')*35)

    	return ZList, elList



    def trainAC(self, g, batch, oldJs, on_policy=True):
        TUI = self.configs['algorithm']['learning']['target_update_interval']

        if on_policy:
            Jv = self.updateV(batch, oldJs[0])
            Jpi, PiInfo = self.updatePi(batch, oldJs[3], on_policy=True)# if (g % PUI == 0) else oldJs[2]
            Jv = Jv.item()
            Jq = oldJs[1]
            Jalpha = oldJs[2]
            Jpi = Jpi.item()
        else: # off-policy
            Jq = self.updateQ(batch, oldJs[1])
            Jalpha = self.updateAlpha(batch, oldJs[2])# if (g % AUI == 0) else oldJs[1]
            Jpi, PiInfo = self.updatePi(batch, oldJs[4], on_policy=False)# if (g % PUI == 0) else oldJs[2]
            Jv = oldJs[0]
            Jq = Jq.item()
            # Jalpha = Jalpha.item()
            Jpi = Jpi.item()
            if g % TUI == 0:
                self.updateTarget()

        return Jv, Jq, Jalpha, Jpi, PiInfo


    def updateV(self, batch, Jv_old):
        """"
        Jv(θ) =
        """
        # max_grad_norm = kl_targ = self.configs['critic-v']['network']['max_grad_norm']

        O, _, _, _, _, _, Z, _, _, _ = batch.values()
        V = self.actor_critic.get_v(O)

        Jv = 0.5 * ( (V - Z) ** 2 ).mean(axis=0)

        self.actor_critic.ov.optimizer.zero_grad()
        Jv.backward()
        # nn.utils.clip_grad_norm_(self.actor_critic.ov.parameters(), max_grad_norm) # PPO-D
        self.actor_critic.ov.optimizer.step()

        return Jv


    def updateQ(self, batch, Jq_old):
        """"
        JQ(θ) = E(st,at)∼D[ 0.5 ( Qθ(st, at)
                            − r(st, at)
                            + γ Est+1∼D[ Eat+1~πφ(at+1|st+1)[ Qθ¯(st+1, at+1)
                                                − α log(πφ(at+1|st+1)) ] ] ]
        """
        gamma = self.configs['critic-q']['gamma']

        O = batch['observations']
        A = batch['actions']
        R = batch['rewards']
        O_next = batch['observations_next']
        D = batch['terminals']

        # Calculate two Q-functions
        Qs = self.actor_critic.get_q(O, A)

        # Bellman backup for Qs
        with T.no_grad():
            pi_next, log_pi_next, entropy_next = self.actor_critic.get_pi(O_next, on_policy=False, reparameterize=True, return_log_pi=True)
            # pi_next, log_pi_next, entropy_next = self.actor_critic.get_pi(O_next, on_policy=False, reparameterize=False, return_log_pi=True)
            A_next = pi_next
            Qs_targ = T.cat( self.actor_critic.get_q_target(O_next, A_next), dim=1 )
            min_Q_targ, _ = T.min(Qs_targ, dim=1, keepdim=True)
            Qs_backup = R + gamma * (1 - D) * (min_Q_targ - self.alpha * log_pi_next)

        # MSE loss
        Jq = 0.5 * sum([F.mse_loss(Q, Qs_backup) for Q in Qs])
        # print('Jq=', Jq)

        # Gradient Descent
        self.actor_critic.oq.optimizer.zero_grad()
        Jq.backward()
        self.actor_critic.oq.optimizer.step()

        return Jq


    def updateAlpha(self, batch, Jalpha_old):
        """

        αt* = arg min_αt Eat∼πt∗[ −αt log( πt*(at|st; αt) ) − αt H¯

        """
        if self.configs['actor']['automatic_entropy']:
            # Learned Temprature
            O, _, _, _, _, _, _, _, _ = batch.values()

            with T.no_grad():
                _, log_pi, _ = self.actor_critic.get_pi(O, on_policy=False, return_log_pi=True)
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


    def updatePi(self, batch, Jpi_old, on_policy=True):
        """
        Jπ(φ) =
        """
        PiInfo = dict()

        constrained = self.configs['actor']['constrained']

        clip_eps = self.configs['actor']['clip_eps']
        entropy_coef = self.configs['actor']['entropy_coef']
        # max_grad_norm = self.configs['actor']['network']['max_grad_norm']
        kl_targ = self.configs['actor']['kl_targ']
        max_dev = self.configs['actor']['max_dev']

        # O, A, _, _, _, _, _, U, log_pis_old = batch.values()

        if on_policy:
            O, pre_A, A, _, _, _, _, _, U, log_pis_old = batch.values()
            _, log_pi, entropy = self.actor_critic.get_pi(O, pre_A)
            logratio = log_pi - log_pis_old
            ratio = logratio.exp()
            with T.no_grad():
                approx_kl_old = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                deviation = ((ratio - 1).abs()).mean()
            clipped_ratio = T.clamp(ratio, 1-clip_eps, 1+clip_eps)
            Jpg = - ( T.min(ratio * U, clipped_ratio * U) ).mean(axis=0)
            Jentropy = - entropy_coef * entropy.mean()
            Jpi = Jpg + Jentropy
            if (constrained) and (deviation > max_dev):
                stop_pi = True
            else:
                stop_pi = False
                self.actor_critic.actor.optimizer.zero_grad()
                Jpi.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), max_grad_norm) # PPO-D
                self.actor_critic.actor.optimizer.step()

            PiInfo['KL'] = approx_kl_old
            PiInfo['ratio'] = ratio.mean().item()
            PiInfo['deviation'] = deviation.item()
            PiInfo['stop_pi'] = stop_pi

        else: # Off-Policy
            O, _, _, _, _ = batch.values()
            # Policy Evaluation
            pi, log_pi, entropy = self.actor_critic.get_pi(O, on_policy=False, reparameterize=True, return_log_pi=True)
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


    def trainACDual(self, g, batch, oldJs, on_policy=True):
        TUI = self.configs['algorithm']['learning']['target_update_interval']

        Jv, Jq = self.updateVQ(batch, oldJs[0], oldJs[1])
        Jalpha = self.updateAlpha(batch, oldJs[2])# if (g % AUI == 0) else oldJs[1]
        Jpi, PiInfo = self.updatePiDual(batch, oldJs[3], on_policy=True)# if (g % PUI == 0) else oldJs[2]
        Jv = Jv.item()
        Jq = Jq.item()
        Jpi = Jpi.item()
        if g % TUI == 0:
            self.updateTarget()

        return Jv, Jq, Jalpha, Jpi, PiInfo


    def updateVQ(self, batch, Jv_old, Jq_old):
        """"
        Jv(θ) =
        """
        # max_grad_norm = kl_targ = self.configs['critic-v']['network']['max_grad_norm']
        gamma = self.configs['critic-q']['gamma']

        O, _, A, O_next, R, D, Z, _, _, _ = batch.values()

        V = self.actor_critic.get_v(O)
        Jv = 0.5 * ( (V - Z) ** 2 ).mean(axis=0)

        # Calculate two Q-functions
        Qs = self.actor_critic.get_q(O, A)
        # Bellman backup for Qs
        with T.no_grad():
            pi_next, log_pi_next, entropy_next = self.actor_critic.get_pi(O_next, on_policy=False, reparameterize=True, return_log_pi=True)
            A_next = pi_next
            Qs_targ = T.cat( self.actor_critic.get_q_target(O_next, A_next), dim=1 )
            min_Q_targ, _ = T.min(Qs_targ, dim=1, keepdim=True)
            Qs_backup = R + gamma * (1 - D) * (min_Q_targ - self.alpha * log_pi_next)
        # MSE loss
        Jq = 0.5 * sum([F.mse_loss(Q, Qs_backup) for Q in Qs])

        self.actor_critic.ov.optimizer.zero_grad()
        Jv.backward()
        # nn.utils.clip_grad_norm_(self.actor_critic.ov.parameters(), max_grad_norm) # PPO-D
        self.actor_critic.ov.optimizer.step()

        # Gradient Descent
        self.actor_critic.oq.optimizer.zero_grad()
        Jq.backward()
        self.actor_critic.oq.optimizer.step()

        return Jv, Jq


    def updatePiDual(self, batch, Jpi_old, on_policy=True):
        """
        Jπ(φ) =
        """
        PiInfo = dict()

        constrained = self.configs['actor']['constrained']

        clip_eps = self.configs['actor']['clip_eps']
        entropy_coef = self.configs['actor']['entropy_coef']
        # max_grad_norm = self.configs['actor']['network']['max_grad_norm']
        kl_targ = self.configs['actor']['kl_targ']
        max_dev = self.configs['actor']['max_dev']

        O, pre_A, A, _, _, _, _, _, U, log_pis_old = batch.values()

        _, log_pi, entropy = self.actor_critic.get_pi(O, pre_A)
        logratio = log_pi - log_pis_old
        ratio = logratio.exp()

        with T.no_grad():
            approx_kl_old = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            deviation = ((ratio - 1).abs()).mean()

        clipped_ratio = T.clamp(ratio, 1-clip_eps, 1+clip_eps)
        Jpg = - ( T.min(ratio * U, clipped_ratio * U) ).mean(axis=0)
        Jentropy = - entropy_coef * entropy.mean()
        Jppo = Jpg + Jentropy

        pi, log_pi, entropy = self.actor_critic.get_pi(O)
        Qs_pi = T.cat(self.actor_critic.get_q(O, pi), dim=1)
        min_Q_pi, _ = T.min(Qs_pi, dim=1, keepdim=True)
        Jsac = (self.alpha * log_pi - min_Q_pi).mean()

        Jpi = Jppo + 0.05*Jsac

        if (constrained) and (deviation > max_dev):
            stop_pi = True
        else:
            stop_pi = False
            self.actor_critic.actor.optimizer.zero_grad()
            Jpi.backward()
            # nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), max_grad_norm) # PPO-D
            self.actor_critic.actor.optimizer.step()

        PiInfo['KL'] = approx_kl_old
        PiInfo['ratio'] = ratio.mean().item()
        PiInfo['deviation'] = deviation.item()
        PiInfo['stop_pi'] = stop_pi

        PiInfo['entropy'] = entropy.mean().item()
        PiInfo['log_pi'] = log_pi.mean().item()

        return Jpi, PiInfo


    def updateTarget(self):
        # print('updateTarget')
        tau = self.configs['critic-q']['tau']
        with T.no_grad():
            for p, p_targ in zip(self.actor_critic.oq.parameters(),
                                 self.actor_critic.oq_target.parameters()):
                p_targ.data.copy_(tau * p.data + (1 - tau) * p_targ.data)







def main(exp_prefix, config, seed, device, wb):

    print('Start an OVOQ-II experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']

    group_name = f"{env_name}-{alg_name}-15"
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

    agent = OVOQ(exp_prefix, configs, seed, device, wb)

    agent.learn()

    print('\n')
    print('... End the OVOQ experiment')

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
