"""
Inspired by:

    1. RLKit: https://github.com/rail-berkeley/rlkit [X]
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
from torch.distributions.normal import Normal
nn = T.nn

from rl.algorithms.mfrl.mfrl import MFRL
from rl.control.policy import PPOPolicy, StochasticPolicy, OVOQPolicy
# from rl.control.policy import NPGPolicy
from rl.value_functions.v_function import VFunction


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
        and a critic (V-function) that evaluate that state given a policy.
    """
    def __init__(self,
                 obs_dim, act_dim,
                 act_up_lim, act_low_lim,
                 configs, seed, device
                 ) -> None:
        print('Initialize AC!')
        # super(ActorCritic, self).__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.act_up_lim, self.act_low_lim = act_up_lim, act_low_lim
        self.configs, self.seed = configs, seed
        self._device_ = device

        self.actor, self.critic = None, None
        self._build()


    def _build(self):
        self.actor = self._set_actor()
        self.critic = self._set_critic()


    def _set_actor(self):
        net_configs = self.configs['actor']['network']
        return PPOPolicy(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            net_configs, self._device_, self.seed)
        # return StochasticPolicy(
        #     self.obs_dim, self.act_dim,
        #     self.act_up_lim, self.act_low_lim,
        #     net_configs, self._device_, self.seed)
        # return OVOQPolicy(
        #     self.obs_dim, self.act_dim,
        #     self.act_up_lim, self.act_low_lim,
        #     net_configs, self._device_, self.seed)


    def _set_critic(self):
        net_configs = self.configs['critic']['network']
        return VFunction(
            self.obs_dim, self.act_dim,
            net_configs, self._device_, self.seed)


    def get_v(self, o):
        return self.critic(o)


    def get_pi(self, o, a=None,
               on_policy=True,
               reparameterize=False,
               deterministic=False,
               return_log_pi=True,
               return_entropy=True):
        action, log_pi, entropy = self.actor(o, a, on_policy,
                                             reparameterize,
                                             deterministic,
                                             return_log_pi,
                                             return_entropy)
        return action, log_pi, entropy


    def get_action(self, o, a=None,
                   on_policy=True,
                   reparameterize=False,
                   deterministic=False,
                   return_log_pi=True,
                   return_entropy=True):
        o = T.Tensor(o)
        if a: a = T.Tensor(a)
        with T.no_grad(): a, _, _ = self.actor(o, a, on_policy,
                                               reparameterize,
                                               deterministic,
                                               return_log_pi,
                                               return_entropy)
        return a.cpu()


    def get_action_np(self, o, a=None,
                      on_policy=True,
                      reparameterize=False,
                      deterministic=False,
                      return_log_pi=True,
                      return_entropy=True):
        return self.get_action(o, a, on_policy,
                               reparameterize,
                               deterministic,
                               return_log_pi,
                               return_entropy).numpy()


    def get_pi_and_v(self, o, a=None,
                    on_policy=True,
                    reparameterize=False,
                    deterministic=False,
                    return_log_pi=True,
                    return_entropy=True):
        pi, log_pi, entropy = self.actor(o, a, on_policy,
                                         reparameterize,
                                         deterministic,
                                         return_log_pi,
                                         return_entropy)
        return pi, log_pi, entropy, self.critic(o)


    def get_a_and_v(self, o, a=None,
                    on_policy=True,
                    reparameterize=False,
                    deterministic=False,
                    return_log_pi=True,
                    return_entropy=True):
        action, log_pi, entropy = self.actor(o, a, on_policy,
                                             reparameterize,
                                             deterministic,
                                             return_log_pi,
                                             return_entropy)
        return action.cpu(), log_pi.cpu(), entropy, self.critic(o).cpu()


    def get_a_and_v_np(self, o, a=None,
                       on_policy=True,
                       reparameterize=False,
                       deterministic=False,
                       return_log_pi=True,
                       return_entropy=True):
        o = T.Tensor(o)
        if a: a = T.Tensor(a)
        with T.no_grad(): a, log_pi, entropy = self.actor(o, a, on_policy,
                                                          reparameterize,
                                                          deterministic,
                                                          return_log_pi,
                                                          return_entropy)
        return a.cpu().numpy(), log_pi.cpu().numpy(), self.critic(o).cpu().numpy()





class PPO(MFRL):
    """
    Algorithm: Proximal Policy Optimization (On-Policy, Model-Free)

        01. Initialize: Models parameters( Policy net πφ, Value net Vψ )
        02. Initialize: Trajectory buffer Dτ = {}
        03. Hyperparameters: Disc. factor γ, GAE λ, num traj's Nτ, rollout horizon H
        04. for n = 0, 1, 2, ..., N:
        05.     Collect set of traj's {τ^πk} by πk = π(φk) for e = 0, 1, 2, ..., E
        06.     Aggregate the traj's in traj buffer, Dτ = Dτ U {τ^πk}
        07.     Compute RTG R^_t, GAE A^_t based on Vk = V(θk)
        08.     for g = 0, 1, 2, ..., G do
        09.         Update πφ by maxz Jπ
                        φ = arg max_φ {(1/T|Dk|) sum sum min((π/πk), 1 +- eps) Aπk }
        10.         Fit Vθ by MSE(Jv)
                        ψ = arg min_ψ {(1/T|Dk|) sum sum (Vψ(st) - RTG)^2 }
        11.     end for
        12. end for
    """
    def __init__(self, exp_prefix, configs, seed, device, wb) -> None:
        super(PPO, self).__init__(exp_prefix, configs, seed, device)
        print('Initialize PPO Algorithm!')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()


    def _build(self):
        super(PPO, self)._build()
        self._build_ppo()


    def _build_ppo(self):
        self._set_actor_critic()


    def _set_actor_critic(self):
        self.actor_critic = ActorCritic(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.configs, self.seed, self._device_)
        # self.actor_critic = ActorCriticII(
        #     self.obs_dim, self.act_dim,
        #     self.act_up_lim, self.act_low_lim,
        #     self.configs, self.seed, self._device_)


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        F = self.configs['algorithm']['learning']['train_AC_freq']
        G = self.configs['algorithm']['learning']['grad_AC_steps']

        batch_size = self.configs['data']['batch_size']
        # mini_batch_size = self.configs['data']['mini_batch_size']

        max_dev = self.configs['actor']['max_dev']


        global_step = 0
        start_time = time.time()
        t = 0
        # o, d, Z, el, t = self.learn_env.reset(), 0, 0, 0, 0
        oldJs = [0, 0]
        logs = dict()
        lastEZ, lastES = 0, -2
        # KLrange = np.linspace(0.025, 0.0025, 200)
        # num_traj = 1000
        # stop_pi = False
        pg = 0

        start_time_real = time.time()
        for n in range(1, N + 1):
            n_real = int(n*NT/1000)
            if self.configs['experiment']['print_logs']:
                print('=' * 50)
                if n > Nx:
                    print(f'\n[ Epoch {n_real}   Learning ]'+(' '*50))
                    oldJs = [0, 0]
                    JVList, JPiList, KLList = [], [], []
                    HList, DevList, LogPiList = [], [], []
                    ho_mean = 0
                elif n > Ni:
                    print(f'\n[ Epoch {n_real}   Exploration + Learning ]'+(' '*50))
                    JVList, JPiList, KLList = [], [], []
                    HList, DevList, LogPiList = [], [], []
                else:
                    print(f'\n[ Epoch {n_real}   Inintial Exploration ]'+(' '*50))
                    oldJs = [0, 0]
                    JVList, JPiList, KLList = [0], [0], [0]
                    HList, DevList, LogPiList = [0], [0], [0]
                    ho_mean = 0


            nt = 0
            self.buffer.reset()
            o, d, Z, el, = self.learn_env.reset(), 0, 0, 0
            ZList, elList = [0], [0]
            ZListImag, elListImag = [0, 0], [0, 0]
            AvgZ, AvgEL = 0, 0
            ppo_grads = 0

            learn_start_real = time.time()
            while nt < NT:
                # Interaction steps (On-Policy)
                for e in range(1, E+1):
                    print('t: ', t, end='\r')
                    # o, d, Z, el, t = self.internact_op(n, o, d, Z, el, t)
                    o, Z, el, t = self.internact_opB(n, o, Z, el, t)
                    # print(f'Steps: e={e} | el={el} || size={self.buffer.total_size()}')

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
                # self.buffer.traj_tail(d, v, el)
                self.buffer.finish_path(el, v)
                # print('self.log_pi_buf: ', self.buffer.log_pi_buf)

                # Taking gradient steps after exploration
                if n > Ni:
                    # Optimizing policy and value networks
                    self.stop_pi = False
                    kl, dev = 0, 0
                    # ppo_grads = 0
                    for g in range(1, G+1):
                        # PPO-P >>>>
                        print(f'[ PPO ] grads={g}/{G} | stopPG={self.stop_pi} | Dev={round(dev, 4)}', end='\r')
                        batch = self.buffer.sample_batch(batch_size, device=self._device_)
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
                        # if PiInfo['deviation'] > max_dev:
                        #     print(f'\nBreak AC grad loop at g={g}'+(' ')*80)
                        #     break
                        # PPO-P <<<<
                    # self.buffer.reset()
                nt += E

            # logs['time/training                       '] = time.time() - learn_start_real
            logs['training/ppo/critic/Jv              '] = np.mean(JVList)
            logs['training/ppo/critic/V(s)            '] = T.mean(self.buffer.val_buf).item()
            logs['training/ppo/critic/V-R             '] = T.mean(self.buffer.val_buf).item()-T.mean(self.buffer.ret_buf).item()
            logs['training/ppo/actor/Jpi              '] = np.mean(JPiList)
            logs['training/ppo/actor/grads            '] = ppo_grads
            logs['training/ppo/actor/H                '] = np.mean(HList)
            logs['training/ppo/actor/KL               '] = np.mean(KLList)
            logs['training/ppo/actor/deviation        '] = np.mean(DevList)
            logs['training/ppo/actor/log_pi           '] = PiInfo['log_pi']

            # logs['time                                '] = t
            # logs['data/average_return                 '] = self.buffer.average_return()
            logs['data/env_buffer                     '] = self.buffer.total_size()
            logs['data/total_interactions             '] = t

            logs['learning/real/rollout_return_mean   '] = np.mean(ZList[1:])
            logs['learning/real/rollout_return_std    '] = np.std(ZList[1:])
            logs['learning/real/rollout_length        '] = np.mean(elList[1:])

            eval_start_real = time.time()
            EZ, ES, EL = self.evaluate_op()
            #
            # logs['time/evaluation                '] = time.time() - eval_start_real
            #
            if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
                logs['evaluation/episodic_score_mean      '] = np.mean(ES)
                logs['evaluation/episodic_score_std       '] = np.std(ES)
            else:
                logs['training/ppo/critic/Jv              '] = np.mean(JVList)
                logs['evaluation/episodic_return_mean     '] = np.mean(EZ)
                logs['evaluation/episodic_return_std      '] = np.std(EZ)
            logs['evaluation/episodic_length_mean     '] = np.mean(EL)
            logs['evaluation/return_to_length         '] = np.mean(EZ)/np.mean(EL)
            logs['evaluation/return_to_full_length    '] = (np.mean(EZ)/1000)

            #
            # logs['time/total                     '] = time.time() - start_time_real
            #
            # if n > (N - 50):
            #     if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
            #         if np.mean(ES) > lastES:
            #             print(f'[ Epoch {n}   Agent Saving ]                    ')
            #             env_name = self.configs['environment']['name']
            #             alg_name = self.configs['algorithm']['name']
            #             T.save(self.actor_critic.actor,
            #             f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pth.tar')
            #             lastES = np.mean(ES)
            #     else:
            #         if np.mean(EZ) > lastEZ:
            #             print(f'[ Epoch {n}   Agent Saving ]                    ')
            #             env_name = self.configs['environment']['name']
            #             alg_name = self.configs['algorithm']['name']
            #             T.save(self.actor_critic.actor,
            #             f'./saved_agents/{env_name}-{alg_name}-seed:{self.seed}-epoch:{n}.pth.tar')
            #             lastEZ = np.mean(EZ)
            #
            # Printing logs
            # Printing logs
            if self.configs['experiment']['print_logs']:
                return_means = ['learning/real/rollout_return_mean   ',
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
                for i in range(int(E/1000)):
                    wandb.log(logs)

        self.learn_env.close()
        self.eval_env.close()


    def trainAC(self, g, batch, oldJs):
        Jv = self.updateV(batch, oldJs[0])
        Jv = Jv.item()
        Jpi, kl, PiInfo = self.updatePi(batch, oldJs[1], g)
        Jpi = Jpi.item()
        kl = kl.item()
        return Jv, Jpi, kl, PiInfo#, deviation, stop_pi


    def updateV(self, batch, Jv_old):
        """"
        Jv(θ) =
        """
        max_grad_norm = kl_targ = self.configs['critic']['network']['max_grad_norm']

        observations, _, _, _, _, returns, _, _, _ = batch.values()
        v = self.actor_critic.get_v(observations)

        Jv = 0.5 * ( (v - returns) ** 2 ).mean(axis=0)

        self.actor_critic.critic.optimizer.zero_grad()
        Jv.backward()
        # nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_grad_norm) # PPO-D
        self.actor_critic.critic.optimizer.step()

        return Jv


    def updatePi(self, batch, Jpi_old, g):
        """
        Jπ(φ) =
        """
        constrained = self.configs['actor']['constrained']
        clip_eps = self.configs['actor']['clip_eps']
        entropy_coef = self.configs['actor']['entropy_coef']
        max_grad_norm = self.configs['actor']['network']['max_grad_norm']
        kl_targ = self.configs['actor']['kl_targ']
        max_dev = self.configs['actor']['max_dev']
        G = self.configs['algorithm']['learning']['grad_AC_steps']

        PiInfo = dict()

        observations, actions, _, _, _, _, _, advantages, log_pis_old = batch.values()

        _, log_pi, entropy = self.actor_critic.get_pi(observations, actions)
        logratio = log_pi - log_pis_old

        ratio = logratio.exp()

        with T.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl_old = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            deviation = ((ratio - 1).abs()).mean()

        clipped_ratio = T.clamp(ratio, 1-clip_eps, 1+clip_eps)
        Jpg = - ( T.min(ratio * advantages, clipped_ratio * advantages) ).mean(axis=0)
        # Jpg = - ( ratio * advantages ).mean(axis=0)
        # Jpi = - ( ratio * advantages + entropy_coef * entropy ).mean(axis=0)

        # Jpg = - (ratio * advantages).mean(axis=0)

        Jentropy = - entropy_coef * entropy.mean()
        Jpi = Jpg + Jentropy

        # if (constrained) and (deviation > max_dev):
        if (constrained) and ((deviation > max_dev) and (g > 0.1*G)):
            self.stop_pi = True
        else:
            self.stop_pi = False
            self.actor_critic.actor.optimizer.zero_grad()
            Jpi.backward()
            # nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), max_grad_norm) # PPO-D
            self.actor_critic.actor.optimizer.step()

        PiInfo['KL'] = approx_kl_old
        # PiInfo['KL-new'] = approx_kl
        PiInfo['entropy'] = entropy.mean().item()
        PiInfo['ratio'] = ratio.mean().item()
        PiInfo['deviation'] = deviation.item()
        PiInfo['log_pi'] = log_pi.mean().item()
        PiInfo['stop_pi'] = self.stop_pi

        return Jpi, approx_kl_old, PiInfo#, stop_pi





def main(exp_prefix, config, seed, device, wb):

    print('Start an PPO experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']

    group_name = f"{env_name}-{alg_name}-ReLU-89" # H < -2.7
    exp_prefix = f"seed:{seed}"

    if wb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # project='test',
            project='AMMI-RL-2022',
            config=configs
        )

    agent = PPO(exp_prefix, configs, seed, device, wb)

    agent.learn()

    print('\n')
    print('... End the PPO experiment')


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
