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
from rl.control.policy import PPOPolicy
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
                 configs, seed, device
                 ) -> None:
        print('Initialize AC!')
        super(ActorCritic, self).__init__()
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


    def _set_critic(self):
        net_configs = self.configs['critic']['network']
        return VFunction(
            self.obs_dim, self.act_dim,
            net_configs, self._device_, self.seed)


    def get_v(self, x):
        return self.critic(x)


    def get_pi(self, x, action=None):
        action, log_pi, entropy = self.actor(x, action)
        return action, log_pi, entropy


    def get_action(self, o, a=None):
        o = T.Tensor(o)
        if a: a = T.Tensor(a)
        with T.no_grad(): action, _, _ = self.actor(T.Tensor(o), a)
        return action.cpu().numpy()


    def get_pi_and_v(self, x, action=None):
        action, log_pi, entropy = self.actor(x, action)
        return action, log_pi, entropy, self.critic(x)



class PPO(MFRL):
    """
    Algorithm: Proximal Policy Optimization (On-policy, Model-free)

        01. Input: θ, φ                                                         > Initial parameters
        02. for k = 0, 1, 2, ... do
        03.     Collect set of traj's D = {τi} by πk = π(φk)
        04.     Compute RTG Rhat_t
        05.     Compute advantage estimate Ahat_t based on Vk = V(θk)
        06.     Update πφ by maxz Jπ
                    φ = arg max_φ {(1/T|Dk|) sum sum min((π/πk), 1 +- eps) Aπk }
        07.     Fit Vθ by MSE(Jv)
                    θ = arg min_θ {(1/T|Dk|) sum sum (Vθ(st) - RTG)^2 }
        08. end for
    """
    def __init__(self, exp_prefix, configs, seed, device, wb) -> None:
        super(PPO, self).__init__(exp_prefix, configs, seed, device)
        # print('init PPO Algorithm!')
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


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']
        Nx = self.configs['algorithm']['learning']['expl_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        F = self.configs['algorithm']['learning']['train_AC_freq']
        G = self.configs['algorithm']['learning']['grad_AC_steps']

        batch_size = self.configs['data']['batch_size']
        mini_batch_size = self.configs['data']['mini_batch_size']


        global_step = 0
        start_time = time.time()
        o, d, Z, el, t = self.learn_env.reset(), 0, 0, 0, 0
        oldJs = [0, 0]
        JVList, JPiList = [0]*Ni, [0]*Ni
        logs = dict()
        lastEZ, lastES = 0, -2

        start_time_real = time.time()
        for n in range(1, N + 1):
            if self.configs['experiment']['print_logs']:
                print('=' * 80)
                if n > Nx:
                    print(f'\n[ Epoch {n}   Learning ]')
                elif n > Ni:
                    print(f'\n[ Epoch {n}   Exploration + Learning ]')
                else:
                    print(f'\n[ Epoch {n}   Inintial Exploration ]')

            nt = 0
            # self.buffer.reset()
            learn_start_real = time.time()
            while nt < NT:
                # Interaction steps (On-Policy)
                for e in range(1, E+1):
                    print('t: ', t, end='\r')
                    # global_step += 1 #* num_envs
                    o, d, Z, el, t = self.internact_op(n, o, d, Z, el, t)

                    if t % 1000 == 0:
                        # print(f"Training: global_step={global_step}, return={round(Z, 2)}, ep_length={el}")
                        EZ, ES, EL = self.evaluate_op()
                        print(f"Evaluation: global_step={t}, return={round(np.mean(EZ), 2)}, ep_length={np.mean(EL)}")
                        logs['evaluation/episodic_return_mean'] = np.mean(EZ)
                        logs['evaluation/episodic_length_mean'] = np.mean(EL)
                        if self.WandB:
                            wandb.log(logs)

                # with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o))
                # self.buffer.traj_tail(d, v)

                # Taking gradient steps after exploration
                if n > Ni and n % F == 0:
                    # print('updateAC')
                    with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o))
                    self.buffer.traj_tail(d, v)
                    # Optimizing policy and value networks
                    b_inds = np.arange(batch_size)
                    for g in range(1, G+1):
                        # PPO-P >>>>
                        for b in range(0, batch_size, mini_batch_size):
                            # print('ptr: ', self.buffer.ptr)
                            mini_batch = self.buffer.sample_batch(mini_batch_size)
                            Jv, Jpi, stop_pi = self.trainAC(g, mini_batch, oldJs)
                            oldJs = [Jv, Jpi]
                            JVList.append(Jv.item())
                            JPiList.append(Jpi.item())
                        # PPO-P <<<<

                        # np.random.shuffle(b_inds)
                        # for start in range(0, batch_size, mini_batch_size):
                        #     end = start + mini_batch_size
                        #     mb_inds = b_inds[start:end]
                        #     mini_batch = self.buffer.sample_inds(mb_inds)
                        #     self.trainAC(mini_batch)
                    self.buffer.reset()
                nt += E

            # logs['time/training                  '] = time.time() - learn_start_real
            # logs['training/ppo/Jv                '] = np.mean(JVList)
            # logs['training/ppo/Jpi               '] = np.mean(JPiList)
            #
            # eval_start_real = time.time()
            # EZ, ES, EL = self.evaluate_op()
            #
            # logs['time/evaluation                '] = time.time() - eval_start_real
            #
            # if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
            #     logs['evaluation/episodic_score_mean '] = np.mean(ES)
            #     logs['evaluation/episodic_score_std  '] = np.std(ES)
            # else:
            #     logs['evaluation/episodic_return_mean'] = np.mean(EZ)
            #     logs['evaluation/episodic_return_std '] = np.std(EZ)
            # logs['evaluation/episodic_length_mean'] = np.mean(EL)
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
            # # Printing logs
            # if self.configs['experiment']['print_logs']:
            #     for k, v in logs.items():
            #         print(f'{k}  {round(v, 2)}')
            #
            # # WandB
            # if self.WandB:
            #     wandb.log(logs)

        self.learn_env.close()
        self.eval_env.close()


    def trainAC(self, g, batch, oldJs):
        Jv = self.updateV(batch, oldJs[0])
        Jpi, stop_pi = self.updatePi(batch, oldJs[1])
        return Jv, Jpi, stop_pi


    def updateV(self, batch, Jv_old):
        """"
        Jv(θ) =
        """
        max_grad_norm = kl_targ = self.configs['critic']['network']['max_grad_norm']

        observations, _, returns, _, _, _ = batch.values()
        v = self.actor_critic.get_v(observations)

        Jv = 0.5 * ( (v - returns) ** 2 ).mean(axis=0)

        self.actor_critic.critic.optimizer.zero_grad()
        Jv.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), max_grad_norm)
        self.actor_critic.critic.optimizer.step()

        return Jv


    def updatePi(self, batch, Jpi_old, stop_pi=False):
        """
        Jπ(φ) =
        """
        clip_eps = self.configs['actor']['clip_eps']
        kl_targ = self.configs['actor']['kl_targ']
        entropy_coef = self.configs['actor']['entropy_coef']
        max_grad_norm = self.configs['actor']['network']['max_grad_norm']

        observations, actions, _, _, advantages, log_pis_old = batch.values()

        _, log_pi, entropy = self.actor_critic.get_pi(observations, actions)
        logratio = log_pi - log_pis_old
        ratio = logratio.exp()

        with T.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl_old = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()

        # clipped_ratio = T.clamp(ratio, 1-clip_eps, 1+clip_eps)
        # Jpg = - ( T.min(ratio * advantages, clipped_ratio * advantages) ).mean(axis=0)
        # Jentropy = entropy_coef * entropy.mean()
        # Jpi = Jpg + Jentropy

        # if approx_kl_old <= kl_targ:
        clipped_ratio = T.clamp(ratio, 1-clip_eps, 1+clip_eps)
        Jpg = - ( T.min(ratio * advantages, clipped_ratio * advantages) ).mean(axis=0)
        Jentropy = entropy_coef * entropy.mean()
        Jpi = Jpg + Jentropy
        self.actor_critic.actor.optimizer.zero_grad()
        Jpi.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), max_grad_norm)
        self.actor_critic.actor.optimizer.step()
        # else:
        #     # print('Stop policy gradient updates!')
        #     Jpi = Jpi_old
        #     stop_pi = True

        # if approx_kl > 1.5 * kl_targ:
        #     stop = True

        return Jpi, stop_pi





def main(exp_prefix, config, seed, device, wb):

    print('Start an PPO experiment...')
    print('\n')

    configs = config.configurations

    if seed:
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

    alg_name = configs['algorithm']['name']
    env_name = configs['environment']['name']
    env_type = configs['environment']['type']

    group_name = f"{env_name}-{alg_name}"
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
