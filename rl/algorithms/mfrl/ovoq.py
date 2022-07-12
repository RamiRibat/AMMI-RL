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
from rl.control.policy import PPOPolicy, StochasticPolicy, OVOQPolicy





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

        self.actor, self.critic, self.critic_target = None, None, None
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
        return OVOQPolicy(
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
        # Update Q
        return self.oq(o, a)


    def get_q_target(self, o, a):
        # Update Q
        return self.oq_target(o, a)


    def get_pi(self, o, a=None,
               on_policy=True,
               reparameterize=False,
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


    def get_action(self, o, a=None,
                   on_policy=True,
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
        return pi, log_pi, entropy, self.ov(o)


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
        return action.cpu(), log_pi.cpu(), entropy, self.ov(o).cpu()


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
        return a.cpu().numpy(), log_pi.cpu().numpy(), self.ov(o).cpu().numpy()


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






class OVOQ:
# class OVOQ(MFRL):
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
        # super(OVOQ, self).__init__(exp_prefix, configs, seed, device)
        print('Initialize OVOQ Algorithm!')
        self.configs = configs
        self.seed = seed
        self._device_ = device
        self.WandB = wb
        self._build()


    def _build(self):
        # super(OVOQ, self)._build()
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
        pass


    def trainAC(self, g, batch, oldJs, on_policy=True):
        # AUI = self.configs['algorithm']['learning']['alpha_update_interval']
        # PUI = self.configs['algorithm']['learning']['policy_update_interval']
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
            Jpi, PiInfo = self.updatePi(batch, oldJs[3], on_policy=False)# if (g % PUI == 0) else oldJs[2]
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
        max_grad_norm = kl_targ = self.configs['critic-v']['network']['max_grad_norm']

        O, _, _, _, _, Z, _, _, _ = batch.values()
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

        O, A, R, O_next, D = batch.values()

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

        clip_eps = self.configs['actor']['clip_eps']
        entropy_coef = self.configs['actor']['entropy_coef']
        max_grad_norm = self.configs['actor']['network']['max_grad_norm']
        kl_targ = self.configs['actor']['kl_targ']
        max_dev = self.configs['actor']['max_dev']

        # O, A, _, _, _, _, _, U, log_pis_old = batch.values()

        if on_policy:
            O, A, _, _, _, _, _, U, log_pis_old = batch.values()
            _, log_pi, entropy = self.actor_critic.get_pi(O, A)
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


    def updateTarget(self):
        # print('updateTarget')
        tau = self.configs['critic-q']['tau']
        with T.no_grad():
            for p, p_targ in zip(self.actor_critic.oq.parameters(),
                                 self.actor_critic.oq_target.parameters()):
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

    group_name = f"{env_name}-{alg_name}-Mac-A"
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
