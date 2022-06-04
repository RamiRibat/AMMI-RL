import gym
from gym.spaces import Box

import numpy as np
import torch as T

import rl.environments
from rl.data.buffer import TrajBuffer, ReplayBuffer
# from rl.data.buffer import TrajBuffer, ReplayBufferNP



# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         print('in thunk')
#         env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env = gym.wrappers.ClipAction(env)
#         env = gym.wrappers.NormalizeObservation(env)
#         env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
#         env = gym.wrappers.NormalizeReward(env)
#         env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env
#
#     return thunk




class MFRL:
    """
    Model-Free Reinforcement Learning
    """
    def __init__(self, exp_prefix, configs, seed, device):
        # super(MFRL, self).__init__(configs, seed)
        # print('init MBRL!')
        self.exp_prefix = exp_prefix
        self.configs = configs
        self.seed = seed
        self._device_ = device


    def _build(self):
        self._set_env()
        self._set_buffer()


    def _set_env(self):
        name = self.configs['environment']['name']
        evaluate = self.configs['algorithm']['evaluation']

        # Inintialize Learning environment
        self.learn_env = gym.make(name)
        self._seed_env(self.learn_env)
        assert isinstance (self.learn_env.action_space, Box), "Works only with continuous action space"

        if evaluate:
            # Ininialize Evaluation environment
            self.eval_env = gym.make(name)
            self._seed_env(self.eval_env)
        else:
            self.eval_env = None

        # Spaces dimensions
        self.obs_dim = self.learn_env.observation_space.shape[0]
        self.act_dim = self.learn_env.action_space.shape[0]
        self.act_up_lim = self.learn_env.action_space.high
        self.act_low_lim = self.learn_env.action_space.low


    def _seed_env(self, env):
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        env.observation_space.seed(self.seed)


    def _set_buffer(self):
        max_size = self.configs['data']['buffer_size']
        device = self._device_
        if self.configs['algorithm']['on-policy']:
            max_size = self.configs['data']['batch_size']
            num_traj = max_size//20
            horizon = 1000
            self.buffer = TrajBuffer(self.obs_dim, self.act_dim, horizon, num_traj, max_size, self.seed, device)
        else:
            self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, max_size, self.seed, device)


    def initialize_buffer(self, num_traj=400):
        # print('Initialize a New Buffer..')
        seed = self.seed
        device = self._device_

        if self.configs['algorithm']['on-policy']:
            # num_traj = 40
            horizon = 1000
            max_size = self.configs['data']['batch_size']
            self.buffer = TrajBuffer(self.obs_dim, self.act_dim, horizon, num_traj, max_size, self.seed, device)


    def initialize_learning(self, NT, Ni):
        max_el = self.configs['environment']['horizon']

        o, Z, el, t = self.learn_env.reset(), 0, 0, 0

        if Ni < 1: return o, Z, el, t

        print(f'[ Initial exploaration ] Starting')
        for ni in range(1, Ni+1):
            print(f'[ Initial exploaration ] Epoch {ni}')
            nt = 0
            while nt < NT:
                # Random actions
                a = self.learn_env.action_space.sample()
                o_next, r, d, info = self.learn_env.step(a)
                d = True if el == max_el else d # Ignore artificial termination

                self.buffer.store_transition(o, a, r, o_next, d)

                o = o_next
                Z += r
                el +=1
                t +=1

                if d or (el == max_el): o, Z, el = self.learn_env.reset(), 0, 0

                nt += 1

        return o, Z, el, t


    def internact_op(self, n, o, d, Z, el, t):
        Nt = self.configs['algorithm']['learning']['epoch_steps']
        max_el = self.configs['environment']['horizon']

        # a = self.actor_critic.get_action_np(o)
        with T.no_grad(): a, log_pi, v = self.actor_critic.get_a_and_v_np(T.Tensor(o))
        # print('log_pi: ', log_pi)

        o_next, r, d_next, _ = self.learn_env.step(a)
        Z += r
        el += 1
        t += 1

        self.buffer.store_transition(o, a, r, d, v, log_pi, el)

        if d_next or (el == max_el):
            # o_next, Z, el = self.learn_env.reset(), 0, 0
            with T.no_grad(): v_next = self.actor_critic.get_v(T.Tensor(o_next)).cpu()
            self.buffer.traj_tail(d_next, v_next, el)

            # print(f'termination: t={t} | el={el} | total_size={self.buffer.total_size()}')

            o_next, d_next, Z, el = self.learn_env.reset(), 0, 0, 0

        o, d = o_next, d_next

        return o, d, Z, el, t


    def internact_opB(self, n, o, Z, el, t):
        Nt = self.configs['algorithm']['learning']['epoch_steps']
        max_el = self.configs['environment']['horizon']

        # a = self.actor_critic.get_action_np(o)
        with T.no_grad(): a, log_pi, v = self.actor_critic.get_a_and_v_np(T.Tensor(o))
        # print('log_pi: ', log_pi)
        o_next, r, d, _ = self.learn_env.step(a)
        Z += r
        el += 1
        t += 1
        self.buffer.store(o, a, r, o_next, v, log_pi, el)
        o = o_next
        if d or (el == max_el):
            if el == max_el:
                with T.no_grad(): v = self.actor_critic.get_v(T.Tensor(o)).cpu()
            else:
                # print('v=0')
                v = T.Tensor([0.0])
            self.buffer.finish_path(el, v)

            # print(f'termination: t={t} | el={el} | total_size={self.buffer.total_size()}')
            o, Z, el = self.learn_env.reset(), 0, 0
        return o, Z, el, t


    def internact(self, n, o, Z, el, t):
        Nx = self.configs['algorithm']['learning']['expl_epochs']
        max_el = self.configs['environment']['horizon']

        if n > Nx:
            a = self.actor_critic.get_action_np(o) # Stochastic action | No reparameterization # Deterministic action | No reparameterization
        else:
            a = self.learn_env.action_space.sample()

        o_next, r, d, _ = self.learn_env.step(a)
        d = False if el == max_el else d # Ignore artificial termination

        self.buffer.store_transition(o, a, r, o_next, d)

        o = o_next
        Z += r
        el +=1
        t +=1

        if d or (el == max_el): o, Z, el = self.learn_env.reset(), 0, 0

        return o, Z, el, t


    def evaluate_op(self):
        evaluate = self.configs['algorithm']['evaluation']
        if evaluate:
            print('[ Evaluation ]')
            EE = self.configs['algorithm']['evaluation']['eval_episodes']
            max_el = self.configs['environment']['horizon']
            EZ = [] # Evaluation episodic return
            ES = [] # Evaluation episodic score
            EL = [] # Evaluation episodic length

            for ee in range(1, EE+1):
                print(f' [ Agent Evaluation ] Episode: {ee}   ', end='\r')
                o, d, Z, S, el = self.eval_env.reset(), False, 0, 0, 0
                while not(d or (el == max_el)):
                    # with T.no_grad(): a, _, _ = self.actor_critic.get_pi(T.Tensor(o))
                    a = self.actor_critic.get_action_np(o)
                    # a = self.actor_critic.get_action_np(o, deterministic=True)
                    o, r, d, info = self.eval_env.step(a)
                    Z += r
                    if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': S += info['score']
                    el += 1

                EZ.append(Z)
                if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': ES.append(S/el)
                EL.append(el)

            # if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
            #     for i in range(len(ES)):
            #         ES[i] /= EL[i]

        return EZ, ES, EL


    def evaluate(self):
        evaluate = self.configs['algorithm']['evaluation']
        if evaluate:
            print('[ Evaluation ]')
            EE = self.configs['algorithm']['evaluation']['eval_episodes']
            max_el = self.configs['environment']['horizon']
            EZ = [] # Evaluation episodic return
            ES = [] # Evaluation episodic score
            EL = [] # Evaluation episodic length

            for ee in range(1, EE+1):
                print(f' [ Agent Evaluation ] Episode: {ee}   ', end='\r')
                o, d, Z, S, el = self.eval_env.reset(), False, 0, 0, 0

                while not(d or (el == max_el)):
                    # Take deterministic actions at evaluation time
                    a = self.actor_critic.get_action_np(o, deterministic=True) # Deterministic action | No reparameterization
                    o, r, d, info = self.eval_env.step(a)
                    Z += r
                    if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': S += info['score']
                    el += 1

                EZ.append(Z)
                if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand': ES.append(S/el)
                EL.append(el)

            # if self.configs['environment']['type'] == 'mujoco-pddm-shadowhand':
            #     for i in range(len(ES)):
            #         ES[i] /= EL[i]

        return EZ, ES, EL
